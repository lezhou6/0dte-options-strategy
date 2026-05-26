"""Process raw SPY greeks into a clean, feature-enriched parquet for modelling."""

import glob
import os

import numpy as np
import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "greeks", "SPY")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
OUT_PATH = os.path.join(PROCESSED_DIR, "spy_processed.parquet")


def load_raw() -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(RAW_DIR, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {RAW_DIR}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def apply_filter(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        (df["implied_vol"] > 0)
        & (df["iv_error"] < 1.0)
        & (df["delta"].abs().between(0.01, 0.99))
        & (df["bid"] > 0)
    ].copy()


def filter_open_close(df: pd.DataFrame) -> pd.DataFrame:
    """Remove the 9:30 open and 16:00 close snapshots."""
    df = df[df["timestamp"].dt.hour != 16]
    df = df[(df["timestamp"].dt.hour != 9) | (df["timestamp"].dt.minute != 30)]
    return df


def merge_daily_price(df: pd.DataFrame, csv_path: str, price_col: str) -> pd.DataFrame:
    prices = pd.read_csv(csv_path)
    prices["date"] = pd.to_datetime(prices["date"])
    df["expiration"] = pd.to_datetime(df["expiration"])
    df = df.merge(
        prices.rename(columns={"date": "expiration", prices.columns[1]: price_col}),
        on="expiration",
        how="left",
    )
    assert df.groupby("expiration")[price_col].nunique().eq(1).all(), \
        f"Some expirations have multiple values for {price_col}"
    return df


def main() -> None:
    print("Loading raw data...")
    raw = load_raw()
    print(f"  {len(raw)} rows loaded")

    df = apply_filter(raw)
    print(f"  {len(df)} rows after basic filter")

    df = filter_open_close(df)
    print(f"  {len(df)} rows after removing 9:30 / 16:00 snapshots")

    closing_csv = os.path.join(PROCESSED_DIR, "spy_closing_prices.csv")
    df = merge_daily_price(df, closing_csv, "spy_close")
    print(f"  spy_close merged and verified")

    opening_csv = os.path.join(PROCESSED_DIR, "spy_opening_prices.csv")
    df = merge_daily_price(df, opening_csv, "spy_open")
    print(f"  spy_open merged and verified")

    df["log_return_from_open"] = np.log(df["underlying_price"] / df["spy_open"])

    close_dt = df["timestamp"].apply(
        lambda t: t.replace(hour=16, minute=0, second=0, microsecond=0)
    )
    df["ttm_min"] = (close_dt - df["timestamp"]).dt.total_seconds() / 60

    df["log_return"] = np.log(df["spy_close"] / df["underlying_price"])

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"Saved {len(df)} rows to {OUT_PATH}")
    print(f"Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
