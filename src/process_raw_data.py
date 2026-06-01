"""Process raw SPY greeks into a clean, feature-enriched parquet for modelling."""

import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import norm

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "greeks", "SPY")
OI_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "oi", "SPY")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
OUT_PATH = os.path.join(PROCESSED_DIR, "spy_processed.parquet")
AGGREGATE_PATH = os.path.join(PROCESSED_DIR, "spy_aggregate.parquet")


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


def _extract_prices(hour: int, minute: int, col_name: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(RAW_DIR, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {RAW_DIR}")
    records = []
    for path in files:
        date_str = os.path.splitext(os.path.basename(path))[0]
        tmp = pd.read_parquet(path, columns=["timestamp", "underlying_price"])
        rows = tmp[(tmp["timestamp"].dt.hour == hour) & (tmp["timestamp"].dt.minute == minute)]
        if rows.empty:
            print(f"Warning: no {hour:02d}:{minute:02d} rows for {date_str}, skipping")
            continue
        records.append({"date": date_str, col_name: round(rows["underlying_price"].iloc[0], 2)})
    return pd.DataFrame(records)


def extract_prices() -> None:
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    for (hour, minute), filename, col in [
        ((9, 30), "spy_opening_prices.csv", "opening_price"),
        ((16, 0), "spy_closing_prices.csv", "closing_price"),
    ]:
        result = _extract_prices(hour, minute, col)
        result.to_csv(os.path.join(PROCESSED_DIR, filename), index=False)
        print(f"  Saved {len(result)} rows to {filename}")


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


def merge_oi(df: pd.DataFrame) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(OI_DIR, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No OI parquet files found in {OI_DIR}")
    oi = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    oi = oi.drop(columns=["symbol", "timestamp"])
    oi = oi.drop_duplicates(subset=["expiration", "strike", "right"], keep="first")
    oi["expiration"] = pd.to_datetime(oi["expiration"])
    df = df.merge(oi, on=["expiration", "strike", "right"], how="left")
    df["open_interest"] = df["open_interest"].fillna(0).astype(int)
    return df


_MIN_TO_YEAR = 1.0 / (365 * 1440)
_RISK_FREE_RATE = 0.04


def add_oi_features(df: pd.DataFrame) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(OI_DIR, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No OI parquet files found in {OI_DIR}")
    oi = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    oi = oi.drop(columns=["symbol", "timestamp"])
    oi = oi.drop_duplicates(subset=["expiration", "strike", "right"], keep="first")
    oi["expiration"] = pd.to_datetime(oi["expiration"])

    total_oi = (
        oi.groupby(["expiration", "strike"])["open_interest"]
        .sum()
        .rename("total_oi")
        .reset_index()
    )
    put_oi = (
        oi[oi["right"] == "PUT"][["expiration", "strike", "open_interest"]]
        .rename(columns={"open_interest": "put_oi"})
    )
    strike_feats = total_oi.merge(put_oi, on=["expiration", "strike"], how="left")
    strike_feats["put_oi_fraction"] = (
        strike_feats["put_oi"].div(strike_feats["total_oi"]).where(strike_feats["total_oi"] > 0)
    )

    max_oi_strike = (
        strike_feats.groupby("expiration")
        .apply(lambda g: g.loc[g["total_oi"].idxmax(), "strike"])
        .rename("max_oi_strike")
        .reset_index()
    )

    top3 = (
        strike_feats.groupby("expiration")["total_oi"]
        .apply(lambda x: x.nlargest(3).sum())
        .rename("top3_oi_sum")
    )
    day_total = strike_feats.groupby("expiration")["total_oi"].sum().rename("day_total_oi")
    concentration = pd.concat([top3, day_total], axis=1).reset_index()
    concentration["oi_concentration_top3"] = concentration["top3_oi_sum"] / concentration["day_total_oi"]

    strike_feats = (
        strike_feats[["expiration", "strike", "total_oi", "put_oi_fraction"]]
        .merge(max_oi_strike, on="expiration")
        .merge(concentration[["expiration", "oi_concentration_top3"]], on="expiration")
    )

    df = df.merge(strike_feats, on=["expiration", "strike"], how="left")
    df["distance_to_max_oi"] = df["underlying_price"] - df["max_oi_strike"]
    return df


def add_bid_ask_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["bid_ask_mid"] = (df["bid"] + df["ask"]) / 2
    df["bid_ask_spread"] = df["ask"] - df["bid"]
    df["bid_ask_spread_norm"] = df["bid_ask_spread"] / df["bid_ask_mid"]
    return df


def add_exposure(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["dex"] = -df["delta"] * df["open_interest"] * df["underlying_price"] * 100
    df["gex"] = (
        df["gamma"].where(df["right"] == "CALL", -df["gamma"])
        * df["open_interest"]
        * df["underlying_price"] ** 2
        * 100
        * 0.01
    )
    df["tex"] = -df["theta"] * df["open_interest"] * 100
    return df


def calc_atm_iv(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.assign(dist=(df["strike"] - df["underlying_price"]).abs())
    min_dist = df2.groupby("timestamp")["dist"].transform("min")
    return (
        df2[df2["dist"] == min_dist]
        .groupby("timestamp")["implied_vol"]
        .mean()
        .rename("atm_iv")
        .reset_index()
    )


def calc_delta_iv(df: pd.DataFrame, target_delta: float) -> pd.DataFrame:
    calls = df[df["right"] == "CALL"].copy()
    puts = df[df["right"] == "PUT"].copy()

    calls["dist"] = (calls["delta"] - target_delta).abs()
    puts["dist"] = (puts["delta"] - (-target_delta)).abs()

    min_call_dist = calls.groupby("timestamp")["dist"].transform("min")
    min_put_dist = puts.groupby("timestamp")["dist"].transform("min")

    d = int(target_delta * 100)
    call_iv = (
        calls[calls["dist"] == min_call_dist]
        .groupby("timestamp")["implied_vol"]
        .mean()
        .rename(f"iv_call_{d}d")
    )
    put_iv = (
        puts[puts["dist"] == min_put_dist]
        .groupby("timestamp")["implied_vol"]
        .mean()
        .rename(f"iv_put_{d}d")
    )
    return pd.concat([call_iv, put_iv], axis=1).reset_index()


def add_iv_features(
    net: pd.DataFrame,
    df: pd.DataFrame,
    delta_targets: list[float] | None = None,
) -> pd.DataFrame:
    if delta_targets is None:
        delta_targets = [0.25]

    net = net.merge(calc_atm_iv(df), on="timestamp", how="left")

    for target in delta_targets:
        d = int(target * 100)
        net = net.merge(calc_delta_iv(df, target), on="timestamp", how="left")
        net[f"iv_skew_{d}d"] = net[f"iv_put_{d}d"] - net[f"iv_call_{d}d"]
        net[f"iv_smile_curvature_{d}d"] = (
            net[f"iv_call_{d}d"] + net[f"iv_put_{d}d"] - 2 * net["atm_iv"]
        )

    return net


def add_tex_features(net: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    net_tex = (
        df.groupby("timestamp")["tex"]
        .sum()
        .rename("net_tex")
        .reset_index()
    )
    net = net.merge(net_tex, on="timestamp", how="left")
    net["net_tex_norm"] = net["net_tex"] / net["underlying_price"]
    ttm = (
        df.groupby("timestamp")["ttm_min"]
        .first()
        .rename("ttm_min")
        .reset_index()
    )
    net = net.merge(ttm, on="timestamp", how="left")
    net["ttm_hours"] = net["ttm_min"] / 60
    net["theta_decay"] = net["net_tex"] / net["ttm_hours"]
    return net


def calc_net_exposure(df: pd.DataFrame) -> pd.DataFrame:
    net = (
        df.groupby("timestamp")
        .agg(
            net_dex=("dex", "sum"),
            net_gex=("gex", "sum"),
            underlying_price=("underlying_price", "first"),
        )
        .reset_index()
    )
    net["net_gex_norm"] = net["net_gex"] / net["underlying_price"]
    net["net_dex_norm"] = net["net_dex"] / net["underlying_price"]
    return net


def add_gamma(df: pd.DataFrame) -> pd.DataFrame:
    t = df["ttm_min"] * _MIN_TO_YEAR
    sigma = df["implied_vol"]
    valid = (t > 0) & (sigma > 0)
    d1 = pd.Series(np.nan, index=df.index)
    gamma = pd.Series(np.nan, index=df.index)
    d1[valid] = (
        np.log(df.loc[valid, "underlying_price"] / df.loc[valid, "strike"])
        + (_RISK_FREE_RATE + 0.5 * sigma[valid] ** 2) * t[valid]
    ) / (sigma[valid] * np.sqrt(t[valid]))
    gamma[valid] = norm.pdf(d1[valid]) / (
        df.loc[valid, "underlying_price"] * sigma[valid] * np.sqrt(t[valid])
    )
    df = df.copy()
    df["d1"] = d1
    df["gamma"] = gamma
    return df


def main() -> None:
    print("Extracting opening and closing prices...")
    extract_prices()

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

    df = add_gamma(df)
    print(f"  d1 and gamma calculated")

    df = merge_oi(df)
    print(f"  open_interest merged ({df['open_interest'].gt(0).sum()} non-zero rows)")

    df = add_exposure(df)
    print(f"  dex and gex calculated")

    df = add_oi_features(df)
    print(f"  OI features added (total_oi, put_oi_fraction, max_oi_strike, oi_concentration_top3, distance_to_max_oi)")

    df = add_bid_ask_features(df)
    print(f"  bid-ask features added")

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"Saved {len(df)} rows to {OUT_PATH}")
    print(f"Columns: {list(df.columns)}")

    aggregate = calc_net_exposure(df)
    aggregate = add_iv_features(aggregate, df)
    aggregate = add_tex_features(aggregate, df)
    aggregate.to_parquet(AGGREGATE_PATH, index=False)
    print(f"Saved {len(aggregate)} rows to {AGGREGATE_PATH}")
    print(f"Columns: {list(aggregate.columns)}")


if __name__ == "__main__":
    main()
