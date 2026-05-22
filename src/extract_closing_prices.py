"""Extract SPY closing price (underlying_price at 16:00 ET) from greeks parquet files."""

import glob
import os

import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "greeks", "SPY")
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "spy_closing_prices.csv")


def extract_closing_prices() -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(RAW_DIR, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {RAW_DIR}")

    records = []
    for path in files:
        date_str = os.path.splitext(os.path.basename(path))[0]
        df = pd.read_parquet(path, columns=["timestamp", "underlying_price"])
        close_rows = df[
            (df["timestamp"].dt.hour == 16) & (df["timestamp"].dt.minute == 0)
        ]
        if close_rows.empty:
            print(f"Warning: no 16:00 rows for {date_str}, skipping")
            continue
        price = round(close_rows["underlying_price"].iloc[0], 2)
        records.append({"date": date_str, "closing_price": price})

    return pd.DataFrame(records)


def main() -> None:
    result = extract_closing_prices()
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    result.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(result)} rows to {OUT_PATH}")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
