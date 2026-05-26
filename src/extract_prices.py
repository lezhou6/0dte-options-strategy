"""Extract SPY opening (09:30) and closing (16:00) prices from greeks parquet files."""

import glob
import os

import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "greeks", "SPY")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


def _extract_prices(hour: int, minute: int, col_name: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(RAW_DIR, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {RAW_DIR}")

    records = []
    for path in files:
        date_str = os.path.splitext(os.path.basename(path))[0]
        df = pd.read_parquet(path, columns=["timestamp", "underlying_price"])
        rows = df[(df["timestamp"].dt.hour == hour) & (df["timestamp"].dt.minute == minute)]
        if rows.empty:
            print(f"Warning: no {hour:02d}:{minute:02d} rows for {date_str}, skipping")
            continue
        price = round(rows["underlying_price"].iloc[0], 2)
        records.append({"date": date_str, col_name: price})

    return pd.DataFrame(records)


def extract_opening_prices() -> pd.DataFrame:
    return _extract_prices(9, 30, "opening_price")


def extract_closing_prices() -> pd.DataFrame:
    return _extract_prices(16, 0, "closing_price")


def main() -> None:
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    for fn, filename in [
        (extract_opening_prices, "spy_opening_prices.csv"),
        (extract_closing_prices, "spy_closing_prices.csv"),
    ]:
        result = fn()
        out_path = os.path.join(PROCESSED_DIR, filename)
        result.to_csv(out_path, index=False)
        print(f"Saved {len(result)} rows to {out_path}")
        print(result.to_string(index=False))


if __name__ == "__main__":
    main()
