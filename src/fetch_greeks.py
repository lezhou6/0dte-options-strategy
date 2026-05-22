"""Fetch SPY raw greeks from ThetaData and store as parquet files."""

import argparse
import os
from datetime import date

import pandas as pd
from thetadata import ThetaClient

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "greeks")


def fetch_greeks(symbol: str, end: str, periods: int) -> None:
    client = ThetaClient(dataframe_type="pandas")
    end_ts = pd.Timestamp(end)
    if end_ts.day_of_week >= 5:
        end_ts = end_ts - pd.offsets.BDay(1)
        print(f"Warning: {end} is a {pd.Timestamp(end).day_name()}, using {end_ts.date()} instead.")

    trading_days = pd.bdate_range(end=end_ts, periods=periods).date
    print(f"Fetching {symbol} greeks for {periods} day(s): {trading_days[0]} to {trading_days[-1]}")

    today = date.today()
    for exp in trading_days:
        if exp >= today:
            print(f"Skip {exp}: future date")
            continue

        out_path = os.path.join(DATA_DIR, symbol, f"{exp}.parquet")
        if os.path.exists(out_path):
            print(f"Skip {exp}: already exists")
            continue

        try:
            df = client.option_history_greeks_first_order(
                symbol=symbol,
                expiration=exp,
                start_date=exp,
                end_date=exp,
                interval="15m",
            )
        except Exception as e:
            print(f"Skip {exp}: {e}")
            continue

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df.to_parquet(out_path, index=False)
        print(f"Saved {exp}: {len(df)} rows")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch SPY raw greeks from ThetaData")
    parser.add_argument("--end", default="2026-05-19", help="End date (YYYY-MM-DD)")
    parser.add_argument("--periods", type=int, default=10, help="Number of trading days")
    args = parser.parse_args()

    fetch_greeks("SPY", args.end, args.periods)


if __name__ == "__main__":
    main()
