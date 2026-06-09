"""Fetch SPY raw greeks and open interest from ThetaData and store as parquet files."""

import argparse
import os
from datetime import date

import pandas as pd
from thetadata import ThetaClient

GREEKS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "greeks")
OI_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "oi")


def _trading_days(end: str, periods: int) -> list[date]:
    end_ts = pd.Timestamp(end)
    if end_ts.day_of_week >= 5:
        end_ts = end_ts - pd.offsets.BDay(1)
        print(f"Warning: {end} is a {pd.Timestamp(end).day_name()}, using {end_ts.date()} instead.")
    return list(pd.bdate_range(end=end_ts, periods=periods).date)


def fetch_greeks(client: ThetaClient, symbol: str, trading_days: list[date]) -> None:
    today = date.today()
    for exp in trading_days:
        if exp >= today:
            print(f"[greeks] Skip {exp}: future date")
            continue
        out_path = os.path.join(GREEKS_DIR, symbol, f"{exp}.parquet")
        if os.path.exists(out_path):
            print(f"[greeks] Skip {exp}: already exists")
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
            print(f"[greeks] Skip {exp}: {e}")
            continue
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df.to_parquet(out_path, index=False)
        print(f"[greeks] Saved {exp}: {len(df)} rows")


def fetch_oi(client: ThetaClient, symbol: str, trading_days: list[date]) -> None:
    today = date.today()
    for exp in trading_days:
        if exp >= today:
            print(f"[oi]     Skip {exp}: future date")
            continue
        out_path = os.path.join(OI_DIR, symbol, f"{exp}.parquet")
        if os.path.exists(out_path):
            print(f"[oi]     Skip {exp}: already exists")
            continue
        try:
            df = client.option_history_open_interest(
                symbol=symbol,
                expiration=exp,
                start_date=exp,
                end_date=exp,
            )
        except Exception as e:
            print(f"[oi]     Skip {exp}: {e}")
            continue
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df.to_parquet(out_path, index=False)
        print(f"[oi]     Saved {exp}: {len(df)} rows")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch SPY raw greeks and OI from ThetaData")
    parser.add_argument("--end", default="2026-05-19", help="End date (YYYY-MM-DD)")
    parser.add_argument("--periods", type=int, default=10, help="Number of trading days")
    parser.add_argument("--symbol", default="SPY", help="Underlying symbol")
    args = parser.parse_args()

    days = _trading_days(args.end, args.periods)
    print(f"Fetching {args.symbol} data for {args.periods} day(s): {days[0]} to {days[-1]}")

    client = ThetaClient(dataframe_type="pandas")
    fetch_greeks(client, args.symbol, days)
    fetch_oi(client, args.symbol, days)


if __name__ == "__main__":
    main()
