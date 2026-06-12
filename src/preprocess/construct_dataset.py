"""Construct model input by joining log returns from {symbol}_processed into {symbol}_aggregate."""

import argparse
import os

import numpy as np
import pandas as pd

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "model_input")

_MIN_PER_YEAR = 365 * 1440  # calendar days × minutes per day



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default="SPY")
    args = parser.parse_args()
    sym = args.symbol.lower()
    processed = pd.read_parquet(os.path.join(PROCESSED_DIR, f"{sym}_processed.parquet"))
    aggregate = pd.read_parquet(os.path.join(PROCESSED_DIR, f"{sym}_aggregate.parquet"))

    log_returns = (
        processed.groupby("timestamp")[["log_return_from_open", "log_return"]]
        .first()
        .reset_index()
    )

    inconsistent = (
        processed.groupby("timestamp")[["log_return_from_open", "log_return", "distance_to_max_oi", "oi_concentration_top3"]]
        .nunique()
    )
    assert (inconsistent > 1).any(axis=None).item() == False, \
        "Inconsistent log_return / distance_to_max_oi / oi_concentration_top3 values found for same timestamp"

    dist = (
        processed.groupby("timestamp")["distance_to_max_oi"]
        .first()
        .reset_index()
    )

    oi_concentration = (
        processed.groupby("timestamp")["oi_concentration_top3"]
        .first()
        .reset_index()
    )

    total_oi = processed.groupby("timestamp")["open_interest"].sum()
    put_oi = processed[processed["right"] == "PUT"].groupby("timestamp")["open_interest"].sum()
    put_oi_fraction = (put_oi / total_oi).rename("put_oi_fraction").reset_index()

    model_input = aggregate.merge(log_returns, on="timestamp", how="left")
    model_input = model_input.merge(dist, on="timestamp", how="left")
    model_input = model_input.merge(oi_concentration, on="timestamp", how="left")
    model_input = model_input.merge(put_oi_fraction, on="timestamp", how="left")

    atm = processed.copy()
    atm["_dist"] = (atm["strike"] - atm["underlying_price"]).abs()
    atm_row = atm.sort_values("_dist").groupby("timestamp").first().reset_index()
    atm_row["atm_spread_norm"] = (atm_row["ask"] - atm_row["bid"]) * 2 / (atm_row["bid"] + atm_row["ask"])
    model_input = model_input.merge(atm_row[["timestamp", "atm_spread_norm"]], on="timestamp", how="left")

    for side, col in [("PUT", "put_max_oi_strike"), ("CALL", "call_max_oi_strike")]:
        side_strike = (
            processed[processed["right"] == side]
            .sort_values("open_interest", ascending=False)
            .groupby("timestamp")["strike"]
            .first()
            .rename(col)
            .reset_index()
        )
        model_input = model_input.merge(side_strike, on="timestamp", how="left")

    ttm_years = model_input["ttm_min"] / _MIN_PER_YEAR
    model_input["norm_factor"] = model_input["atm_iv"] * np.sqrt(ttm_years)
    model_input["log_return_norm"] = model_input["log_return"] / model_input["norm_factor"]

    for col, norm_col in [
        ("distance_to_max_oi", "distance_to_max_oi_norm"),
        ("put_max_oi_strike", "put_max_oi_strike_norm"),
        ("call_max_oi_strike", "call_max_oi_strike_norm"),
    ]:
        model_input[norm_col] = model_input[col] / model_input["underlying_price"]
        model_input = model_input.drop(columns=[col])

    model_input.insert(0, "expiration", model_input["timestamp"].dt.date)

    model_input = model_input.drop(columns=["net_dex", "net_gex", "net_tex", "ttm_hours", "log_return"])

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{args.symbol}.parquet")
    model_input.to_parquet(path, index=False)
    print(f"Saved {len(model_input)} rows ({model_input['expiration'].nunique()} days) to {path}")
    print(f"Columns: {list(model_input.columns)}")


if __name__ == "__main__":
    main()
