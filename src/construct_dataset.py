"""Construct model input by joining log returns from spy_processed into spy_aggregate."""

import os

import numpy as np
import pandas as pd

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "model_input", "model_input.parquet")

_MIN_PER_YEAR = 365 * 1440  # calendar days × minutes per day


def main() -> None:
    processed = pd.read_parquet(os.path.join(PROCESSED_DIR, "spy_processed.parquet"))
    aggregate = pd.read_parquet(os.path.join(PROCESSED_DIR, "spy_aggregate.parquet"))

    log_returns = (
        processed.groupby("timestamp")[["log_return_from_open", "log_return"]]
        .first()
        .reset_index()
    )

    inconsistent = (
        processed.groupby("timestamp")[["log_return_from_open", "log_return"]]
        .nunique()
    )
    assert (inconsistent > 1).any(axis=None).item() == False, \
        "Inconsistent log_return values found for same timestamp"

    model_input = aggregate.merge(log_returns, on="timestamp", how="left")

    maxoi = (
        processed.sort_values("open_interest", ascending=False)
        .groupby("timestamp")
        .first()
        .reset_index()
        .rename(columns={c: "max_oi_" + c for c in processed.columns if c != "timestamp"})
    )
    model_input = model_input.merge(maxoi, on="timestamp", how="left")

    nan_rows = model_input[model_input["max_oi_strike"].isna()]["timestamp"]
    if not nan_rows.empty:
        print(f"  {len(nan_rows)} timestamps missing max_oi row:")
        for ts in nan_rows:
            print(f"    {ts}")

    ttm_years = model_input["ttm_min"] / _MIN_PER_YEAR
    model_input["log_return_normalized"] = (
        model_input["log_return"] / (model_input["atm_iv"] * np.sqrt(ttm_years))
    )

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    model_input.to_parquet(OUT_PATH, index=False)
    print(f"Saved {len(model_input)} rows to {OUT_PATH}")
    print(f"Columns: {list(model_input.columns)}")


if __name__ == "__main__":
    main()
