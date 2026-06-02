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

    ttm_years = model_input["ttm_min"] / _MIN_PER_YEAR
    model_input["log_return_norm"] = (
        model_input["log_return"] / (model_input["atm_iv"] * np.sqrt(ttm_years))
    )

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    model_input.to_parquet(OUT_PATH, index=False)
    print(f"Saved {len(model_input)} rows to {OUT_PATH}")
    print(f"Columns: {list(model_input.columns)}")


if __name__ == "__main__":
    main()
