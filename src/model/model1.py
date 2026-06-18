"""Model 1 — Black-Scholes analytic baseline.

Parameter-free quantile model: under GBM the EOD log-return is
Normal(mu*T - 0.5*sigma**2*T, sigma*sqrt(T)). Since the data's norm_factor
equals sigma*sqrt(T), the normalized target log_return_norm = log_return / norm_factor
is Normal(mean, 1) with mean = mu*T/norm_factor - 0.5*norm_factor, so the q-th
quantile is simply mean + Phi^{-1}(q). No fitting and no features.
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import mean_pinball_loss

MODEL_INPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "model_input")
PRED_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "predictions")

TARGET = "log_return_norm"
GROUP_COL = "expiration"        # holds datetime.date objects
NORM_FACTOR_COL = "norm_factor"
TTM_COL = "ttm_min"

QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)
MU = 0.04                       # assumed annual drift (matches 10_train_v1)
MIN_PER_YEAR = 365 * 1440       # calendar-minute year (matches 10_train_v1)


def load_data(symbol: str, start_date: str | None, end_date: str | None) -> pd.DataFrame:
    """Load one symbol's model input, optionally restricted to [start_date, end_date]."""
    path = os.path.join(MODEL_INPUT_DIR, f"{symbol}.parquet")
    df = pd.read_parquet(path)

    # group_col holds datetime.date objects; coerce ISO-string bounds to match.
    if start_date is not None:
        df = df[df[GROUP_COL] >= pd.Timestamp(start_date).date()]
    if end_date is not None:
        df = df[df[GROUP_COL] <= pd.Timestamp(end_date).date()]

    assert (df[NORM_FACTOR_COL] > 0).all(), "norm_factor must be > 0 to de-normalize"
    return df.sort_values([GROUP_COL, "timestamp"]).reset_index(drop=True)


def predict(df: pd.DataFrame) -> np.ndarray:
    """Closed-form Black-Scholes quantiles in normalized target space. Shape [n, n_q]."""
    ttm_years = df[TTM_COL].to_numpy(dtype=float) / MIN_PER_YEAR
    nf = df[NORM_FACTOR_COL].to_numpy(dtype=float)
    mean = MU * ttm_years / nf - 0.5 * nf
    z = norm.ppf(np.array(QUANTILES))
    return mean[:, None] + z[None, :]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", type=str, default="SPY")
    parser.add_argument("--start-date", type=str, default="2025-01-01", help="ISO YYYY-MM-DD, inclusive")
    parser.add_argument("--end-date", type=str, default="2026-06-01", help="ISO YYYY-MM-DD, inclusive")
    args = parser.parse_args()

    df = load_data(args.symbol, args.start_date, args.end_date)
    preds = predict(df)
    y_true = df[TARGET].to_numpy(dtype=float)
    nf = df[NORM_FACTOR_COL].to_numpy(dtype=float)

    # Assemble prediction frame: normalized + de-normalized log-return quantiles.
    out = df[[GROUP_COL, "timestamp", NORM_FACTOR_COL]].copy()
    out["realized_norm"] = y_true
    out["realized_log_return"] = y_true * nf
    for j, q in enumerate(QUANTILES):
        out[f"q{int(q * 100)}_norm"] = preds[:, j]
        out[f"q{int(q * 100)}_lr"] = preds[:, j] * nf

    os.makedirs(PRED_DIR, exist_ok=True)
    out_path = os.path.join(PRED_DIR, f"{args.symbol}_model1.parquet")
    out.to_parquet(out_path, index=False)

    print(f"Model 1 (Black-Scholes) — {args.symbol}  [{args.start_date} -> {args.end_date}]")
    print(f"  {len(df)} rows ({df[GROUP_COL].nunique()} days)")
    print("  Pinball loss (normalized space):")
    losses = []
    for j, q in enumerate(QUANTILES):
        loss = mean_pinball_loss(y_true, preds[:, j], alpha=q)
        losses.append(loss)
        print(f"    q={q:.2f}  pinball={loss:.4f}")
    print(f"    avg     pinball={np.mean(losses):.4f}")
    print(f"  Saved predictions to {out_path}")


if __name__ == "__main__":
    main()
