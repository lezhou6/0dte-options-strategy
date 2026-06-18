"""Plot the log_return_norm distribution with a fitted normal for one symbol."""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

MODEL_INPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "model_input")
VIZ_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "visualization", "log_return_norm")

TARGET = "log_return_norm"
GROUP_COL = "expiration"  # holds datetime.date objects


def load_target(symbol: str, start_date: str | None, end_date: str | None) -> pd.Series:
    """Load log_return_norm for one symbol, optionally restricted to [start_date, end_date]."""
    path = os.path.join(MODEL_INPUT_DIR, f"{symbol}.parquet")
    df = pd.read_parquet(path)

    # group_col holds datetime.date objects; coerce ISO-string bounds to match.
    if start_date is not None:
        df = df[df[GROUP_COL] >= pd.Timestamp(start_date).date()]
    if end_date is not None:
        df = df[df[GROUP_COL] <= pd.Timestamp(end_date).date()]

    return df[TARGET].dropna()


def plot_distribution(x: pd.Series, symbol: str, start_date: str | None, end_date: str | None) -> str:
    """Histogram of x with the MLE-fitted normal overlaid; returns the saved path."""
    mu, sigma = norm.fit(x)
    print(f"Fitted normal for {symbol}: mu = {mu:.4f}, sigma = {sigma:.4f}  (n = {len(x)})")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(x, bins=100, density=True, alpha=0.6, edgecolor="black", label=TARGET)

    grid = np.linspace(x.min(), x.max(), 200)
    ax.plot(grid, norm.pdf(grid, mu, sigma), "r-", linewidth=2,
            label=f"Fitted N({mu:.3f}, {sigma:.3f}²)")

    span = ""
    if start_date is not None or end_date is not None:
        span = f"  [{start_date or 'start'} → {end_date or 'end'}]"
    ax.set_xlabel(TARGET)
    ax.set_ylabel("Density")
    ax.set_title(f"{symbol} {TARGET} with fitted normal{span}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    os.makedirs(VIZ_DIR, exist_ok=True)
    suffix = ""
    if start_date is not None or end_date is not None:
        suffix = f"_{start_date or 'start'}_{end_date or 'end'}"
    out_path = os.path.join(VIZ_DIR, f"{symbol}{suffix}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", type=str, default="SPY")
    parser.add_argument("--start-date", type=str, default=None, help="ISO YYYY-MM-DD, inclusive")
    parser.add_argument("--end-date", type=str, default=None, help="ISO YYYY-MM-DD, inclusive")
    args = parser.parse_args()

    x = load_target(args.symbol, args.start_date, args.end_date)
    plot_distribution(x, args.symbol, args.start_date, args.end_date)


if __name__ == "__main__":
    main()
