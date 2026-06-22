"""Model 2 — walk-forward LightGBM quantile regression for the 0DTE EOD close.

Configurable rolling/expanding walk-forward harness (ported from 11_train_v2.ipynb) that
predicts quantiles of the end-of-day close via the normalized target
`log_return_norm = log_return / norm_factor`. One LightGBM regressor per quantile, refit
per fold and early-stopped on the most-recent days of the window. Day-level splitting keeps
every trading day atomic (never split across train/val/test) and strictly temporal.

The run is compared against the Black-Scholes analytic model (closed-form GBM quantiles,
parameter-free) on the identical pooled test rows, and an HTML report with a fan-chart
visualization is written to data/model_report.

test_days and stride_days are always 1.
"""

from __future__ import annotations

import argparse
import base64
import io
import logging
import os
import random
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Sequence

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm as _norm

# LightGBM is fed numpy arrays; silence sklearn's benign feature-name mismatch warning.
warnings.filterwarnings("ignore", message="X does not have valid feature names")

BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
MODEL_INPUT_DIR = os.path.join(BASE_DIR, "data", "model_input")
PRED_DIR = os.path.join(BASE_DIR, "data", "predictions")
REPORT_DIR = os.path.join(BASE_DIR, "data", "model_report")
LOG_DIR = os.path.join(BASE_DIR, "logs", "train")

DEFAULT_FEATURES: tuple[str, ...] = (
    "net_gex_norm", "net_dex_norm", "net_tex_norm",
    "atm_iv", "iv_call_25d", "iv_put_25d", "iv_skew_25d", "iv_smile_curvature_25d",
    "ttm_min", "theta_decay", "log_return_from_open",
    "oi_concentration_top3", "put_oi_fraction", "atm_spread_norm",
    "distance_to_max_oi_norm", "put_max_oi_strike_norm", "call_max_oi_strike_norm",
)

BS_MU = 0.04                 # assumed annual drift (matches model1 / 10_train_v1)
MIN_PER_YEAR = 365 * 1440    # calendar-minute year


def _default_lgbm_params() -> dict:
    """LightGBM quantile defaults. NaNs are handled natively; deterministic for repro."""
    return {
        "n_estimators": 2000,
        "learning_rate": 0.03,
        "num_leaves": 15,
        "min_child_samples": 100,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "verbosity": -1,
        "deterministic": True,
        "force_row_wise": True,
    }


@dataclass
class Config:
    """Methodological knobs for one walk-forward experiment."""

    symbol: str = "SPY"
    feature_cols: tuple[str, ...] = DEFAULT_FEATURES
    target_col: str = "log_return_norm"
    group_col: str = "expiration"          # trading-day key (0DTE -> expiration == trade date)
    norm_factor_col: str = "norm_factor"
    underlying_col: str = "underlying_price"
    ttm_col: str = "ttm_min"

    quantiles: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9)

    window_mode: str = "rolling"           # "rolling" | "expanding"
    train_window_days: int = 30            # total days in the window (fit + val)
    val_days: int = 1                      # most-recent days of the window held out for val
    test_days: int = 1                     # always 1
    stride_days: int = 1                   # always 1

    start_date: str | None = "2026-03-31"
    end_date: str | None = "2026-06-01"

    seed: int = 42
    early_stopping_rounds: int = 50
    model_params: dict = field(default_factory=_default_lgbm_params)

    def __post_init__(self) -> None:
        assert self.window_mode in ("rolling", "expanding"), self.window_mode
        assert 0 < self.val_days < self.train_window_days, \
            "need 0 < val_days < train_window_days (val is carved from the window)"
        assert self.test_days >= 1 and self.stride_days >= 1
        assert all(0.0 < q < 1.0 for q in self.quantiles)
        assert list(self.quantiles) == sorted(self.quantiles), "quantiles must be ascending"
        leaked = [c for c in self.feature_cols
                  if c in {self.group_col, self.underlying_col, self.norm_factor_col, self.target_col}]
        assert not leaked, f"metadata/target leaked into feature_cols: {leaked}"

    @property
    def data_path(self) -> str:
        return os.path.join(MODEL_INPUT_DIR, f"{self.symbol}.parquet")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


# ── data ────────────────────────────────────────────────────────────────────────────────
def load_dataset(cfg: Config) -> pd.DataFrame:
    """Load the model-input parquet, validate column groups, apply optional date subset."""
    df = pd.read_parquet(cfg.data_path)

    missing = [c for c in cfg.feature_cols if c not in df.columns]
    assert not missing, f"feature_cols missing from data: {missing}"
    for col in (cfg.group_col, cfg.target_col, cfg.norm_factor_col, cfg.underlying_col, cfg.ttm_col):
        assert col in df.columns, f"required column missing: {col}"
    assert (df[cfg.norm_factor_col] > 0).all(), "norm_factor must be > 0 to de-normalize"

    df = df.sort_values([cfg.group_col, "timestamp"]).reset_index(drop=True)
    # group_col holds datetime.date objects; coerce ISO-string bounds to match.
    if cfg.start_date is not None:
        df = df[df[cfg.group_col] >= pd.Timestamp(cfg.start_date).date()]
    if cfg.end_date is not None:
        df = df[df[cfg.group_col] <= pd.Timestamp(cfg.end_date).date()]
    return df.reset_index(drop=True)


# ── folds ───────────────────────────────────────────────────────────────────────────────
@dataclass
class Fold:
    idx: int
    train_days: list
    val_days: list
    test_days: list


def generate_folds(days: Sequence, cfg: Config) -> list[Fold]:
    """Build the ordered list of walk-forward folds. Days must be sorted & unique."""
    days = list(days)
    n = len(days)
    folds: list[Fold] = []
    s = cfg.train_window_days  # warm-up: full window of history must precede the first test
    idx = 0
    while s + cfg.test_days <= n:
        val_block = days[s - cfg.val_days:s]
        if cfg.window_mode == "rolling":
            train_block = days[s - cfg.train_window_days:s - cfg.val_days]
        else:  # expanding
            train_block = days[0:s - cfg.val_days]
        test_block = days[s:s + cfg.test_days]
        assert max(train_block) < min(val_block) < min(test_block)
        assert not (set(train_block) & set(val_block) & set(test_block))
        folds.append(Fold(idx, train_block, val_block, test_block))
        idx += 1
        s += cfg.stride_days
    return folds


def build_xy(df: pd.DataFrame, days: Sequence, cfg: Config):
    """Return (X, y, frame) for the rows whose group_col is in `days`."""
    sub = df[df[cfg.group_col].isin(days)]
    X = sub[list(cfg.feature_cols)].to_numpy(dtype=float)
    y = sub[cfg.target_col].to_numpy(dtype=float)
    return X, y, sub


# ── model ───────────────────────────────────────────────────────────────────────────────
class LGBMQuantileModel:
    """Independent LightGBM quantile regressors, one per quantile level."""

    def __init__(self, cfg: Config) -> None:
        self.quantiles = list(cfg.quantiles)
        self.params = dict(cfg.model_params)
        self.seed = cfg.seed
        self.early_stopping_rounds = cfg.early_stopping_rounds
        self.models_: dict[float, lgb.LGBMRegressor] = {}

    def fit(self, X_train, y_train, X_val, y_val) -> "LGBMQuantileModel":
        for q in self.quantiles:
            model = lgb.LGBMRegressor(
                objective="quantile", alpha=q, random_state=self.seed, **self.params
            )
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(self.early_stopping_rounds, verbose=False),
                           lgb.log_evaluation(0)],
            )
            self.models_[q] = model
        return self

    def predict_quantiles(self, X) -> np.ndarray:
        preds = np.column_stack([self.models_[q].predict(X) for q in self.quantiles])
        preds.sort(axis=1)  # enforce monotonicity: no quantile crossing
        return preds


# ── metrics ─────────────────────────────────────────────────────────────────────────────
def pinball_loss(y_true: np.ndarray, preds: np.ndarray, quantiles: Sequence[float]):
    """Per-quantile pinball loss and its mean. preds shape [n, n_q]."""
    per_q = np.empty(len(quantiles))
    for j, q in enumerate(quantiles):
        d = y_true - preds[:, j]
        per_q[j] = np.mean(np.maximum(q * d, (q - 1.0) * d))
    return per_q, float(per_q.mean())


def coverage(y_true: np.ndarray, preds: np.ndarray, quantiles: Sequence[float]) -> np.ndarray:
    """Fraction of realized targets <= each predicted quantile (should approx the level)."""
    return np.array([(y_true <= preds[:, j]).mean() for j in range(len(quantiles))])


# ── Black-Scholes analytic model ────────────────────────────────────────────────────────
def bs_predict_quantiles(frame: pd.DataFrame, cfg: Config) -> np.ndarray:
    """Closed-form Black-Scholes (GBM) quantiles in NORMALIZED target space.

    Under GBM the EOD log-return ~ Normal(mu*T - 0.5*sigma**2*T, sigma*sqrt(T)). Since
    norm_factor == sigma*sqrt(T), the normalized target is Normal(mean, 1) with
    mean = mu*T/norm_factor - 0.5*norm_factor, and the q-th quantile is mean + Phi^{-1}(q).
    """
    ttm_years = frame[cfg.ttm_col].to_numpy(dtype=float) / MIN_PER_YEAR
    nf = frame[cfg.norm_factor_col].to_numpy(dtype=float)
    mean = BS_MU * ttm_years / nf - 0.5 * nf
    z = _norm.ppf(np.array(cfg.quantiles))
    return mean[:, None] + z[None, :]


# ── walk-forward driver ─────────────────────────────────────────────────────────────────
def run_walk_forward(cfg: Config, df: pd.DataFrame, log: logging.Logger) -> dict:
    """Run the full walk-forward LightGBM quantile harness, logging per-fold train info."""
    set_seed(cfg.seed)
    days = sorted(df[cfg.group_col].unique())
    folds = generate_folds(days, cfg)
    assert folds, "no folds emitted — not enough history for train_window_days + test_days"

    q = list(cfg.quantiles)
    qpct = [int(round(ql * 100)) for ql in q]
    fold_rows: list[dict] = []
    pred_frames: list[pd.DataFrame] = []
    pool_y, pool_pred, pool_nf = [], [], []

    log.info("Walk-forward: %d folds over %d trading days (%s -> %s)",
             len(folds), len(days), days[0], days[-1])

    for fold in folds:
        X_tr, y_tr, _ = build_xy(df, fold.train_days, cfg)
        X_va, y_va, _ = build_xy(df, fold.val_days, cfg)
        X_te, y_te, te = build_xy(df, fold.test_days, cfg)

        model = LGBMQuantileModel(cfg).fit(X_tr, y_tr, X_va, y_va)
        best_iters = [int(model.models_[ql].best_iteration_ or cfg.model_params["n_estimators"])
                      for ql in q]

        pred_norm = model.predict_quantiles(X_te)
        nf = te[cfg.norm_factor_col].to_numpy()
        pred_lr = pred_norm * nf[:, None]
        y_lr = y_te * nf

        pool_y.append(y_te); pool_pred.append(pred_norm); pool_nf.append(nf)

        _, pl_norm_m = pinball_loss(y_te, pred_norm, q)
        _, pl_lr_m = pinball_loss(y_lr, pred_lr, q)
        cov = coverage(y_te, pred_norm, q)
        fold_rows.append({
            "fold": fold.idx,
            "test_day": str(fold.test_days[0]),
            "n_test_rows": len(y_te),
            "pinball_norm": pl_norm_m,
            "pinball_lr": pl_lr_m,
            **{f"cov@{ql}": cov[j] for j, ql in enumerate(q)},
        })

        log.info(
            "  fold %3d | train %s..%s (%dd) val %s..%s (%dd) test %s (%d rows) | "
            "pinball_norm=%.5f best_iter=%s",
            fold.idx, fold.train_days[0], fold.train_days[-1], len(fold.train_days),
            fold.val_days[0], fold.val_days[-1], len(fold.val_days),
            fold.test_days[0], len(y_te), pl_norm_m, best_iters,
        )

        spot = te[cfg.underlying_col].to_numpy(dtype=float)
        rec = te[[cfg.group_col, "timestamp", cfg.underlying_col, cfg.norm_factor_col]].copy()
        rec["realized_norm"] = y_te
        rec["realized_log_return"] = y_lr
        rec["true_close"] = spot * np.exp(y_lr)
        for j, p in enumerate(qpct):
            rec[f"q{p}_norm"] = pred_norm[:, j]
            rec[f"q{p}_lr"] = pred_lr[:, j]
            rec[f"q{p}_close"] = spot * np.exp(pred_lr[:, j])
        pred_frames.append(rec)

    # aggregate across ALL pooled test rows
    Y = np.concatenate(pool_y)
    P = np.vstack(pool_pred)
    NF = np.concatenate(pool_nf)
    agg_pl_norm, agg_pl_norm_m = pinball_loss(Y, P, q)
    agg_pl_lr, agg_pl_lr_m = pinball_loss(Y * NF, P * NF[:, None], q)
    agg_cov = coverage(Y, P, q)

    test_days_used = sorted({d for f in folds for d in f.test_days})
    summary = pd.DataFrame({
        "quantile": q,
        "pinball_norm": agg_pl_norm,
        "pinball_lr": agg_pl_lr,
        "coverage": agg_cov,
        "coverage_error": agg_cov - np.array(q),
    })

    return {
        "folds": pd.DataFrame(fold_rows),
        "summary": summary,
        "predictions": pd.concat(pred_frames, ignore_index=True),
        "mean_pinball_norm": agg_pl_norm_m,
        "mean_pinball_lr": agg_pl_lr_m,
        "n_test_days": len(test_days_used),
        "test_days": test_days_used,
    }


def run_bs_model(cfg: Config, df: pd.DataFrame, folds_test_days: Sequence) -> dict:
    """Evaluate the Black-Scholes analytic model on EXACTLY the walk-forward test rows."""
    q = list(cfg.quantiles)
    qpct = [int(round(ql * 100)) for ql in q]
    _, y_te, te = build_xy(df, folds_test_days, cfg)

    pred_norm = bs_predict_quantiles(te, cfg)
    nf = te[cfg.norm_factor_col].to_numpy()
    pred_lr = pred_norm * nf[:, None]

    agg_pl_norm, agg_pl_norm_m = pinball_loss(y_te, pred_norm, q)
    agg_pl_lr, agg_pl_lr_m = pinball_loss(y_te * nf, pred_lr, q)
    agg_cov = coverage(y_te, pred_norm, q)

    spot = te[cfg.underlying_col].to_numpy(dtype=float)
    rec = te[[cfg.group_col, "timestamp", cfg.underlying_col, cfg.norm_factor_col]].copy()
    for j, p in enumerate(qpct):
        rec[f"q{p}_close"] = spot * np.exp(pred_lr[:, j])

    summary = pd.DataFrame({
        "quantile": q,
        "pinball_norm": agg_pl_norm,
        "pinball_lr": agg_pl_lr,
        "coverage": agg_cov,
        "coverage_error": agg_cov - np.array(q),
    })
    return {
        "summary": summary,
        "bs_close": rec,
        "mean_pinball_norm": agg_pl_norm_m,
        "mean_pinball_lr": agg_pl_lr_m,
    }


# ── report ──────────────────────────────────────────────────────────────────────────────
def _fan_png(model_preds: pd.DataFrame, bs_close: pd.DataFrame, cfg: Config) -> str:
    """Two-panel fan chart (model vs Black-Scholes) of predicted close vs true close.

    One snapshot per test day (nearest noon). Returns a base64-encoded PNG for HTML embed.
    """
    q = list(cfg.quantiles)
    qpct = [int(round(ql * 100)) for ql in q]
    nq = len(q)
    mid = nq // 2

    model_preds = model_preds.drop_duplicates(subset=[cfg.group_col, "timestamp"])
    bs_cols = [f"q{p}_close" for p in qpct]
    merged = model_preds.merge(
        bs_close[[cfg.group_col, "timestamp"] + bs_cols].rename(
            columns={f"q{p}_close": f"bs_q{p}_close" for p in qpct}),
        on=[cfg.group_col, "timestamp"], how="left",
    )

    ts = pd.to_datetime(merged["timestamp"])
    merged = merged.assign(_dist=(ts.dt.hour * 60 + ts.dt.minute - 12 * 60).abs().to_numpy())
    daily = (merged.sort_values([cfg.group_col, "_dist"])
                   .groupby(cfg.group_col, as_index=False)
                   .head(1)
                   .sort_values(cfg.group_col))
    x = pd.to_datetime(daily[cfg.group_col])

    def fan(ax, prefix, title):
        ax.fill_between(x, daily[f"{prefix}{qpct[0]}_close"], daily[f"{prefix}{qpct[-1]}_close"],
                        color="steelblue", alpha=0.20, label=f"q{q[0]}–q{q[-1]}")
        if nq >= 4:
            ax.fill_between(x, daily[f"{prefix}{qpct[1]}_close"], daily[f"{prefix}{qpct[-2]}_close"],
                            color="steelblue", alpha=0.35, label=f"q{q[1]}–q{q[-2]}")
        ax.plot(x, daily[f"{prefix}{qpct[mid]}_close"], color="steelblue", linestyle="--",
                linewidth=1, label=f"predicted median (q{q[mid]})")
        ax.plot(x, daily["true_close"], color="black", marker="o", markersize=3,
                linewidth=1, label="true close")
        ax.set_ylabel("Closing price ($)")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig, axes = plt.subplots(2, 1, figsize=(13, 10), sharex=True)
    fan(axes[0], "q",
        f"{cfg.symbol} Model 2 (LightGBM quantile) — predicted close at noon vs true close "
        f"[{len(daily)} test days]")
    fan(axes[1], "bs_q",
        f"{cfg.symbol} Black-Scholes analytic model — predicted close at noon vs true close "
        f"[{len(daily)} test days]")
    axes[-1].set_xlabel("Date")
    fig.autofmt_xdate()
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _comparison_table_html(model_res: dict, bs_res: dict, cfg: Config) -> str:
    q = list(cfg.quantiles)
    mrows = model_res["summary"]
    brows = bs_res["summary"]
    body = ""
    for i, ql in enumerate(q):
        mp = mrows["pinball_norm"].iloc[i]
        bp = brows["pinball_norm"].iloc[i]
        win = "✓" if mp < bp else ""
        body += (
            f"<tr><td>{ql}</td>"
            f"<td>{mp:.5f}</td><td>{bp:.5f}</td><td class='win'>{win}</td>"
            f"<td>{mrows['coverage'].iloc[i]:.3f}</td><td>{brows['coverage'].iloc[i]:.3f}</td>"
            f"<td>{mrows['coverage_error'].iloc[i]:+.3f}</td></tr>"
        )
    return body


def generate_report(cfg: Config, model_res: dict, bs_res: dict,
                    start: str, end: str, log_path: str) -> str:
    """Write the HTML model-vs-Black-Scholes comparison report with the fan-chart viz."""
    img_b64 = _fan_png(model_res["predictions"], bs_res["bs_close"], cfg)

    m_norm = model_res["mean_pinball_norm"]
    b_norm = bs_res["mean_pinball_norm"]
    gain = (1 - m_norm / b_norm) * 100
    rows = _comparison_table_html(model_res, bs_res, cfg)
    verdict = ("Model 2 beats the Black-Scholes analytic model"
               if m_norm < b_norm else
               "Model 2 does NOT beat the Black-Scholes analytic model")

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{cfg.symbol} — Model 2 vs Black-Scholes</title>
<style>
  body {{ background:#131722; color:#d1d4dc;
         font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
         margin:0; padding:24px 32px; }}
  h1 {{ font-size:1.4rem; border-bottom:1px solid #2a2e39; padding-bottom:12px;
        margin-bottom:8px; letter-spacing:.02em; }}
  h2 {{ color:#9598a1; margin-top:40px; margin-bottom:8px; font-size:.8rem;
        text-transform:uppercase; letter-spacing:.08em; }}
  .meta {{ color:#9598a1; font-size:.85rem; margin-bottom:8px; }}
  .verdict {{ font-size:1.05rem; margin:16px 0; padding:12px 16px; border-radius:6px;
             background:#1c2230; border-left:4px solid {'#00CC96' if m_norm < b_norm else '#EF553B'}; }}
  table {{ border-collapse:collapse; margin-top:8px; font-size:.9rem; }}
  th,td {{ padding:6px 14px; text-align:right; border-bottom:1px solid #2a2e39; }}
  th {{ color:#9598a1; font-weight:600; }}
  td.win {{ color:#00CC96; text-align:center; }}
  img {{ max-width:100%; border-radius:6px; margin-top:8px; }}
  .section {{ margin-bottom:40px; }}
</style>
</head>
<body>
<h1>{cfg.symbol} — Model 2 (LightGBM quantile) vs Black-Scholes analytic model</h1>
<div class="meta">
  Window: {cfg.window_mode} &nbsp;|&nbsp; train_window_days={cfg.train_window_days} &nbsp;|&nbsp;
  val_days={cfg.val_days} &nbsp;|&nbsp; test_days={cfg.test_days} &nbsp;|&nbsp;
  stride_days={cfg.stride_days}<br>
  Date range: {start} → {end} &nbsp;|&nbsp; {model_res['n_test_days']} test days &nbsp;|&nbsp;
  {len(model_res['predictions'])} snapshot rows &nbsp;|&nbsp; seed={cfg.seed}<br>
  Log: {os.path.relpath(log_path, BASE_DIR)}
</div>

<div class="verdict">
  {verdict}: mean pinball (normalized) <b>{m_norm:.5f}</b> vs <b>{b_norm:.5f}</b>
  &nbsp;({gain:+.1f}% vs Black-Scholes).<br>
  Mean pinball (log-return): model <b>{model_res['mean_pinball_lr']:.3e}</b> vs
  Black-Scholes <b>{bs_res['mean_pinball_lr']:.3e}</b>.
</div>

<div class="section">
<h2>Per-quantile comparison</h2>
<table>
  <tr><th>quantile</th><th>model pinball</th><th>BS pinball</th><th>model wins</th>
      <th>model coverage</th><th>BS coverage</th><th>model cov. error</th></tr>
  {rows}
</table>
<div class="meta">Lower pinball is better; coverage should approximate the quantile level
  (coverage_error = empirical − nominal, ~0 is well-calibrated).</div>
</div>

<div class="section">
<h2>Predicted close vs true close</h2>
<img src="data:image/png;base64,{img_b64}" alt="fan chart">
</div>
</body>
</html>"""

    os.makedirs(REPORT_DIR, exist_ok=True)
    out_path = os.path.join(REPORT_DIR, f"model2_vs_black_scholes_{cfg.symbol}_{start}_{end}.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return out_path


# ── logging ─────────────────────────────────────────────────────────────────────────────
def setup_logging(cfg: Config, start: str, end: str) -> tuple[logging.Logger, str]:
    """Console + file logger. File: logs/train/model2_{SYM}_{start}_{end}_{timestamp}.log."""
    os.makedirs(LOG_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(LOG_DIR, f"model2_{cfg.symbol}_{start}_{end}_{stamp}.log")

    log = logging.getLogger(f"model2.{stamp}")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    log.propagate = False
    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    for handler in (logging.StreamHandler(), logging.FileHandler(log_path, encoding="utf-8")):
        handler.setFormatter(fmt)
        log.addHandler(handler)
    return log, log_path


def log_summary(log: logging.Logger, title: str, summary: pd.DataFrame) -> None:
    log.info("%s", title)
    for _, r in summary.iterrows():
        log.info("    q=%.2f  pinball_norm=%.5f  pinball_lr=%.3e  coverage=%.3f  cov_err=%+.3f",
                 r["quantile"], r["pinball_norm"], r["pinball_lr"],
                 r["coverage"], r["coverage_error"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", type=str, default="SPY")
    parser.add_argument("--start-date", type=str, default="2026-03-31", help="ISO YYYY-MM-DD, inclusive")
    parser.add_argument("--end-date", type=str, default="2026-06-01", help="ISO YYYY-MM-DD, inclusive")
    parser.add_argument("--window-mode", choices=["rolling", "expanding"], default="rolling")
    parser.add_argument("--train-window-days", type=int, default=30)
    parser.add_argument("--val-days", type=int, default=1)
    args = parser.parse_args()

    cfg = Config(
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date,
        window_mode=args.window_mode,
        train_window_days=args.train_window_days,
        val_days=args.val_days,
        test_days=1,    # always 1
        stride_days=1,  # always 1
    )

    log, log_path = setup_logging(cfg, args.start_date, args.end_date)
    log.info("=" * 78)
    log.info("Model 2 — walk-forward LightGBM quantile regression")
    log.info("Arguments:")
    log.info("  symbol            = %s", cfg.symbol)
    log.info("  start_date        = %s", cfg.start_date)
    log.info("  end_date          = %s", cfg.end_date)
    log.info("  window_mode       = %s", cfg.window_mode)
    log.info("  train_window_days = %d", cfg.train_window_days)
    log.info("  val_days          = %d", cfg.val_days)
    log.info("  test_days         = %d (fixed)", cfg.test_days)
    log.info("  stride_days       = %d (fixed)", cfg.stride_days)
    log.info("  quantiles         = %s", list(cfg.quantiles))
    log.info("  seed              = %d", cfg.seed)
    log.info("  data_path         = %s", os.path.relpath(cfg.data_path, BASE_DIR))
    log.info("  log file          = %s", os.path.relpath(log_path, BASE_DIR))
    log.info("=" * 78)

    df = load_dataset(cfg)
    days = sorted(df[cfg.group_col].unique())
    log.info("Loaded %d rows across %d trading days (%s -> %s)",
             len(df), len(days), days[0], days[-1])

    model_res = run_walk_forward(cfg, df, log)
    log.info("-" * 78)
    log.info("Model 2 aggregate over %d pooled test days (%d snapshot rows):",
             model_res["n_test_days"], len(model_res["predictions"]))
    log.info("  mean pinball  norm=%.5f  log-return=%.6e",
             model_res["mean_pinball_norm"], model_res["mean_pinball_lr"])
    log_summary(log, "Model 2 per-quantile:", model_res["summary"])

    bs_res = run_bs_model(cfg, df, model_res["test_days"])
    log.info("-" * 78)
    log.info("Black-Scholes analytic model on the SAME %d test days:", model_res["n_test_days"])
    log.info("  mean pinball  norm=%.5f  log-return=%.6e",
             bs_res["mean_pinball_norm"], bs_res["mean_pinball_lr"])
    log_summary(log, "Black-Scholes per-quantile:", bs_res["summary"])

    gain = (1 - model_res["mean_pinball_norm"] / bs_res["mean_pinball_norm"]) * 100
    log.info("-" * 78)
    log.info("Model 2 vs Black-Scholes (normalized pinball): %.5f vs %.5f  (model %+.1f%%)",
             model_res["mean_pinball_norm"], bs_res["mean_pinball_norm"], gain)

    # save model 2 predictions for the test days
    os.makedirs(PRED_DIR, exist_ok=True)
    pred_path = os.path.join(PRED_DIR, f"{cfg.symbol}_model2.parquet")
    model_res["predictions"].to_parquet(pred_path, index=False)
    log.info("Saved Model 2 test predictions to %s", os.path.relpath(pred_path, BASE_DIR))

    report_path = generate_report(cfg, model_res, bs_res, args.start_date, args.end_date, log_path)
    log.info("Saved comparison report to %s", os.path.relpath(report_path, BASE_DIR))
    log.info("Done.")


if __name__ == "__main__":
    main()
