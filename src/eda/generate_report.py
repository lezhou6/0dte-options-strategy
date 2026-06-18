"""Generate SPY 0DTE HTML analysis report with interactive Plotly charts."""

import glob
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

GREEKS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "greeks")
VIZ_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "visualization")

MAX_DAYS = 10

COLORS = [
    "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
    "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52",
]


def load_data(raw_dir: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(raw_dir, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet files in {raw_dir}")
    raw = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = raw[
        (raw["implied_vol"] > 0)
        & (raw["iv_error"] < 1.0)
        & (raw["delta"].abs().between(0.01, 0.99))
        & (raw["bid"] > 0)
    ].copy()
    df["mid"] = (df["bid"] + df["ask"]) / 2
    return df


def build_atm_fig(df: pd.DataFrame) -> go.Figure:
    """ATM mid price vs minutes to expiration."""
    df = df.copy()
    df["strike_dist"] = (df["strike"] - df["underlying_price"]).abs()
    atm = df.loc[df.groupby(["expiration", "timestamp", "right"])["strike_dist"].idxmin()].copy()
    close_dt = atm["timestamp"].apply(
        lambda t: t.replace(hour=16, minute=0, second=0, microsecond=0)
    )
    atm["ttm_min"] = (close_dt - atm["timestamp"]).dt.total_seconds() / 60

    exps = sorted(atm["expiration"].unique())

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["ATM Call (mid)", "ATM Put (mid)"],
        horizontal_spacing=0.12,
    )

    trace_meta = []
    for i, exp in enumerate(exps):
        color = COLORS[i % len(COLORS)]
        for right in ["CALL", "PUT"]:
            sub = (
                atm[(atm["expiration"] == exp) & (atm["right"] == right)]
                .sort_values("ttm_min", ascending=False)
            )
            col = 1 if right == "CALL" else 2
            fig.add_trace(
                go.Scatter(
                    x=sub["ttm_min"].tolist(),
                    y=sub["mid"].round(2).tolist(),
                    name=str(exp),
                    mode="lines+markers",
                    marker=dict(size=5),
                    line=dict(color=color),
                    legendgroup=str(exp),
                    showlegend=(right == "CALL"),
                    visible=True,
                ),
                row=1, col=col,
            )
            trace_meta.append((exp, right))

    fig.update_xaxes(autorange="reversed", title_text="Minutes to expiration")
    fig.update_yaxes(title_text="Mid price ($)")
    fig.update_layout(
        template="plotly_dark",
        title="ATM Option Mid Price vs Time to Expiration",
        legend=dict(title="Expiration"),
        height=520,
    )
    return fig


def build_price_fig(df: pd.DataFrame) -> go.Figure:
    """Intraday underlying price, one trace per day so days are not connected."""
    price_df = (
        df[["expiration", "timestamp", "underlying_price"]]
        .drop_duplicates(subset=["expiration", "timestamp"])
        .sort_values(["expiration", "timestamp"])
    )

    exps = sorted(price_df["expiration"].unique())
    fig = go.Figure()
    for i, exp in enumerate(exps):
        day = price_df[price_df["expiration"] == exp]
        fig.add_trace(go.Scatter(
            x=day["timestamp"].tolist(),
            y=day["underlying_price"].tolist(),
            mode="lines",
            line=dict(color=COLORS[i % len(COLORS)], width=1.5),
            name=str(exp),
        ))

    fig.update_layout(
        template="plotly_dark",
        title="Underlying Price (15m intervals)",
        xaxis=dict(
            title="Date",
            rangebreaks=[
                dict(bounds=["sat", "mon"]),
                dict(bounds=[16, 9.5], pattern="hour"),
            ],
        ),
        yaxis=dict(title="Price ($)"),
        legend=dict(title="Expiration"),
        height=420,
    )
    return fig


def _smile_xy(
    df: pd.DataFrame, time_str: str, exp, right: str, window_pct: float = 0.08
) -> tuple[list, list]:
    """Return (moneyness%, implied_vol) for one (time, expiration, right) slice."""
    h, m = int(time_str[:2]), int(time_str[3:])
    sub = df[
        (df["timestamp"].dt.hour == h)
        & (df["timestamp"].dt.minute == m)
        & (df["expiration"] == exp)
        & (df["right"] == right)
    ]
    if sub.empty:
        return [], []
    underlying = sub["underlying_price"].iloc[0]
    sub = sub[
        sub["strike"].between(underlying * (1 - window_pct), underlying * (1 + window_pct))
    ].sort_values("strike")
    moneyness = ((sub["strike"] - underlying) / underlying * 100).round(2).tolist()
    iv = sub["implied_vol"].round(4).tolist()
    return moneyness, iv


def build_smile_fig(df: pd.DataFrame, default_time: str = "12:00") -> go.Figure:
    """IV smile (IV vs moneyness) with time dropdown."""
    exps = sorted(df["expiration"].unique())
    times = sorted(df["timestamp"].dt.strftime("%H:%M").unique())
    if default_time not in times:
        default_time = times[len(times) // 2]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["IV Smile — Calls", "IV Smile — Puts"],
        horizontal_spacing=0.12,
    )

    trace_meta = []
    for i, exp in enumerate(exps):
        color = COLORS[i % len(COLORS)]
        for right in ["CALL", "PUT"]:
            x, y = _smile_xy(df, default_time, exp, right)
            col = 1 if right == "CALL" else 2
            fig.add_trace(
                go.Scatter(
                    x=x, y=y,
                    name=str(exp),
                    mode="lines+markers",
                    marker=dict(size=5),
                    line=dict(color=color),
                    legendgroup=str(exp),
                    showlegend=(right == "CALL"),
                    visible=True,
                ),
                row=1, col=col,
            )
            trace_meta.append((exp, right))

    # Time dropdown — updates x/y data for all traces, does not touch visibility
    time_buttons = []
    for t in times:
        new_x, new_y = [], []
        for exp, right in trace_meta:
            x, y = _smile_xy(df, t, exp, right)
            new_x.append(x)
            new_y.append(y)
        time_buttons.append(dict(label=t, method="restyle", args=[{"x": new_x, "y": new_y}]))

    fig.update_xaxes(
        title_text="Moneyness (%)",
        zeroline=True,
        zerolinecolor="#555",
        zerolinewidth=1,
    )
    fig.update_yaxes(title_text="Implied Volatility")
    fig.update_layout(
        template="plotly_dark",
        title="IV Smile",
        legend=dict(title="Expiration"),
        updatemenus=[dict(
            buttons=time_buttons,
            direction="down",
            showactive=True,
            x=0.5,
            xanchor="center",
            y=1.2,
            yanchor="top",
            active=times.index(default_time),
            bgcolor="#2a2e39",
            bordercolor="#555",
            font=dict(color="#d1d4dc"),
        )],
        height=560,
        margin=dict(t=120),
        annotations=[dict(
            text="Time:",
            x=0.38, xanchor="right",
            y=1.17, yanchor="top",
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(color="#9598a1"),
        )],
    )
    return fig


def generate_symbol_report(symbol: str) -> None:
    raw_dir = os.path.join(GREEKS_DIR, symbol)
    print(f"Loading data for {symbol}...")
    df = load_data(raw_dir)
    exps = sorted(df["expiration"].unique())
    print(f"  {len(exps)} expirations: {exps[0]} → {exps[-1]}")

    df_full = df
    if len(exps) > MAX_DAYS:
        indices = np.round(np.linspace(0, len(exps) - 1, MAX_DAYS)).astype(int)
        selected = [exps[i] for i in indices]
        print(f"  Sampling {MAX_DAYS} days across full range: {selected[0]} → {selected[-1]}")
        df = df[df["expiration"].isin(selected)]

    print(f"  Building figures...")
    fig_atm = build_atm_fig(df)
    fig_price = build_price_fig(df_full)
    fig_smile = build_smile_fig(df)

    os.makedirs(VIZ_DIR, exist_ok=True)
    out_path = os.path.join(VIZ_DIR, f"{symbol}_report.html")
    html_atm = fig_atm.to_html(full_html=False, include_plotlyjs=False)
    html_price = fig_price.to_html(full_html=False, include_plotlyjs=False)
    html_smile = fig_smile.to_html(full_html=False, include_plotlyjs=False)

    report = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{symbol} 0DTE Analysis</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
  body {{
    background: #131722;
    color: #d1d4dc;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    margin: 0;
    padding: 24px 32px;
  }}
  h1 {{
    color: #d1d4dc;
    border-bottom: 1px solid #2a2e39;
    padding-bottom: 12px;
    margin-bottom: 36px;
    font-size: 1.4rem;
    letter-spacing: 0.02em;
  }}
  h2 {{
    color: #9598a1;
    margin-top: 52px;
    margin-bottom: 4px;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }}
  .section {{ margin-bottom: 52px; }}
</style>
</head>
<body>
<h1>{symbol} 0DTE Analysis</h1>
<div class="section"><h2>ATM Option Price Decay</h2>{html_atm}</div>
<div class="section"><h2>Underlying Price</h2>{html_price}</div>
<div class="section"><h2>IV Smile</h2>{html_smile}</div>
</body>
</html>"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  Report saved to {out_path}")


def main() -> None:
    symbols = [
        d for d in os.listdir(GREEKS_DIR)
        if os.path.isdir(os.path.join(GREEKS_DIR, d))
    ]
    if not symbols:
        raise FileNotFoundError(f"No symbol directories found in {GREEKS_DIR}")
    for symbol in sorted(symbols):
        generate_symbol_report(symbol)


if __name__ == "__main__":
    main()
