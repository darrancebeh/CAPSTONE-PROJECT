"""Interactive dashboard over a completed backtest.

Run with::

    streamlit run dashboard/app.py

The application reads the cached forecast panel rather than re-estimating
anything, so every table below is recomputed live against whatever date range
and regime the user selects, while the expensive walk-forward stays offline.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from volforecast.backtest.cache import ForecastStore  # noqa: E402
from volforecast.config import Config  # noqa: E402
from volforecast.evaluation.diebold_mariano import (  # noqa: E402
    pairwise_tests,
    results_to_frame,
    statistic_matrix,
)
from volforecast.evaluation.metrics import loss_series, summarise_forecast  # noqa: E402
from volforecast.evaluation.var_backtest import (  # noqa: E402
    exception_indicators,
    run_var_backtests,
    value_at_risk,
    var_results_to_frame,
)

TRADING_DAYS = 252
META_COLUMNS = ["proxy", "return", "regime"]

st.set_page_config(
    page_title="Volatility Forecast Backtest",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ----------------------------------------------------------------------
# Data access
# ----------------------------------------------------------------------
@st.cache_resource
def load_config() -> Config:
    return Config.load()


@st.cache_data(show_spinner=False)
def load_panel(path: str) -> pd.DataFrame:
    panel = pd.read_parquet(path)
    panel.index = pd.DatetimeIndex(panel.index)
    panel.index.name = "date"
    return panel


@st.cache_data(show_spinner=False)
def load_metadata(directory: str) -> Dict:
    return ForecastStore(Path(directory)).load_metadata()


@st.cache_data(show_spinner=False)
def load_fit_log(directory: str) -> pd.DataFrame:
    return ForecastStore(Path(directory)).load_fit_log()


def annualised_vol(variance_pct_squared) -> np.ndarray:
    """Convert a daily variance in %^2 into an annualised volatility in %."""
    return np.sqrt(np.asarray(variance_pct_squared, dtype=float) * TRADING_DAYS)


# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
def sidebar_controls(panel: pd.DataFrame, labels: Dict[str, str]):
    st.sidebar.title("Backtest controls")

    keys = [k for k in panel.columns if k not in META_COLUMNS]
    default = [k for k in keys if k in {"garch", "gjr_garch", "lstm_qlike", "lstm_mse"}] or keys

    selected = st.sidebar.multiselect(
        "Models",
        options=keys,
        default=default,
        format_func=lambda k: labels.get(k, k),
    )

    regimes = ["All regimes"] + [
        r for r in panel["regime"].dropna().unique().tolist() if r != "Unclassified"
    ]
    regime = st.sidebar.selectbox("Market regime", regimes)

    if regime != "All regimes":
        scoped = panel.loc[panel["regime"] == regime]
    else:
        scoped = panel

    lo, hi = scoped.index.min().date(), scoped.index.max().date()
    start, end = st.sidebar.slider(
        "Date range",
        min_value=lo,
        max_value=hi,
        value=(lo, hi),
        format="YYYY-MM-DD",
    )

    st.sidebar.divider()
    loss_name = st.sidebar.radio(
        "Loss function",
        options=["qlike", "mse", "mae"],
        format_func=str.upper,
        help="QLIKE is the primary criterion; it is the only one of the three that "
        "ranks forecasts consistently when the target is a noisy proxy.",
    )
    confidence = st.sidebar.select_slider(
        "VaR confidence", options=[0.95, 0.99], value=0.95, format_func=lambda c: f"{c:.0%}"
    )

    window = scoped.loc[str(start) : str(end)]
    return selected, regime, window, loss_name, confidence


# ----------------------------------------------------------------------
# Panels
# ----------------------------------------------------------------------
def render_overview(window: pd.DataFrame, keys: List[str], labels: Dict, loss_name: str, meta: Dict):
    proxy = window["proxy"]

    scores = {
        key: float(loss_series(proxy, window[key], loss_name).mean())
        for key in keys
        if window[key].notna().any()
    }
    if not scores:
        st.warning("No forecasts available for the current selection.")
        return

    ranked = sorted(scores.items(), key=lambda item: item[1])
    best_key, best_score = ranked[0]

    columns = st.columns(4)
    columns[0].metric("Sessions evaluated", f"{len(window):,}")
    columns[1].metric(
        "Realised volatility",
        f"{np.sqrt(proxy.mean() * TRADING_DAYS):.1f}%",
        help="Annualised, from the squared-return proxy over the selected window",
    )
    columns[2].metric(f"Best {loss_name.upper()}", f"{best_score:.4f}", labels.get(best_key, best_key))

    if len(ranked) > 1:
        runner_up_key, runner_up = ranked[1]
        gap = (runner_up - best_score) / runner_up * 100 if runner_up else 0.0
        columns[3].metric(
            "Margin over next best",
            f"{gap:.2f}%",
            labels.get(runner_up_key, runner_up_key),
            delta_color="off",
        )

    rows = []
    for key in keys:
        summary = summarise_forecast(proxy, window[key])
        if not summary:
            continue
        summary["Model"] = labels.get(key, key)
        rows.append(summary)

    table = pd.DataFrame(rows).set_index("Model")
    display = table[["qlike", "mse", "rmse", "mae", "mz_slope", "mz_r2", "bias_volatility"]]
    display.columns = ["QLIKE", "MSE", "RMSE", "MAE", "MZ slope", "MZ R2", "Bias (vol pts)"]

    st.subheader("Out-of-sample accuracy")
    st.dataframe(
        display.style.format("{:.4f}")
        .highlight_min(subset=["QLIKE", "MSE", "RMSE", "MAE"], color="#1b5e20")
        .highlight_max(subset=["MZ R2"], color="#1b5e20"),
        use_container_width=True,
    )
    st.caption(
        "The Mincer-Zarnowitz slope regresses the realised proxy on the forecast. "
        "A slope near one indicates a correctly scaled forecast; a slope above one "
        "indicates forecasts that are too compressed."
    )

    with st.expander("Run configuration"):
        left, right = st.columns(2)
        left.write(
            {
                "Symbol": meta.get("symbol"),
                "Evaluation start": meta.get("evaluation_start"),
                "Evaluation end": meta.get("evaluation_end"),
                "Sessions": meta.get("n_evaluation_days"),
            }
        )
        right.write(
            {
                "Sequence length": meta.get("sequence_length"),
                "Predictors": meta.get("n_features"),
                "Random seed": meta.get("random_seed"),
                "Refit cadence": meta.get("refit_cadence"),
            }
        )
        audit = meta.get("lookahead_audit_max_difference")
        if audit is not None:
            st.success(
                f"Look-ahead audit passed: forecasts were unchanged to within {audit:.2e} "
                "when all observations after the information boundary were removed."
            )


def render_forecasts(window: pd.DataFrame, keys: List[str], labels: Dict):
    st.subheader("Forecast overlay")
    log_scale = st.checkbox("Logarithmic axis", value=True)

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=window.index,
            y=annualised_vol(window["proxy"]),
            name="Realised proxy",
            mode="markers",
            marker=dict(size=3, color="rgba(140,140,140,0.45)"),
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.1f}%<extra>Realised</extra>",
        )
    )
    for key in keys:
        figure.add_trace(
            go.Scatter(
                x=window.index,
                y=annualised_vol(window[key]),
                name=labels.get(key, key),
                mode="lines",
                line=dict(width=1.8),
                hovertemplate="%{x|%Y-%m-%d}<br>%{y:.1f}%<extra>"
                + labels.get(key, key)
                + "</extra>",
            )
        )

    figure.update_layout(
        height=520,
        hovermode="x unified",
        yaxis_title="Annualised volatility (%)",
        xaxis_title=None,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    if log_scale:
        figure.update_yaxes(type="log")
    st.plotly_chart(figure, use_container_width=True)
    st.caption(
        "Each grey point is one squared return, an unbiased but very noisy reading "
        "of that day's latent variance. Model lines are not expected to track "
        "individual points."
    )


def render_accuracy(window: pd.DataFrame, keys: List[str], labels: Dict, loss_name: str, panel):
    st.subheader("Loss accumulation")
    benchmark = st.selectbox(
        "Benchmark", options=keys, format_func=lambda k: labels.get(k, k), key="loss_benchmark"
    )

    proxy = window["proxy"]
    base = loss_series(proxy, window[benchmark], loss_name)

    figure = go.Figure()
    for key in keys:
        if key == benchmark:
            continue
        differential = (loss_series(proxy, window[key], loss_name) - base).cumsum()
        figure.add_trace(
            go.Scatter(
                x=differential.index,
                y=differential.to_numpy(),
                name=labels.get(key, key),
                mode="lines",
            )
        )
    figure.add_hline(y=0, line_dash="dash", line_color="grey")
    figure.update_layout(
        height=420,
        hovermode="x unified",
        yaxis_title=f"Cumulative {loss_name.upper()} minus {labels.get(benchmark, benchmark)}",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(figure, use_container_width=True)
    st.caption(
        "A rising line means the model is losing ground to the benchmark. The slope "
        "shows when the difference was accumulated."
    )

    st.subheader("Average loss by regime")
    regimes = [r for r in panel["regime"].unique() if r != "Unclassified"]
    matrix = pd.DataFrame(index=[labels.get(k, k) for k in keys], columns=regimes, dtype=float)
    for regime in regimes:
        subset = panel.loc[panel["regime"] == regime]
        for key in keys:
            matrix.loc[labels.get(key, key), regime] = float(
                loss_series(subset["proxy"], subset[key], loss_name).mean()
            )

    heatmap = px.imshow(
        matrix.astype(float),
        text_auto=".3f",
        color_continuous_scale="RdYlGn_r",
        aspect="auto",
        labels=dict(color=loss_name.upper()),
    )
    heatmap.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(heatmap, use_container_width=True)
    st.caption(
        "Computed over the full regime window regardless of the date slider, so the "
        "columns stay comparable. Green is better."
    )


def render_significance(window: pd.DataFrame, keys: List[str], labels: Dict, loss_name: str, config):
    st.subheader("Diebold-Mariano tests")
    st.write(
        "Tests the null that two forecasts have equal expected loss. The statistic is "
        "scaled by a Newey-West long-run variance and carries the "
        "Harvey-Leybourne-Newbold small-sample correction."
    )

    if len(keys) < 2:
        st.info("Select at least two models to compare.")
        return

    results = pairwise_tests(
        proxy=window["proxy"],
        forecasts=window[keys],
        loss=loss_name,
        horizon=1,
        lags=config.evaluation.hac_lags,
        harvey_correction=config.evaluation.harvey_correction,
        labels=labels,
    )
    if not results:
        st.warning("Not enough overlapping observations for a test.")
        return

    order = [labels.get(k, k) for k in keys]
    statistics = statistic_matrix(results, order=order, value="statistic")
    p_values = statistic_matrix(results, order=order, value="p_value")

    left, right = st.columns(2)
    with left:
        figure = px.imshow(
            statistics.astype(float),
            text_auto=".2f",
            color_continuous_scale="RdBu",
            color_continuous_midpoint=0.0,
            aspect="auto",
            labels=dict(color="DM stat"),
        )
        figure.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(figure, use_container_width=True)
        st.caption("Negative (blue) means the row model has the lower average loss.")

    with right:
        figure = px.imshow(
            p_values.astype(float),
            text_auto=".3f",
            color_continuous_scale="Greens_r",
            range_color=[0.0, 0.2],
            aspect="auto",
            labels=dict(color="p-value"),
        )
        figure.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(figure, use_container_width=True)
        st.caption("Darker means a more significant difference. Values above 0.2 are clipped.")

    frame = results_to_frame(results)
    frame["verdict"] = [r.verdict(config.evaluation.var_test_significance) for r in results]
    display = frame[
        ["model_a", "model_b", "mean_differential", "dm_statistic", "p_value", "verdict"]
    ]
    display.columns = ["Model A", "Model B", "Mean difference", "DM statistic", "p-value", "Verdict"]
    st.dataframe(
        display.style.format(
            {"Mean difference": "{:.5f}", "DM statistic": "{:.3f}", "p-value": "{:.4f}"}
        ),
        use_container_width=True,
        hide_index=True,
    )


def render_risk(window: pd.DataFrame, keys: List[str], labels: Dict, confidence: float, config):
    st.subheader(f"Value-at-Risk coverage at {confidence:.0%}")

    results = run_var_backtests(
        returns=window["return"],
        forecasts=window[keys],
        confidence_levels=[confidence],
        labels=labels,
    )
    frame = var_results_to_frame(results, alpha=config.evaluation.var_test_significance)
    display = frame[
        [
            "model",
            "n_exceptions",
            "expected_exceptions",
            "exception_rate",
            "kupiec_p",
            "christoffersen_p",
            "cc_p",
            "pass_conditional",
        ]
    ]
    display.columns = [
        "Model",
        "Exceptions",
        "Expected",
        "Rate",
        "Kupiec p",
        "Christoffersen p",
        "Joint p",
        "Passes",
    ]
    st.dataframe(
        display.style.format(
            {
                "Expected": "{:.1f}",
                "Rate": "{:.3%}",
                "Kupiec p": "{:.4f}",
                "Christoffersen p": "{:.4f}",
                "Joint p": "{:.4f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "Kupiec tests whether the number of breaches matches the nominal rate. "
        "Christoffersen tests whether breaches arrive independently rather than "
        "clustering during a single episode."
    )

    st.subheader("Breach timeline")
    chosen = st.selectbox(
        "Model", options=keys, format_func=lambda k: labels.get(k, k), key="var_model"
    )
    threshold = -value_at_risk(window[chosen], confidence)
    breaches = exception_indicators(window["return"], window[chosen], confidence)
    breach_dates = breaches.index[breaches == 1]

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=window.index,
            y=window["return"],
            name="Daily return",
            mode="lines",
            line=dict(width=1, color="rgba(120,120,120,0.6)"),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=window.index,
            y=threshold,
            name=f"{confidence:.0%} VaR",
            mode="lines",
            line=dict(width=1.6, color="#d62728"),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=breach_dates,
            y=window.loc[breach_dates, "return"],
            name="Exception",
            mode="markers",
            marker=dict(size=7, color="#d62728", symbol="x"),
        )
    )
    figure.update_layout(
        height=420,
        hovermode="x unified",
        yaxis_title="Return (%)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(figure, use_container_width=True)


def render_diagnostics(meta: Dict, fit_log: pd.DataFrame, labels: Dict):
    st.subheader("Computational cost")
    runtime = pd.DataFrame(meta.get("models", []))
    if not runtime.empty:
        display = runtime[
            ["label", "family", "n_forecasts", "n_refits", "fit_seconds", "total_seconds"]
        ].copy()
        display.columns = ["Model", "Family", "Forecasts", "Refits", "Fit (s)", "Total (s)"]
        st.dataframe(display, use_container_width=True, hide_index=True)

        figure = px.bar(
            display, x="Model", y="Total (s)", color="Family", text_auto=".1f"
        )
        figure.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(figure, use_container_width=True)
        st.caption(
            "Wall-clock cost of the full walk-forward, including every refit."
        )

    if fit_log.empty:
        return

    st.subheader("Parameter stability")
    models = sorted(fit_log["model"].unique())
    chosen = st.selectbox(
        "Model", options=models, format_func=lambda k: labels.get(k, k), key="fit_log_model"
    )
    subset = fit_log.loc[fit_log["model"] == chosen].set_index("refit_date")
    numeric = subset.select_dtypes("number").drop(columns=["n_train_observations"], errors="ignore")
    numeric = numeric.dropna(axis=1, how="all")

    if numeric.empty:
        st.info("This model has no estimated parameters to display.")
        return

    figure = go.Figure()
    for column in numeric.columns:
        figure.add_trace(
            go.Scatter(x=numeric.index, y=numeric[column], name=column, mode="lines")
        )
    figure.update_layout(
        height=400,
        hovermode="x unified",
        yaxis_title="Estimate",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(figure, use_container_width=True)
    st.caption(
        "Parameters re-estimated on each expanding window. Drift indicates the "
        "specification is absorbing a structural change."
    )


# ----------------------------------------------------------------------
def main() -> None:
    config = load_config()
    store = ForecastStore(config.backtest.forecast_dir)
    panel_path = config.backtest.forecast_dir / "evaluation_panel.parquet"

    st.title("GARCH versus LSTM volatility forecasting")
    st.caption(
        f"One-day-ahead conditional variance for {config.data.symbol}, evaluated out of "
        "sample under proxy-robust loss with formal significance testing."
    )

    if not panel_path.exists() or not store.exists():
        st.error("No cached backtest found.")
        st.code("python scripts/run_pipeline.py", language="bash")
        st.stop()

    panel = load_panel(str(panel_path))
    meta = load_metadata(str(config.backtest.forecast_dir))
    fit_log = load_fit_log(str(config.backtest.forecast_dir))
    labels = {entry["key"]: entry["label"] for entry in meta.get("models", [])}

    keys, regime, window, loss_name, confidence = sidebar_controls(panel, labels)
    if not keys:
        st.info("Select at least one model in the sidebar.")
        st.stop()
    if window.empty:
        st.warning("The selected date range contains no sessions.")
        st.stop()

    if regime != "All regimes":
        st.info(f"Scoped to **{regime}** ({len(window)} sessions).")

    tabs = st.tabs(
        ["Overview", "Forecasts", "Accuracy", "Significance", "Risk", "Diagnostics"]
    )
    with tabs[0]:
        render_overview(window, keys, labels, loss_name, meta)
    with tabs[1]:
        render_forecasts(window, keys, labels)
    with tabs[2]:
        render_accuracy(window, keys, labels, loss_name, panel)
    with tabs[3]:
        render_significance(window, keys, labels, loss_name, config)
    with tabs[4]:
        render_risk(window, keys, labels, confidence, config)
    with tabs[5]:
        render_diagnostics(meta, fit_log, labels)


if __name__ == "__main__":
    main()
