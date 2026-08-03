"""Render the static figures used in the report from a cached backtest.

    python scripts/make_figures.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import _bootstrap  # noqa: F401

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from volforecast.backtest.cache import ForecastStore
from volforecast.config import Config
from volforecast.evaluation.metrics import loss_series
from volforecast.evaluation.var_backtest import exception_indicators, value_at_risk
from volforecast.utils import get_logger

logger = get_logger("make_figures")

TRADING_DAYS = 252
META_COLUMNS = ["proxy", "return", "regime"]

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    }
)

PALETTE = {
    "garch": "#1f77b4",
    "egarch": "#2ca02c",
    "gjr_garch": "#17becf",
    "lstm_qlike": "#d62728",
    "lstm_mse": "#ff7f0e",
}


def annualised(variance) -> np.ndarray:
    return np.sqrt(np.asarray(variance, dtype=float) * TRADING_DAYS)


def figure_regime_context(panel: pd.DataFrame, config: Config, path: Path) -> None:
    """Realised volatility over the evaluation window with the regimes shaded."""
    figure, axis = plt.subplots(figsize=(9.5, 3.4))

    realised = pd.Series(annualised(panel["proxy"]), index=panel.index)
    axis.fill_between(realised.index, 0, realised, color="#cccccc", alpha=0.6, linewidth=0)
    axis.plot(realised.index, realised.rolling(21).mean(), color="#222222", linewidth=1.3)

    ceiling = float(realised.max()) * 1.12
    axis.set_ylim(0, ceiling)

    total_span = (panel.index[-1] - panel.index[0]).days
    colours = plt.cm.Set3(np.linspace(0, 1, len(config.regimes)))
    for regime, colour in zip(config.regimes, colours):
        start = max(pd.Timestamp(regime.start), panel.index[0])
        end = min(pd.Timestamp(regime.end), panel.index[-1])
        if end < panel.index[0] or start > panel.index[-1]:
            continue
        axis.axvspan(start, end, color=colour, alpha=0.35, linewidth=0)

        # A narrow regime cannot fit a horizontal label, so it is set upright.
        share = (end - start).days / total_span
        axis.text(
            start + (end - start) / 2,
            ceiling * 0.97,
            regime.name,
            ha="center",
            va="top",
            fontsize=7.5,
            style="italic",
            rotation=0 if share > 0.10 else 90,
        )

    axis.set_ylabel("Annualised volatility (%)")
    axis.set_title(
        "Realised volatility over the evaluation period, 21-day moving average",
        fontsize=9.5,
        loc="left",
    )
    axis.xaxis.set_major_locator(mdates.YearLocator())
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    figure.savefig(path)
    plt.close(figure)


def figure_regime_heatmap(
    panel: pd.DataFrame, keys: List[str], labels: Dict, config: Config, path: Path
) -> None:
    """Average QLIKE by model and regime, shaded within each column."""
    regimes = [r.name for r in config.regimes if (panel["regime"] == r.name).any()]
    matrix = np.full((len(keys), len(regimes) + 1), np.nan)

    for row, key in enumerate(keys):
        for column, regime in enumerate(regimes):
            subset = panel.loc[panel["regime"] == regime]
            matrix[row, column] = loss_series(subset["proxy"], subset[key], "qlike").mean()
        matrix[row, -1] = loss_series(panel["proxy"], panel[key], "qlike").mean()

    figure, axis = plt.subplots(figsize=(8.2, 3.0))
    # Each column is shaded against its own range: absolute QLIKE varies far
    # more across regimes than across models, which would otherwise wash out
    # the within-regime comparison the figure exists to show.
    spread = np.ptp(matrix, axis=0)
    normalised = (matrix - matrix.min(axis=0)) / np.where(spread > 0, spread, 1.0)
    axis.imshow(normalised, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=1)

    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column, row, f"{matrix[row, column]:.3f}",
                ha="center", va="center", fontsize=7.5, color="black",
            )

    axis.set_xticks(range(len(regimes) + 1))
    axis.set_xticklabels(regimes + ["Full sample"], rotation=20, ha="right", fontsize=8)
    axis.set_yticks(range(len(keys)))
    axis.set_yticklabels([labels.get(k, k) for k in keys], fontsize=8)
    axis.set_title("Average QLIKE by regime (shaded within column)", fontsize=9.5, loc="left")
    axis.grid(False)
    figure.savefig(path)
    plt.close(figure)


def figure_price_and_volatility(prices: pd.Series, scale: float, path: Path) -> None:
    """Closing price above, realised volatility below, over the full sample."""
    figure, axes = plt.subplots(2, 1, figsize=(9.0, 5.0), sharex=True)

    axes[0].plot(prices.index, prices.to_numpy(), color="#1f77b4", linewidth=1.0)
    axes[0].set_ylabel("Closing price (USD)")
    axes[0].set_title("SPY daily closing prices", fontsize=9.5, loc="left")

    # Derived from the full price history so that both panels span the same
    # dates; the evaluation panel begins later than the sample does.
    returns = np.log(prices).diff().dropna() * scale
    realised = pd.Series(annualised(returns.pow(2)), index=returns.index)
    axes[1].fill_between(realised.index, 0, realised, color="#c6c6c6", linewidth=0)
    axes[1].plot(realised.index, realised.rolling(21).mean(), color="#222222", linewidth=1.2)
    axes[1].set_ylabel("Annualised volatility (%)")
    axes[1].set_ylim(bottom=0)
    axes[1].set_title(
        "Realised volatility from squared returns, with 21-day moving average",
        fontsize=9.5, loc="left",
    )

    axes[1].xaxis.set_major_locator(mdates.YearLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def figure_forecast_overlay(
    panel: pd.DataFrame, keys: List[str], labels: Dict, path: Path
) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(9.5, 6.4), sharey=False)

    windows = [
        (panel.index[0], panel.index[-1], "Full out-of-sample period"),
        (pd.Timestamp("2020-01-02"), pd.Timestamp("2020-06-30"), "COVID-19 crash and recovery"),
    ]

    for axis, (start, end, title) in zip(axes, windows):
        window = panel.loc[start:end]
        axis.plot(
            window.index,
            annualised(window["proxy"]),
            color="#bbbbbb",
            linewidth=0.6,
            label="Realised proxy",
            zorder=1,
        )
        for key in keys:
            axis.plot(
                window.index,
                annualised(window[key]),
                color=PALETTE.get(key),
                linewidth=1.2,
                label=labels.get(key, key),
                zorder=2,
            )
        axis.set_yscale("log")
        axis.set_ylabel("Annualised volatility (%)")
        axis.set_title(title, fontsize=9.5, loc="left")

    axes[0].legend(ncol=4, loc="upper left", fontsize=7.5)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def figure_cumulative_loss(
    panel: pd.DataFrame, keys: List[str], labels: Dict, benchmark: str, path: Path
) -> None:
    figure, axis = plt.subplots(figsize=(9.5, 3.6))
    base = loss_series(panel["proxy"], panel[benchmark], "qlike")

    for key in keys:
        if key == benchmark:
            continue
        differential = (loss_series(panel["proxy"], panel[key], "qlike") - base).cumsum()
        axis.plot(
            differential.index,
            differential.to_numpy(),
            color=PALETTE.get(key),
            linewidth=1.3,
            label=labels.get(key, key),
        )

    axis.axhline(0, color="black", linewidth=0.8, linestyle="--")
    axis.set_ylabel(f"Cumulative QLIKE less {labels.get(benchmark, benchmark)}")
    axis.set_title(
        "Loss accumulated relative to the benchmark; rising means losing ground",
        fontsize=9.5,
        loc="left",
    )
    axis.legend(ncol=3, fontsize=8)
    axis.xaxis.set_major_locator(mdates.YearLocator())
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    figure.savefig(path)
    plt.close(figure)


def figure_var_breaches(
    panel: pd.DataFrame, keys: List[str], labels: Dict, confidence: float, path: Path
) -> None:
    chosen = [k for k in ("garch", "lstm_qlike") if k in keys] or keys[:2]
    figure, axes = plt.subplots(len(chosen), 1, figsize=(9.5, 2.4 * len(chosen)), sharex=True)
    axes = np.atleast_1d(axes)

    for axis, key in zip(axes, chosen):
        threshold = -value_at_risk(panel[key], confidence)
        breaches = exception_indicators(panel["return"], panel[key], confidence)
        breach_dates = breaches.index[breaches == 1]

        axis.plot(panel.index, panel["return"], color="#999999", linewidth=0.5, label="Return")
        axis.plot(
            panel.index,
            threshold,
            color=PALETTE.get(key, "#d62728"),
            linewidth=1.0,
            label=f"{confidence:.0%} VaR",
        )
        axis.scatter(
            breach_dates,
            panel.loc[breach_dates, "return"],
            s=9,
            color="#d62728",
            marker="x",
            zorder=3,
            label=f"Exceptions ({len(breach_dates)})",
        )
        axis.set_ylabel("Return (%)")
        axis.set_title(labels.get(key, key), fontsize=9, loc="left")
        axis.legend(ncol=3, fontsize=7.5, loc="lower left")

    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()

    config = Config.load(args.config)
    panel_path = config.backtest.forecast_dir / "evaluation_panel.parquet"
    if not panel_path.exists():
        logger.error("No evaluation panel at %s. Run scripts/run_pipeline.py first.", panel_path)
        return 1

    panel = pd.read_parquet(panel_path)
    panel.index = pd.DatetimeIndex(panel.index)
    metadata = ForecastStore(config.backtest.forecast_dir).load_metadata()
    labels = {entry["key"]: entry["label"] for entry in metadata.get("models", [])}

    keys = [c for c in panel.columns if c not in META_COLUMNS]
    output = config.backtest.figure_dir
    output.mkdir(parents=True, exist_ok=True)

    prices = pd.read_parquet(config.data.raw_path)
    prices["date"] = pd.to_datetime(prices["date"])
    close = prices.set_index("date")[config.data.price_column]
    benchmark = "garch" if "garch" in keys else keys[0]

    figure_price_and_volatility(
        close, config.data.return_scale, output / "figure_4_1_price_and_volatility.png"
    )
    figure_regime_context(panel, config, output / "figure_4_2_regime_context.png")
    figure_regime_heatmap(panel, keys, labels, config, output / "figure_4_3_regime_qlike.png")
    figure_cumulative_loss(
        panel, keys, labels, benchmark, output / "figure_4_4_cumulative_qlike.png"
    )
    figure_var_breaches(
        panel, keys, labels, config.evaluation.var_confidence_levels[0],
        output / "figure_4_5_var_breaches.png",
    )
    figure_forecast_overlay(panel, keys, labels, output / "figure_4_6_forecast_overlay.png")
    n_written = 6

    logger.info("Wrote %d figures to %s", n_written, output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
