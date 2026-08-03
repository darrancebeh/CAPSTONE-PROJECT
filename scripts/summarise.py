"""Condense the backtest into the five tables used in the report.

The full evaluation writes fourteen tables, which is the right level of detail
for auditing a run and the wrong level for presenting one. This script selects
the five that carry the argument, rounds them for print, and writes them to
results/summary as both CSV and a single Markdown file.

    python scripts/summarise.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd

from volforecast.backtest.cache import ForecastStore
from volforecast.config import Config
from volforecast.evaluation.diebold_mariano import diebold_mariano
from volforecast.evaluation.metrics import loss_series, summarise_forecast
from volforecast.evaluation.var_backtest import run_var_backtests, var_results_to_frame
from volforecast.pipeline import load_evaluation_panel
from volforecast.utils import get_logger

logger = get_logger("summarise")

META_COLUMNS = ["proxy", "return", "regime"]
BENCHMARK = "garch"
TRADING_DAYS = 252


def table_descriptive(config: Config) -> pd.DataFrame:
    """Distributional summary of the return series over the full sample.

    Reported because the case for using ARCH-type models at all rests on these
    numbers: excess kurtosis and a rejected normality test are what rule out a
    constant-variance Gaussian model.
    """
    from scipy import stats

    from volforecast.data.features import FeatureEngineer
    from volforecast.data.loader import load_market_data

    prices, _ = load_market_data(config.data)
    dataset = FeatureEngineer(config.data, config.features).build(prices)

    rows = []
    for name, series in (
        ("Return (%)", dataset.returns),
        ("Squared return (%²)", dataset.proxy),
    ):
        values = series.to_numpy(dtype=float)
        _, jb_p = stats.jarque_bera(values)
        rows.append(
            {
                "Series": name,
                "N": len(values),
                "Mean": values.mean(),
                "Std. dev.": values.std(ddof=1),
                "Min": values.min(),
                "Max": values.max(),
                "Skewness": float(stats.skew(values)),
                "Kurtosis": float(stats.kurtosis(values, fisher=False)),
                "JB p-value": float(jb_p),
            }
        )
    return pd.DataFrame(rows).set_index("Series").round(4)


def table_accuracy(panel: pd.DataFrame, keys, labels: Dict[str, str]) -> pd.DataFrame:
    rows = []
    for key in keys:
        summary = summarise_forecast(panel["proxy"], panel[key])
        rows.append(
            {
                "Model": labels[key],
                "QLIKE": summary["qlike"],
                "RMSE": summary["rmse"],
                "MAE": summary["mae"],
                "R2": summary["r2_oos"],
            }
        )
    frame = pd.DataFrame(rows).set_index("Model")
    frame.insert(1, "Rank (QLIKE)", frame["QLIKE"].rank(method="min").astype(int))
    return frame.round(4)


def table_regime(panel: pd.DataFrame, keys, labels: Dict[str, str], config: Config) -> pd.DataFrame:
    names = [r.name for r in config.regimes if (panel["regime"] == r.name).any()]
    data = {}
    for name in names:
        subset = panel.loc[panel["regime"] == name]
        data[name] = {
            labels[k]: loss_series(subset["proxy"], subset[k], "qlike").mean() for k in keys
        }
    data["Full sample"] = {
        labels[k]: loss_series(panel["proxy"], panel[k], "qlike").mean() for k in keys
    }
    frame = pd.DataFrame(data).loc[[labels[k] for k in keys]]
    frame.index.name = "Model"
    return frame.round(3)


def table_significance(
    panel: pd.DataFrame, keys, labels: Dict[str, str], config: Config
) -> pd.DataFrame:
    """Every model against the GARCH(1,1) benchmark, full sample then by regime."""
    rows = []

    def test(subset: pd.DataFrame, key: str, scope: str):
        paired = pd.concat(
            [
                loss_series(subset["proxy"], subset[key], "qlike").rename("a"),
                loss_series(subset["proxy"], subset[BENCHMARK], "qlike").rename("b"),
            ],
            axis=1,
        ).dropna()
        if len(paired) < 20:
            return None
        outcome = diebold_mariano(
            paired["a"].to_numpy(),
            paired["b"].to_numpy(),
            horizon=1,
            lags=config.evaluation.hac_lags,
            harvey_correction=config.evaluation.harvey_correction,
            model_a=labels[key],
            model_b=labels[BENCHMARK],
        )
        alpha = config.evaluation.var_test_significance
        if outcome.p_value >= alpha:
            verdict = "No difference"
        else:
            verdict = f"{outcome.favoured} better"
        return {
            "Scope": scope,
            "Model vs GARCH(1,1)": labels[key],
            "DM statistic": round(outcome.statistic, 3),
            "p-value": round(outcome.p_value, 4),
            "Verdict": verdict,
        }

    for key in keys:
        if key == BENCHMARK:
            continue
        record = test(panel, key, "Full sample")
        if record:
            rows.append(record)

    focus = [k for k in ("lstm_qlike",) if k in keys]
    for name in [r.name for r in config.regimes]:
        subset = panel.loc[panel["regime"] == name]
        if subset.empty:
            continue
        for key in focus:
            record = test(subset, key, name)
            if record:
                rows.append(record)

    return pd.DataFrame(rows)


def table_var(panel: pd.DataFrame, keys, labels: Dict[str, str], config: Config) -> pd.DataFrame:
    results = run_var_backtests(
        returns=panel["return"],
        forecasts=panel[list(keys)],
        confidence_levels=[0.95],
        labels=labels,
    )
    frame = var_results_to_frame(results, alpha=config.evaluation.var_test_significance)
    out = frame[
        ["model", "n_exceptions", "expected_exceptions", "kupiec_p", "cc_p", "pass_conditional"]
    ].copy()
    out.columns = ["Model", "Breaches", "Expected", "Kupiec p", "Joint p", "Passes"]
    out["Passes"] = np.where(out["Passes"], "Yes", "No")
    return out.round(4)


def table_capacity(panel: pd.DataFrame, keys, labels, metadata) -> pd.DataFrame:
    runtime = {entry["key"]: entry for entry in metadata.get("models", [])}
    worst = float(panel["proxy"].max())
    rows = []
    for key in keys:
        highest = float(panel[key].max())
        rows.append(
            {
                "Model": labels[key],
                "Highest forecast (ann. %)": np.sqrt(highest * TRADING_DAYS),
                "Share of worst day": highest / worst,
                "Refits": runtime.get(key, {}).get("n_refits", np.nan),
                "Runtime (s)": runtime.get(key, {}).get("total_seconds", np.nan),
            }
        )
    frame = pd.DataFrame(rows).set_index("Model")
    return frame.round(3)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()

    config = Config.load(args.config)
    panel = load_evaluation_panel(config)
    metadata = ForecastStore(config.backtest.forecast_dir).load_metadata()
    labels = {entry["key"]: entry["label"] for entry in metadata.get("models", [])}
    keys = [c for c in panel.columns if c not in META_COLUMNS]

    tables = {
        "table_1_descriptive": table_descriptive(config),
        "table_2_accuracy": table_accuracy(panel, keys, labels),
        "table_3_regime_qlike": table_regime(panel, keys, labels, config),
        "table_4_significance": table_significance(panel, keys, labels, config),
        "table_5_var_coverage": table_var(panel, keys, labels, config),
        "table_6_capacity_and_cost": table_capacity(panel, keys, labels, metadata),
    }

    titles = {
        "table_1_descriptive": "Table 4.1. Descriptive statistics, full sample",
        "table_2_accuracy": "Table 4.2. Out-of-sample forecast accuracy, 1,760 sessions",
        "table_3_regime_qlike": "Table 4.3. Average QLIKE by market regime",
        "table_4_significance": "Table 4.4. Diebold-Mariano tests against GARCH(1,1)",
        "table_5_var_coverage": "Table 4.5. 95% Value-at-Risk coverage",
        "table_6_capacity_and_cost": "Table 4.6. Forecast ceiling and computational cost",
    }

    output = Path(config.backtest.table_dir).parent / "summary"
    output.mkdir(parents=True, exist_ok=True)
    # Clear previous output so that a renumbered or renamed table cannot leave
    # a stale file behind to be mistaken for part of the current run.
    for existing in output.glob("table_*.csv"):
        existing.unlink()

    pd.set_option("display.width", 200)
    lines = ["# Findings summary", ""]
    lines.append(
        f"Sample {metadata['sample_start']} to {metadata['sample_end']}. "
        f"Out-of-sample {metadata['evaluation_start']} to {metadata['evaluation_end']}, "
        f"{metadata['n_evaluation_days']} sessions."
    )
    lines.append("")

    for name, table in tables.items():
        keep_index = not isinstance(table.index, pd.RangeIndex)
        table.to_csv(output / f"{name}.csv", index=keep_index)

        print(f"\n{titles[name]}")
        print(table.to_string(index=keep_index))

        lines.append(f"## {titles[name]}")
        lines.append("")
        lines.append(table.to_markdown(index=keep_index))
        lines.append("")

    (output / "findings.md").write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote 5 tables and findings.md to %s", output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
