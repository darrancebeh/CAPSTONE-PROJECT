"""Robustness of the headline conclusion to the LSTM retraining cadence.

The main backtest re-optimises the GARCH family every session but retrains the
network only twice a year, on the grounds that a network of this cost would be
operated that way. That asymmetry is the most obvious objection to the
comparison: it could be argued that the econometric models win because they are
allowed to update more often, not because they forecast better.

This script settles the question directly by re-running only the neural models
at a range of cadences and re-testing each against the cached GARCH forecasts.
If the ranking is unchanged at a 21-session cadence, the conclusion does not
rest on the retraining schedule.

    python scripts/sensitivity.py --cadences 21 63 126 252
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import _bootstrap  # noqa: F401

import pandas as pd

from volforecast.backtest.cache import ForecastStore
from volforecast.backtest.engine import WalkForwardBacktester
from volforecast.config import Config
from volforecast.evaluation.diebold_mariano import diebold_mariano
from volforecast.evaluation.metrics import loss_series
from volforecast.models import build_model_suite
from volforecast.pipeline import prepare_dataset
from volforecast.utils import get_logger, set_global_seed

logger = get_logger("sensitivity")

NEURAL_KEYS = ("lstm_qlike", "lstm_mse")
BENCHMARK_KEY = "garch"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--cadences",
        type=int,
        nargs="+",
        default=[21, 63, 126, 252],
        help="Sessions between LSTM retrainings",
    )
    parser.add_argument(
        "--benchmark",
        default=BENCHMARK_KEY,
        help="Cached model key to test the network against",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = Config.load(args.config)
    set_global_seed(config.random_seed)

    store = ForecastStore(config.backtest.forecast_dir)
    if not store.exists():
        logger.error("No cached backtest found. Run scripts/run_pipeline.py first.")
        return 1

    cached = store.load_forecasts()
    metadata = store.load_metadata()
    labels = {entry["key"]: entry["label"] for entry in metadata.get("models", [])}

    if args.benchmark not in cached.columns:
        logger.error("Benchmark '%s' is not in the cached panel", args.benchmark)
        return 1

    dataset, _ = prepare_dataset(config)
    backtester = WalkForwardBacktester(config, dataset)
    proxy = dataset.proxy.loc[backtester.evaluation_dates]
    benchmark_loss = loss_series(proxy, cached[args.benchmark], "qlike")

    records: List[dict] = []
    for cadence in args.cadences:
        logger.info("Re-running the neural models at a %d-session cadence", cadence)
        object.__setattr__(
            config.backtest,
            "refit_every",
            {**config.backtest.refit_every, **{key: cadence for key in NEURAL_KEYS}},
        )

        suite = [m for m in build_model_suite(config) if m.key in NEURAL_KEYS]
        result = backtester.run(suite)

        for summary in result.summaries:
            forecasts = result.forecasts[summary.key]
            model_loss = loss_series(proxy, forecasts, "qlike")

            paired = pd.concat(
                [model_loss.rename("model"), benchmark_loss.rename("benchmark")], axis=1
            ).dropna()
            test = diebold_mariano(
                paired["model"].to_numpy(),
                paired["benchmark"].to_numpy(),
                horizon=1,
                lags=config.evaluation.hac_lags,
                harvey_correction=config.evaluation.harvey_correction,
                model_a=summary.label,
                model_b=labels.get(args.benchmark, args.benchmark),
                loss_name="qlike",
            )

            records.append(
                {
                    "cadence": cadence,
                    "model": summary.label,
                    "n_refits": summary.n_refits,
                    "qlike": float(model_loss.mean()),
                    "benchmark_qlike": float(benchmark_loss.mean()),
                    "dm_statistic": test.statistic,
                    "p_value": test.p_value,
                    "favoured": test.favoured,
                    "total_seconds": round(summary.total_seconds, 1),
                }
            )

    frame = pd.DataFrame(records)
    output = config.backtest.table_dir / "sensitivity_refit_cadence.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)

    pd.set_option("display.width", 180)
    print("\nLSTM retraining cadence sensitivity")
    print(frame.round(4).to_string(index=False))

    benchmark_label = labels.get(args.benchmark, args.benchmark)
    overturned = frame.loc[frame["favoured"] != benchmark_label]
    print()
    if overturned.empty:
        print(
            f"{benchmark_label} retains the lower QLIKE at every cadence tested, so the "
            "conclusion does not depend on how often the network is retrained."
        )
    else:
        print("The ranking reverses at these settings:")
        print(overturned[["cadence", "model", "qlike", "p_value"]].round(4).to_string(index=False))

    logger.info("Wrote %s", output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
