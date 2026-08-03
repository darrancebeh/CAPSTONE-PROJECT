"""Command line entry point for the full backtest.

    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --config config/config.yaml --no-audit
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

import pandas as pd

from volforecast.config import Config
from volforecast.models import build_model_suite
from volforecast.pipeline import evaluate_cached_run, run_pipeline
from volforecast.utils import get_logger

logger = get_logger("run_pipeline")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to the YAML configuration (defaults to config/config.yaml)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        metavar="KEY",
        help="Restrict the run to these model keys (default: every model in the suite)",
    )
    parser.add_argument(
        "--no-audit",
        action="store_true",
        help="Skip the look-ahead audit, which re-estimates models at sampled checkpoints",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Compute results without writing any artefacts to disk",
    )
    parser.add_argument(
        "--tables-only",
        action="store_true",
        help="Rebuild the result tables from the cached forecasts without re-running "
        "the walk-forward",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = Config.load(args.config)

    if args.tables_only:
        _, tables = evaluate_cached_run(config, save=not args.no_save)
        report(tables)
        return 0

    suite = build_model_suite(config)
    if args.models:
        requested = set(args.models)
        available = {model.key for model in suite}
        unknown = requested - available
        if unknown:
            logger.error("Unknown model keys: %s. Available: %s", sorted(unknown), sorted(available))
            return 1
        suite = [model for model in suite if model.key in requested]

    output = run_pipeline(
        config=config,
        models=suite,
        run_audit=not args.no_audit,
        save=not args.no_save,
    )
    report(output.tables)
    return 0


def report(tables) -> None:
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 40)

    print("\nOut-of-sample accuracy")
    columns = [
        "n_observations",
        "qlike",
        "mse",
        "rmse",
        "mae",
        "mz_slope",
        "mz_r2",
        "max_forecast",
        "tail_reach",
    ]
    print(tables["overall_metrics"][columns].round(4).to_string())

    print("\nAverage QLIKE by regime")
    print(tables["regime_qlike"].round(4).to_string())

    dm = tables["dm_qlike"]
    print("\nDiebold-Mariano tests on QLIKE")
    print(
        dm[["model_a", "model_b", "mean_differential", "dm_statistic", "p_value", "favoured"]]
        .round(4)
        .to_string(index=False)
    )

    print("\nVaR coverage")
    var_columns = [
        "model",
        "confidence",
        "n_exceptions",
        "expected_exceptions",
        "kupiec_p",
        "christoffersen_p",
        "cc_p",
        "pass_conditional",
    ]
    print(tables["var_backtests"][var_columns].round(4).to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
