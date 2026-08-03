"""End-to-end orchestration: ingest, engineer, backtest, evaluate, persist."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

from .backtest.cache import ForecastStore
from .backtest.engine import BacktestResult, ModelRunSummary, WalkForwardBacktester
from .config import Config
from .data.features import Dataset, FeatureEngineer
from .data.loader import DataQualityReport, load_market_data
from .evaluation.scorecard import Evaluator
from .models import build_model_suite
from .models.base import VolatilityModel
from .utils import get_logger, set_global_seed

logger = get_logger(__name__)


@dataclass
class PipelineOutput:
    dataset: Dataset
    data_report: DataQualityReport
    result: BacktestResult
    tables: Dict[str, pd.DataFrame]
    audit: Optional[pd.DataFrame] = None


def prepare_dataset(config: Config) -> tuple[Dataset, DataQualityReport]:
    """Load the raw feed and derive returns, proxies and predictors."""
    prices, report = load_market_data(config.data)
    dataset = FeatureEngineer(config.data, config.features).build(prices)

    processed_dir = config.data.processed_dir
    processed_dir.mkdir(parents=True, exist_ok=True)
    dataset.features.to_parquet(processed_dir / "features.parquet")
    pd.concat([dataset.returns, dataset.proxy], axis=1).to_parquet(
        processed_dir / "returns_and_proxy.parquet"
    )
    return dataset, report


def run_pipeline(
    config: Optional[Config] = None,
    models: Optional[Sequence[VolatilityModel]] = None,
    run_audit: bool = True,
    save: bool = True,
) -> PipelineOutput:
    config = config or Config.load()
    config.ensure_output_dirs()
    set_global_seed(config.random_seed)

    dataset, data_report = prepare_dataset(config)
    logger.info("Data quality report:\n%s", data_report.summary())

    suite: List[VolatilityModel] = list(models) if models is not None else build_model_suite(config)
    backtester = WalkForwardBacktester(config, dataset)
    result = backtester.run(suite)

    audit = None
    if run_audit:
        audit = run_lookahead_audit(backtester, suite)

    evaluator = Evaluator(config, result)
    tables = evaluator.build_all()

    if save:
        store = ForecastStore(config.backtest.forecast_dir)
        metadata = dict(result.metadata)
        metadata["data_quality"] = data_report.to_dict()
        if audit is not None:
            metadata["lookahead_audit_max_difference"] = float(
                audit["absolute_difference"].max()
            )
        store.save(forecasts=result.forecasts, metadata=metadata, fit_log=result.fit_log)

        panel = pd.concat(
            [result.proxy, result.returns, result.regimes, result.forecasts], axis=1
        )
        panel.to_parquet(config.backtest.forecast_dir / "evaluation_panel.parquet")

        evaluator.save(tables)
        if audit is not None:
            audit.to_csv(config.backtest.table_dir / "lookahead_audit.csv", index=False)

    return PipelineOutput(
        dataset=dataset,
        data_report=data_report,
        result=result,
        tables=tables,
        audit=audit,
    )


def run_lookahead_audit(
    backtester: WalkForwardBacktester,
    models: Sequence[VolatilityModel],
    tolerance: float = 1e-8,
) -> pd.DataFrame:
    """Verify that no model changes its forecast when the future is removed."""
    frames = []
    for model in models:
        # Each checkpoint costs two full re-estimations, so the neural models
        # are sampled less densely than the closed-form ones.
        n_checkpoints = 2 if model.family == "Deep learning" else 5
        frames.append(backtester.audit_no_lookahead(model, n_checkpoints=n_checkpoints))
    audit = pd.concat(frames, ignore_index=True)

    worst = float(audit["absolute_difference"].max())
    if worst > tolerance:
        offenders = sorted(
            audit.loc[audit["absolute_difference"] > tolerance, "model"].unique()
        )
        raise RuntimeError(
            f"Look-ahead audit failed for {offenders}: forecasts changed by up to {worst:.3e} "
            "when observations after the information boundary were removed"
        )

    logger.info(
        "Look-ahead audit passed for %d models across %d checkpoints (max difference %.2e)",
        audit["model"].nunique(),
        len(audit),
        worst,
    )
    return audit


def evaluate_cached_run(config: Optional[Config] = None, save: bool = True):
    """Rebuild every result table from a cached run without re-forecasting.

    The walk-forward is the expensive part and its output is deterministic, so
    changing how results are presented should not require re-estimating
    anything.
    """
    config = config or Config.load()
    store = ForecastStore(config.backtest.forecast_dir)
    metadata = store.load_metadata()
    panel = load_evaluation_panel(config)

    forecasts = panel.drop(columns=["proxy", "return", "regime"])
    summaries = [
        ModelRunSummary(
            key=entry["key"],
            label=entry["label"],
            family=entry["family"],
            n_forecasts=entry["n_forecasts"],
            n_refits=entry["n_refits"],
            n_failures=entry["n_failures"],
            fit_seconds=entry["fit_seconds"],
            predict_seconds=entry["predict_seconds"],
        )
        for entry in metadata.get("models", [])
    ]

    result = BacktestResult(
        forecasts=forecasts,
        proxy=panel["proxy"],
        returns=panel["return"],
        regimes=panel["regime"],
        summaries=summaries,
        fit_log=store.load_fit_log(),
        metadata=metadata,
    )

    evaluator = Evaluator(config, result)
    tables = evaluator.build_all()
    if save:
        evaluator.save(tables)
    return result, tables


def load_cached_results(config: Optional[Config] = None) -> tuple[pd.DataFrame, Dict]:
    """Reload a completed run without recomputing anything."""
    config = config or Config.load()
    store = ForecastStore(config.backtest.forecast_dir)
    return store.load_forecasts(), store.load_metadata()


def load_evaluation_panel(config: Optional[Config] = None) -> pd.DataFrame:
    config = config or Config.load()
    path = Path(config.backtest.forecast_dir) / "evaluation_panel.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No evaluation panel at {path}. Run the pipeline first.")
    panel = pd.read_parquet(path)
    panel.index = pd.DatetimeIndex(panel.index)
    return panel
