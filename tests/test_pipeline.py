"""End-to-end pipeline behaviour on a synthetic sample.

The unit tests elsewhere check components in isolation. This module runs the
whole chain once, from a raw parquet on disk through to written result tables,
so that the orchestration, the scorecard assembly and the persistence layer are
exercised rather than only the pieces they call.
"""

import json

import pandas as pd
import pytest

from volforecast.backtest.cache import ForecastStore
from volforecast.config import (
    BacktestConfig,
    Config,
    EvaluationConfig,
    Regime,
    SplitConfig,
)
from volforecast.models import build_model_suite
from volforecast.models.garch import Egarch11, Garch11
from volforecast.pipeline import evaluate_cached_run, prepare_dataset, run_pipeline


@pytest.fixture
def project(tmp_path, synthetic_prices, data_config, feature_config):
    """A complete configuration whose paths all live under tmp_path."""
    data_config.raw_path.parent.mkdir(parents=True, exist_ok=True)
    synthetic_prices.to_parquet(data_config.raw_path)

    return Config(
        name="integration",
        random_seed=13,
        data=data_config,
        features=feature_config,
        split=SplitConfig(initial_train_end="2019-06-28", min_train_observations=500),
        backtest=BacktestConfig(
            refit_every={"garch": 10, "egarch": 10},
            forecast_dir=tmp_path / "forecasts",
            table_dir=tmp_path / "tables",
            figure_dir=tmp_path / "figures",
        ),
        models={
            "garch_family": {"distribution": "normal", "mean": "Constant"},
            "lstm": {
                "hidden_size": 8, "num_layers": 1, "dropout": 0.1,
                "learning_rate": 1e-3, "weight_decay": 0.0, "batch_size": 32,
                "max_epochs": 2, "patience": 2, "validation_fraction": 0.15,
                "grad_clip": 1.0, "min_variance": 1e-3,
            },
        },
        regimes=[
            Regime(name="First half", start="2019-07-01", end="2019-09-30"),
            Regime(name="Second half", start="2019-10-01", end="2019-12-31"),
        ],
        evaluation=EvaluationConfig(
            hac_lags=None,
            harvey_correction=True,
            var_confidence_levels=[0.95],
            var_test_significance=0.05,
        ),
    )


@pytest.fixture
def completed(project):
    """Run the pipeline once and share the result across the tests below."""
    return run_pipeline(
        config=project, models=[Garch11(), Egarch11()], run_audit=True, save=True
    )


class TestPipeline:
    def test_produces_a_forecast_for_every_evaluation_date(self, completed):
        forecasts = completed.result.forecasts
        assert list(forecasts.columns) == ["garch", "egarch"]
        assert forecasts.notna().all().all()
        assert (forecasts > 0).all().all()

    def test_data_quality_report_is_populated(self, completed):
        report = completed.data_report.to_dict()
        assert report["rows_clean"] > 0
        assert report["ohlc_violations"] == 0
        assert report["missing_sessions"] == 0

    def test_lookahead_audit_runs_and_finds_nothing(self, completed):
        assert completed.audit is not None
        assert (completed.audit["absolute_difference"] == 0).all()

    def test_every_expected_table_is_built(self, completed):
        expected = {
            "overall_metrics", "rank_comparison", "regime_qlike", "regime_mse",
            "regime_detail", "regime_profile", "dm_qlike", "dm_mse",
            "dm_matrix_qlike", "var_backtests", "runtime",
        }
        assert expected.issubset(set(completed.tables))
        assert not completed.tables["overall_metrics"].empty

    def test_headline_metrics_are_finite(self, completed):
        overall = completed.tables["overall_metrics"]
        for column in ("qlike", "rmse", "mae", "r2_oos"):
            assert overall[column].notna().all()
        assert (overall["qlike"] > 0).all()


class TestPersistence:
    def test_artefacts_are_written_and_reload(self, completed, project):
        store = ForecastStore(project.backtest.forecast_dir)
        assert store.exists()

        reloaded = store.load_forecasts()
        pd.testing.assert_frame_equal(reloaded, completed.result.forecasts)

        metadata = store.load_metadata()
        assert metadata["symbol"] == project.data.symbol
        assert metadata["n_evaluation_days"] == len(completed.result.forecasts)
        assert len(metadata["models"]) == 2

    def test_metadata_is_valid_json_on_disk(self, completed, project):
        path = project.backtest.forecast_dir / "run_metadata.json"
        with open(path, encoding="utf-8") as handle:
            assert json.load(handle)["config_name"] == "integration"

    def test_tables_are_written_as_csv(self, completed, project):
        written = {p.stem for p in project.backtest.table_dir.glob("*.csv")}
        assert "overall_metrics" in written
        assert "lookahead_audit" in written

    def test_cached_run_reproduces_the_tables(self, completed, project):
        """Re-deriving from cache must match the tables built in-memory."""
        _, tables = evaluate_cached_run(project, save=False)
        pd.testing.assert_frame_equal(
            tables["overall_metrics"], completed.tables["overall_metrics"]
        )


class TestModelRegistry:
    def test_build_model_suite_returns_the_five_compared_models(self, project):
        suite = build_model_suite(project)
        assert [m.key for m in suite] == [
            "garch", "egarch", "gjr_garch", "lstm_qlike", "lstm_mse"
        ]

    def test_both_networks_share_one_architecture(self, project):
        """The loss ablation is only valid if the architectures are identical."""
        suite = {m.key: m for m in build_model_suite(project)}
        qlike, mse = suite["lstm_qlike"], suite["lstm_mse"]
        for attribute in ("hidden_size", "num_layers", "dropout", "sequence_length"):
            assert getattr(qlike, attribute) == getattr(mse, attribute)
        assert qlike.loss_name == "qlike" and mse.loss_name == "mse"


class TestDatasetPreparation:
    def test_processed_files_are_written(self, project):
        prepare_dataset(project)
        assert (project.data.processed_dir / "features.parquet").exists()
        assert (project.data.processed_dir / "returns_and_proxy.parquet").exists()
