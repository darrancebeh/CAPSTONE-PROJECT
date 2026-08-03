"""Model estimation, the information boundary, and the walk-forward engine."""

import numpy as np
import pandas as pd
import pytest

from volforecast.backtest.engine import WalkForwardBacktester
from volforecast.backtest.regimes import UNCLASSIFIED, assign_regimes
from volforecast.config import (
    BacktestConfig,
    Config,
    EvaluationConfig,
    Regime,
    SplitConfig,
)
from volforecast.data.features import FeatureEngineer
from volforecast.models.base import ModelFitError, previous_session
from volforecast.models.garch import Egarch11, Garch11, GjrGarch11
from volforecast.models.lstm import LstmVolatilityModel, VolatilityLSTM


@pytest.fixture
def dataset(synthetic_prices, data_config, feature_config):
    prices = synthetic_prices.set_index("date")
    return FeatureEngineer(data_config, feature_config).build(prices)


@pytest.fixture
def config(tmp_path, data_config, feature_config):
    return Config(
        name="test",
        random_seed=7,
        data=data_config,
        features=feature_config,
        split=SplitConfig(initial_train_end="2019-06-28", min_train_observations=500),
        backtest=BacktestConfig(
            refit_every={"garch": 1, "egarch": 5, "gjr_garch": 5, "lstm_qlike": 200},
            forecast_dir=tmp_path / "forecasts",
            table_dir=tmp_path / "tables",
            figure_dir=tmp_path / "figures",
        ),
        models={},
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


class TestGarchFamily:
    @pytest.mark.parametrize("model_class", [Garch11, Egarch11, GjrGarch11])
    def test_estimation_recovers_a_persistent_variance_process(self, dataset, model_class):
        model = model_class()
        train_end = dataset.returns.index[-100]
        model.fit(dataset, train_end=train_end)

        parameters = model.parameters_snapshot()
        assert model.is_fitted
        assert "omega" in parameters
        assert parameters["beta[1]"] > 0.5  # the data-generating process is persistent

    def test_forecast_is_positive_and_finite(self, dataset):
        model = Garch11()
        target = dataset.returns.index[-1]
        model.fit(dataset, train_end=dataset.returns.index[-2])

        forecast = model.predict(dataset, target)
        assert np.isfinite(forecast)
        assert forecast > 0

    def test_predicting_before_fitting_raises(self, dataset):
        with pytest.raises(ModelFitError):
            Garch11().predict(dataset, dataset.returns.index[-1])

    def test_a_short_window_is_refused(self, dataset):
        with pytest.raises(ModelFitError, match="at least 100"):
            Garch11().fit(dataset, train_end=dataset.returns.index[50])

    def test_filtering_forward_updates_the_forecast_without_refitting(self, dataset):
        """Between refits the recursion must still absorb new returns."""
        model = Garch11()
        index = dataset.returns.index
        model.fit(dataset, train_end=index[-30])

        first = model.predict(dataset, index[-29])
        later = model.predict(dataset, index[-1])

        assert model.train_end == index[-30]
        assert first != pytest.approx(later)

    def test_gjr_identifies_the_leverage_parameter(self, dataset):
        model = GjrGarch11()
        model.fit(dataset, train_end=dataset.returns.index[-1])
        assert "gamma[1]" in model.parameters_snapshot()


class TestLstm:
    def test_output_is_strictly_positive(self):
        import torch

        network = VolatilityLSTM(n_features=4, hidden_size=8, num_layers=1, min_variance=1e-3)
        output = network(torch.randn(16, 5, 4) * 50.0)
        assert output.shape == (16,)
        assert bool((output >= 1e-3).all())

    def test_output_bias_centres_on_the_unconditional_variance(self):
        import torch

        network = VolatilityLSTM(n_features=3, hidden_size=8, num_layers=1, min_variance=1e-3)
        network.set_output_bias(2.5)
        network.eval()
        with torch.no_grad():
            output = network(torch.randn(8, 5, 3))
        assert float(output.mean()) == pytest.approx(2.5, rel=1e-4)

    def test_training_reduces_the_objective(self, dataset):
        model = LstmVolatilityModel(
            loss_name="qlike",
            sequence_length=5,
            hidden_size=8,
            num_layers=1,
            max_epochs=8,
            patience=8,
            seed=3,
        )
        model.fit(dataset, train_end=dataset.features.index[-1])

        history = model.history
        assert history.epochs_run > 0
        assert np.isfinite(history.best_validation_loss)
        assert history.train_losses[-1] <= history.train_losses[0]

    def test_refitting_the_same_window_is_deterministic(self, dataset):
        train_end = dataset.features.index[-50]
        target = dataset.features.index[-49]

        forecasts = []
        for _ in range(2):
            model = LstmVolatilityModel(
                loss_name="qlike", sequence_length=5, hidden_size=8, num_layers=1,
                max_epochs=5, patience=5, seed=11,
            )
            model.fit(dataset, train_end=train_end)
            forecasts.append(model.predict(dataset, target))

        assert forecasts[0] == pytest.approx(forecasts[1], rel=1e-12)

    def test_scaler_is_fitted_only_on_the_training_window(self, dataset):
        train_end = dataset.features.index[-200]
        model = LstmVolatilityModel(
            loss_name="qlike", sequence_length=5, hidden_size=8, num_layers=1,
            max_epochs=2, patience=2, seed=5,
        )
        model.fit(dataset, train_end=train_end)

        expected = dataset.features.loc[:train_end].to_numpy(dtype=np.float64).mean(axis=0)
        assert model.scaler.mean_ == pytest.approx(expected)


class TestInformationBoundary:
    def test_previous_session_is_strictly_earlier(self, dataset):
        index = dataset.returns.index
        assert previous_session(index, index[10]) == index[9]

    def test_previous_session_of_the_first_date_raises(self, dataset):
        index = dataset.returns.index
        with pytest.raises(ModelFitError):
            previous_session(index, index[0])

    def test_a_date_beyond_the_index_maps_to_the_final_session(self, dataset):
        index = dataset.returns.index
        assert previous_session(index, index[-1] + pd.Timedelta(days=30)) == index[-1]


class TestWalkForward:
    def test_engine_produces_one_forecast_per_evaluation_date(self, config, dataset):
        backtester = WalkForwardBacktester(config, dataset)
        result = backtester.run([Garch11(), Egarch11()])

        assert list(result.forecasts.columns) == ["garch", "egarch"]
        assert len(result.forecasts) == len(backtester.evaluation_dates)
        assert result.forecasts.notna().all().all()
        assert (result.forecasts > 0).all().all()

    def test_evaluation_window_starts_after_the_split(self, config, dataset):
        backtester = WalkForwardBacktester(config, dataset)
        assert backtester.evaluation_dates[0] > pd.Timestamp(config.split.initial_train_end)

    def test_refit_cadence_controls_the_number_of_estimations(self, config, dataset):
        backtester = WalkForwardBacktester(config, dataset)
        result = backtester.run([Garch11(), Egarch11()])

        summaries = {s.key: s for s in result.summaries}
        n_days = len(backtester.evaluation_dates)
        assert summaries["garch"].n_refits == n_days
        assert summaries["egarch"].n_refits == pytest.approx(np.ceil(n_days / 5), abs=1)

    def test_lookahead_audit_reports_no_difference(self, config, dataset):
        backtester = WalkForwardBacktester(config, dataset)
        audit = backtester.audit_no_lookahead(Garch11(), n_checkpoints=3)
        assert (audit["absolute_difference"] < 1e-10).all()

    def test_too_short_a_training_window_is_rejected(self, config, dataset):
        object.__setattr__(config.split, "min_train_observations", 10_000)
        with pytest.raises(ValueError, match="below the configured minimum"):
            WalkForwardBacktester(config, dataset)


class TestRegimes:
    def test_dates_are_labelled_by_window(self):
        index = pd.bdate_range("2020-01-01", periods=100)
        regimes = [
            Regime(name="Early", start="2020-01-01", end="2020-02-14"),
            Regime(name="Late", start="2020-02-17", end="2020-03-31"),
        ]
        labels = assign_regimes(index, regimes)
        assert labels.loc[pd.Timestamp("2020-01-15")] == "Early"
        assert labels.loc[pd.Timestamp("2020-03-02")] == "Late"
        assert set(labels.unique()) <= {"Early", "Late", UNCLASSIFIED}

    def test_overlapping_windows_are_rejected(self):
        index = pd.bdate_range("2020-01-01", periods=60)
        regimes = [
            Regime(name="A", start="2020-01-01", end="2020-02-28"),
            Regime(name="B", start="2020-02-01", end="2020-03-31"),
        ]
        with pytest.raises(ValueError, match="overlaps"):
            assign_regimes(index, regimes)

    def test_inverted_window_is_rejected(self):
        index = pd.bdate_range("2020-01-01", periods=10)
        with pytest.raises(ValueError, match="start after end"):
            assign_regimes(index, [Regime(name="Bad", start="2020-03-01", end="2020-01-01")])
