"""Calendar handling, ingestion repairs and feature construction."""

import numpy as np
import pandas as pd
import pytest

from volforecast.data.calendar import trading_sessions
from volforecast.data.features import FeatureEngineer, build_sequences, ewma_variance
from volforecast.data.loader import DataValidationError, MarketDataLoader


class TestTradingCalendar:
    def test_scheduled_holidays_are_excluded(self):
        sessions = trading_sessions("2023-01-01", "2023-12-31")
        for holiday in (
            "2023-01-02",  # New Year's Day observed
            "2023-01-16",  # Martin Luther King Jr Day
            "2023-04-07",  # Good Friday, a market holiday but not a federal one
            "2023-06-19",  # Juneteenth
            "2023-11-23",  # Thanksgiving
            "2023-12-25",  # Christmas Day
        ):
            assert pd.Timestamp(holiday) not in sessions

    def test_federal_holidays_that_the_exchange_ignores_remain_sessions(self):
        sessions = trading_sessions("2023-01-01", "2023-12-31")
        assert pd.Timestamp("2023-10-09") in sessions  # Columbus Day
        assert pd.Timestamp("2023-11-10") in sessions  # Veterans Day observed

    def test_a_saturday_new_year_does_not_close_the_preceding_friday(self):
        """The NYSE traded on 31 December 2021; a naive rule would skip it."""
        sessions = trading_sessions("2021-12-01", "2022-01-31")
        assert pd.Timestamp("2021-12-31") in sessions
        assert pd.Timestamp("2021-12-24") not in sessions  # Christmas observed

    def test_ad_hoc_closures_are_removed(self):
        sessions = trading_sessions("2018-12-01", "2018-12-31")
        assert pd.Timestamp("2018-12-05") not in sessions

    def test_juneteenth_only_applies_from_2022(self):
        assert pd.Timestamp("2021-06-18") in trading_sessions("2021-06-01", "2021-06-30")
        assert pd.Timestamp("2022-06-20") not in trading_sessions("2022-06-01", "2022-06-30")


class TestLoader:
    def _write(self, frame, data_config):
        data_config.raw_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(data_config.raw_path)

    def test_clean_input_passes_untouched(self, synthetic_prices, data_config):
        self._write(synthetic_prices, data_config)
        loader = MarketDataLoader(data_config)
        frame = loader.load()

        assert len(frame) == len(synthetic_prices)
        assert loader.report.missing_sessions == []
        assert loader.report.ohlc_violations == 0
        assert frame.index.is_monotonic_increasing

    def test_missing_sessions_are_forward_filled(self, synthetic_prices, data_config):
        dropped = synthetic_prices.drop(index=[100, 101, 102]).reset_index(drop=True)
        self._write(dropped, data_config)

        loader = MarketDataLoader(data_config)
        frame = loader.load()

        assert len(loader.report.missing_sessions) == 3
        assert loader.report.forward_filled_prices == 3
        assert len(frame) == len(synthetic_prices)
        # Filled sessions carry the previous close and no fabricated volume.
        filled = pd.Timestamp(synthetic_prices.loc[100, "date"])
        previous = pd.Timestamp(synthetic_prices.loc[99, "date"])
        assert frame.loc[filled, "close"] == pytest.approx(frame.loc[previous, "close"])
        assert frame.loc[filled, "volume"] == 0.0

    def test_duplicate_timestamps_are_collapsed(self, synthetic_prices, data_config):
        duplicated = pd.concat(
            [synthetic_prices, synthetic_prices.iloc[[50]]], ignore_index=True
        ).sort_values("date")
        self._write(duplicated, data_config)

        loader = MarketDataLoader(data_config)
        frame = loader.load()
        assert loader.report.duplicate_dates_dropped == 1
        assert not frame.index.duplicated().any()

    def test_broken_high_low_envelope_is_rejected(self, synthetic_prices, data_config):
        corrupted = synthetic_prices.copy()
        corrupted.loc[10, "high"] = corrupted.loc[10, "low"] - 1.0
        self._write(corrupted, data_config)

        with pytest.raises(DataValidationError, match="envelope"):
            MarketDataLoader(data_config).load()

    def test_non_positive_prices_are_rejected(self, synthetic_prices, data_config):
        corrupted = synthetic_prices.copy()
        corrupted.loc[20, "close"] = 0.0
        self._write(corrupted, data_config)

        with pytest.raises(DataValidationError, match="Non-positive"):
            MarketDataLoader(data_config).load()

    def test_missing_columns_are_rejected(self, synthetic_prices, data_config):
        self._write(synthetic_prices.drop(columns=["volume"]), data_config)
        with pytest.raises(DataValidationError, match="missing required columns"):
            MarketDataLoader(data_config).load()


class TestFeatures:
    def _dataset(self, synthetic_prices, data_config, feature_config):
        prices = synthetic_prices.set_index("date")
        return FeatureEngineer(data_config, feature_config).build(prices)

    def test_returns_and_proxy_are_consistent(
        self, synthetic_prices, data_config, feature_config
    ):
        dataset = self._dataset(synthetic_prices, data_config, feature_config)
        expected = dataset.returns.pow(2).clip(lower=data_config.min_variance_proxy)
        assert dataset.proxy.equals(expected)
        assert (dataset.proxy > 0).all()

    def test_no_missing_values_survive_the_warm_up(
        self, synthetic_prices, data_config, feature_config
    ):
        dataset = self._dataset(synthetic_prices, data_config, feature_config)
        assert not dataset.features.isna().any().any()
        assert np.isfinite(dataset.features.to_numpy()).all()

    def test_features_are_causal(self, synthetic_prices, data_config, feature_config):
        """Overwriting the tail of the price series must not alter earlier rows.

        This is the property that the whole no-look-ahead claim rests on: every
        predictor at date t is a function of prices at t and before, so
        truncating the future leaves it unchanged.
        """
        prices = synthetic_prices.set_index("date")
        cut = prices.index[600]

        full = FeatureEngineer(data_config, feature_config).build(prices)
        truncated = FeatureEngineer(data_config, feature_config).build(prices.loc[:cut])

        overlap = truncated.features.index
        pd.testing.assert_frame_equal(full.features.loc[overlap], truncated.features)

    def test_ewma_filter_matches_the_riskmetrics_recursion(self):
        proxy = pd.Series([1.0, 4.0, 2.0, 9.0], index=pd.bdate_range("2020-01-01", periods=4))
        lam = 0.94

        computed = ewma_variance(proxy, lam)
        expected = [1.0]
        for value in proxy.to_numpy()[1:]:
            expected.append(lam * expected[-1] + (1 - lam) * value)

        assert computed.to_numpy() == pytest.approx(np.array(expected))


class TestSequenceAlignment:
    def _frames(self):
        index = pd.bdate_range("2020-01-01", periods=60)
        features = pd.DataFrame(
            {"a": np.arange(60, dtype=float), "b": np.arange(60, dtype=float) * 2.0},
            index=index,
        )
        target = pd.Series(np.arange(60, dtype=float) * 10.0, index=index)
        return features, target, index

    def test_window_ends_on_the_session_before_the_target(self):
        features, target, index = self._frames()
        inputs, targets, served = build_sequences(features, target, 5, index[10:15])

        assert inputs.shape == (5, 5, 2)
        for i, date in enumerate(served):
            position = index.get_loc(date)
            # Final row of the window is the previous session, never the target.
            assert inputs[i, -1, 0] == pytest.approx(float(position - 1))
            assert inputs[i, 0, 0] == pytest.approx(float(position - 5))
            assert targets[i] == pytest.approx(float(position) * 10.0)

    def test_targets_are_never_visible_in_the_inputs(self):
        features, target, index = self._frames()
        # Make the target a feature column as well, the worst case for leakage.
        features = features.assign(target_copy=target)
        inputs, targets, _ = build_sequences(features, target, 5, index[20:30])

        for i in range(len(targets)):
            assert not np.isclose(inputs[i, :, 2], targets[i]).any()

    def test_future_rows_cannot_affect_earlier_sequences(self):
        features, target, index = self._frames()
        early = index[10:20]

        original, original_targets, _ = build_sequences(features, target, 5, early)

        corrupted = features.copy()
        corrupted.iloc[25:] = 999.0
        corrupted_inputs, corrupted_targets, _ = build_sequences(corrupted, target, 5, early)

        np.testing.assert_array_equal(original, corrupted_inputs)
        np.testing.assert_array_equal(original_targets, corrupted_targets)

    def test_dates_without_enough_history_are_skipped(self):
        features, target, index = self._frames()
        _, _, served = build_sequences(features, target, 5, index[:10])
        assert served[0] == index[5]
        assert len(served) == 5

    def test_empty_request_returns_empty_arrays(self):
        features, target, _ = self._frames()
        inputs, targets, served = build_sequences(features, target, 5, pd.DatetimeIndex([]))
        assert inputs.shape == (0, 5, 2)
        assert len(targets) == 0
        assert len(served) == 0
