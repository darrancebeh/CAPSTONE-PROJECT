"""Accuracy metrics and the Mincer-Zarnowitz regression."""

import numpy as np
import pandas as pd
import pytest

from volforecast.evaluation.metrics import (
    absolute_error,
    loss_series,
    mincer_zarnowitz,
    out_of_sample_r2,
    qlike_loss,
    squared_error,
    summarise_forecast,
)


def test_error_metrics_on_a_known_case():
    proxy = np.array([1.0, 2.0, 3.0])
    forecast = np.array([1.0, 3.0, 1.0])
    assert squared_error(proxy, forecast) == pytest.approx([0.0, 1.0, 4.0])
    assert absolute_error(proxy, forecast) == pytest.approx([0.0, 1.0, 2.0])


def test_mincer_zarnowitz_recovers_a_perfect_forecast():
    proxy = np.array([0.5, 1.0, 2.0, 4.0, 8.0])
    result = mincer_zarnowitz(proxy, proxy)
    assert result["mz_intercept"] == pytest.approx(0.0, abs=1e-10)
    assert result["mz_slope"] == pytest.approx(1.0, abs=1e-10)
    assert result["mz_r2"] == pytest.approx(1.0, abs=1e-10)


def test_mincer_zarnowitz_recovers_a_known_scaling():
    """A forecast that is uniformly half the proxy implies a slope of two."""
    rng = np.random.default_rng(41)
    proxy = rng.gamma(2.0, 1.0, size=500)
    result = mincer_zarnowitz(proxy, proxy * 0.5)
    assert result["mz_slope"] == pytest.approx(2.0, rel=1e-9)
    assert result["mz_intercept"] == pytest.approx(0.0, abs=1e-9)


def test_out_of_sample_r2_on_a_perfect_forecast():
    proxy = np.array([1.0, 2.0, 4.0, 8.0])
    assert out_of_sample_r2(proxy, proxy) == pytest.approx(1.0)


def test_out_of_sample_r2_is_zero_for_the_sample_mean():
    proxy = np.array([1.0, 2.0, 4.0, 8.0])
    assert out_of_sample_r2(proxy, np.full(4, proxy.mean())) == pytest.approx(0.0)


def test_out_of_sample_r2_goes_negative_for_a_poor_forecast():
    """Unlike the Mincer-Zarnowitz R2, this version is unbounded below.

    The distinction matters when placing results next to studies that report a
    negative R2: they are using this definition, not the regression one.
    """
    proxy = np.array([1.0, 2.0, 4.0, 8.0])
    forecast = np.full(4, 50.0)
    assert out_of_sample_r2(proxy, forecast) < 0
    assert mincer_zarnowitz(proxy, forecast)["mz_r2"] >= 0


def test_alignment_drops_missing_observations_pairwise():
    proxy = pd.Series([1.0, np.nan, 3.0, 4.0])
    forecast = pd.Series([1.0, 2.0, np.nan, 4.0])
    assert len(qlike_loss(proxy, forecast)) == 2


def test_loss_series_keeps_the_date_index():
    index = pd.bdate_range("2020-01-01", periods=5)
    proxy = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0], index=index)
    forecast = pd.Series([1.0, 1.0, 1.0, 1.0, 1.0], index=index)

    losses = loss_series(proxy, forecast, "qlike")
    assert len(losses) == 4
    assert index[2] not in losses.index
    assert losses.index.equals(index.delete(2))


def test_summary_reports_every_headline_statistic():
    rng = np.random.default_rng(53)
    proxy = pd.Series(rng.gamma(2.0, 0.5, size=300))
    forecast = pd.Series(np.full(300, 1.0))

    summary = summarise_forecast(proxy, forecast)
    for field in ("qlike", "mse", "rmse", "mae", "mz_slope", "mz_r2", "bias_volatility"):
        assert field in summary
        assert np.isfinite(summary[field])
    assert summary["rmse"] == pytest.approx(np.sqrt(summary["mse"]))
    assert summary["n_observations"] == 300


def test_shape_mismatch_is_rejected():
    with pytest.raises(ValueError):
        squared_error(np.ones(5), np.ones(4))
