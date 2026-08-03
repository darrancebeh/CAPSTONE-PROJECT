"""Kupiec and Christoffersen coverage tests against hand-computed values."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from volforecast.evaluation.var_backtest import (
    backtest_var,
    christoffersen_independence,
    conditional_coverage,
    kupiec_pof,
    value_at_risk,
)


def test_var_uses_the_gaussian_quantile():
    variance = np.array([1.0, 4.0])
    computed = value_at_risk(variance, confidence=0.95)
    assert computed[0] == pytest.approx(1.6448536, rel=1e-6)
    assert computed[1] == pytest.approx(2.0 * 1.6448536, rel=1e-6)

    ninety_nine = value_at_risk(np.array([1.0]), confidence=0.99)
    assert ninety_nine[0] == pytest.approx(2.3263479, rel=1e-6)


def test_kupiec_is_exactly_zero_when_coverage_is_nominal():
    statistic, p_value = kupiec_pof(n_observations=1000, n_exceptions=50, p=0.05)
    assert statistic == pytest.approx(0.0, abs=1e-12)
    assert p_value == pytest.approx(1.0)


def test_kupiec_matches_the_closed_form_likelihood_ratio():
    n, x, p = 1000, 100, 0.05
    statistic, p_value = kupiec_pof(n, x, p)

    pi_hat = x / n
    expected = -2.0 * (
        (n - x) * np.log(1 - p)
        + x * np.log(p)
        - (n - x) * np.log(1 - pi_hat)
        - x * np.log(pi_hat)
    )
    assert statistic == pytest.approx(expected, rel=1e-10)
    assert p_value == pytest.approx(stats.chi2.sf(expected, df=1), rel=1e-10)
    assert p_value < 1e-6


def test_kupiec_handles_a_sample_with_no_exceptions():
    """0 * log(0) must evaluate to zero rather than propagating a NaN."""
    statistic, p_value = kupiec_pof(n_observations=500, n_exceptions=0, p=0.05)
    assert np.isfinite(statistic)
    assert statistic == pytest.approx(-2.0 * 500 * np.log(0.95), rel=1e-10)
    assert p_value < 0.01


def test_independence_is_not_rejected_without_exceptions():
    statistic, p_value = christoffersen_independence(np.zeros(500, dtype=int))
    assert statistic == 0.0
    assert p_value == 1.0


def test_independence_detects_clustered_breaches():
    exceptions = np.zeros(200, dtype=int)
    exceptions[50:60] = 1  # a single run of ten consecutive breaches

    statistic, p_value = christoffersen_independence(exceptions)
    assert statistic == pytest.approx(60.33, abs=0.05)
    assert p_value < 1e-10


def test_independence_accepts_evenly_spaced_breaches():
    exceptions = np.zeros(400, dtype=int)
    exceptions[::20] = 1
    _, p_value = christoffersen_independence(exceptions)
    assert p_value > 0.05


def test_conditional_coverage_is_the_sum_of_both_statistics():
    statistic, p_value = conditional_coverage(4.0, 2.0)
    assert statistic == pytest.approx(6.0)
    assert p_value == pytest.approx(stats.chi2.sf(6.0, df=2), rel=1e-10)


def test_a_correctly_specified_forecast_passes_every_test():
    """Returns drawn from the forecast distribution should clear all three."""
    rng = np.random.default_rng(2024)
    n = 4000
    index = pd.bdate_range("2010-01-01", periods=n)

    variance = pd.Series(rng.gamma(shape=4.0, scale=0.25, size=n), index=index)
    returns = pd.Series(np.sqrt(variance.to_numpy()) * rng.standard_normal(n), index=index)

    result = backtest_var(returns, variance, confidence=0.95, model="oracle")
    assert result.exception_rate == pytest.approx(0.05, abs=0.012)
    outcome = result.passes(alpha=0.01)
    assert outcome["unconditional_coverage"]
    assert outcome["independence"]
    assert outcome["conditional_coverage"]


def test_a_systematically_low_forecast_is_rejected():
    rng = np.random.default_rng(99)
    n = 2000
    index = pd.bdate_range("2010-01-01", periods=n)

    true_variance = rng.gamma(shape=4.0, scale=0.25, size=n)
    returns = pd.Series(np.sqrt(true_variance) * rng.standard_normal(n), index=index)
    understated = pd.Series(true_variance * 0.25, index=index)

    result = backtest_var(returns, understated, confidence=0.95, model="understated")
    assert result.exception_rate > 0.05
    assert not result.passes()["unconditional_coverage"]
