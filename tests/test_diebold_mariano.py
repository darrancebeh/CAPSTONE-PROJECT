"""Diebold-Mariano test behaviour, including the degenerate cases."""

import numpy as np
import pytest

from volforecast.evaluation.diebold_mariano import (
    diebold_mariano,
    newey_west_variance,
    schwert_lags,
)


def test_newey_west_with_zero_lags_is_the_sample_variance():
    rng = np.random.default_rng(3)
    values = rng.standard_normal(500)
    expected = float(((values - values.mean()) ** 2).mean())
    assert newey_west_variance(values, lags=0) == pytest.approx(expected)


def test_newey_west_inflates_the_variance_of_a_persistent_series():
    """Positively autocorrelated losses must widen the standard error."""
    rng = np.random.default_rng(11)
    innovations = rng.standard_normal(2000)
    persistent = np.zeros(2000)
    for t in range(1, 2000):
        persistent[t] = 0.7 * persistent[t - 1] + innovations[t]

    naive = newey_west_variance(persistent, lags=0)
    corrected = newey_west_variance(persistent, lags=schwert_lags(2000))
    assert corrected > naive


def test_bartlett_weights_keep_the_variance_non_negative():
    rng = np.random.default_rng(5)
    alternating = np.array([(-1.0) ** i for i in range(400)]) + 0.01 * rng.standard_normal(400)
    assert newey_west_variance(alternating, lags=20) >= 0.0


def test_identical_losses_give_no_evidence_of_a_difference():
    rng = np.random.default_rng(19)
    losses = rng.gamma(2.0, 1.0, size=400)
    result = diebold_mariano(losses, losses)
    assert result.statistic == pytest.approx(0.0)
    assert result.p_value == pytest.approx(1.0)


def test_a_consistent_loss_advantage_is_detected():
    rng = np.random.default_rng(23)
    baseline = rng.gamma(2.0, 1.0, size=1000)
    worse = baseline + 0.30 + 0.05 * rng.standard_normal(1000)

    result = diebold_mariano(worse, baseline, model_a="worse", model_b="baseline")
    assert result.mean_differential > 0
    assert result.statistic > 0
    assert result.p_value < 0.01
    assert result.favoured == "baseline"
    assert "significantly more accurate" in result.verdict()


def test_pure_noise_is_not_reported_as_significant():
    rng = np.random.default_rng(29)
    a = rng.gamma(2.0, 1.0, size=1500)
    b = rng.gamma(2.0, 1.0, size=1500)
    result = diebold_mariano(a, b)
    assert result.p_value > 0.05
    assert result.verdict() == "no significant difference"


def test_statistic_is_antisymmetric_in_model_order():
    rng = np.random.default_rng(31)
    a = rng.gamma(2.0, 1.0, size=600)
    b = a + 0.1 + 0.02 * rng.standard_normal(600)

    forward = diebold_mariano(a, b)
    reverse = diebold_mariano(b, a)
    assert forward.statistic == pytest.approx(-reverse.statistic)
    assert forward.p_value == pytest.approx(reverse.p_value)


def test_harvey_correction_shrinks_the_statistic():
    rng = np.random.default_rng(37)
    a = rng.gamma(2.0, 1.0, size=120)
    b = a + 0.2 + 0.05 * rng.standard_normal(120)

    corrected = diebold_mariano(a, b, harvey_correction=True)
    uncorrected = diebold_mariano(a, b, harvey_correction=False)
    assert abs(corrected.statistic) < abs(uncorrected.statistic)


def test_non_finite_observations_are_dropped_pairwise():
    a = np.array([1.0, 2.0, np.nan, 4.0] * 30)
    b = np.array([1.5, 2.5, 3.5, np.nan] * 30)
    result = diebold_mariano(a, b)
    assert result.n_observations == 60


def test_too_few_observations_raises():
    with pytest.raises(ValueError):
        diebold_mariano(np.ones(5), np.zeros(5))


def test_schwert_rule_grows_with_sample_size():
    assert schwert_lags(100) < schwert_lags(2000)
    assert schwert_lags(1) == 0
