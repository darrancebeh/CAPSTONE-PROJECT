"""Regulatory backtesting of Value-at-Risk implied by the variance forecasts.

Statistical loss functions score a forecast against a proxy. They do not say
whether the forecast would have kept a book adequately capitalised, which is
the question a risk function actually faces. Mapping each variance forecast to
a VaR threshold and testing the resulting exception sequence answers that
question directly, and it separates two failures that a single loss number
conflates: producing the wrong number of exceptions, and producing the right
number but bunching them together in a crisis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import xlogy


@dataclass
class VarBacktestResult:
    """Coverage diagnostics for one model at one confidence level."""

    model: str
    confidence: float
    n_observations: int
    n_exceptions: int
    expected_exceptions: float
    exception_rate: float
    kupiec_statistic: float
    kupiec_p_value: float
    christoffersen_statistic: float
    christoffersen_p_value: float
    conditional_coverage_statistic: float
    conditional_coverage_p_value: float
    mean_var: float

    def passes(self, alpha: float = 0.05) -> Dict[str, bool]:
        """Whether each null of correct coverage survives at level ``alpha``."""
        return {
            "unconditional_coverage": bool(self.kupiec_p_value >= alpha),
            "independence": bool(self.christoffersen_p_value >= alpha),
            "conditional_coverage": bool(self.conditional_coverage_p_value >= alpha),
        }

    def to_dict(self, alpha: float = 0.05) -> Dict[str, object]:
        outcome = self.passes(alpha)
        return {
            "model": self.model,
            "confidence": self.confidence,
            "n_observations": self.n_observations,
            "n_exceptions": self.n_exceptions,
            "expected_exceptions": round(self.expected_exceptions, 2),
            "exception_rate": self.exception_rate,
            "mean_var": self.mean_var,
            "kupiec_lr": self.kupiec_statistic,
            "kupiec_p": self.kupiec_p_value,
            "christoffersen_lr": self.christoffersen_statistic,
            "christoffersen_p": self.christoffersen_p_value,
            "cc_lr": self.conditional_coverage_statistic,
            "cc_p": self.conditional_coverage_p_value,
            "pass_unconditional": outcome["unconditional_coverage"],
            "pass_independence": outcome["independence"],
            "pass_conditional": outcome["conditional_coverage"],
        }


def value_at_risk(variance_forecast, confidence: float) -> np.ndarray:
    r"""Parametric VaR, :math:`z_{1-c}\sqrt{\hat{\sigma}^2_t}`.

    Reported as a positive loss magnitude in the same units as the returns.
    The Gaussian quantile is applied to every model, including the network,
    so that differences in coverage come from the variance forecasts rather
    than from differing distributional assumptions bolted on afterwards.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between 0 and 1")
    z = stats.norm.ppf(confidence)
    return z * np.sqrt(np.asarray(variance_forecast, dtype=float))


def kupiec_pof(n_observations: int, n_exceptions: int, p: float) -> tuple[float, float]:
    """Unconditional coverage likelihood ratio, distributed chi-squared(1)."""
    if n_observations == 0:
        return np.nan, np.nan

    x = n_exceptions
    n = n_observations

    log_null = xlogy(n - x, 1.0 - p) + xlogy(x, p)
    pi_hat = x / n
    log_alt = xlogy(n - x, 1.0 - pi_hat) + xlogy(x, pi_hat)

    statistic = float(-2.0 * (log_null - log_alt))
    statistic = max(statistic, 0.0)
    return statistic, float(stats.chi2.sf(statistic, df=1))


def christoffersen_independence(exceptions: Sequence[int]) -> tuple[float, float]:
    """Markov independence likelihood ratio, distributed chi-squared(1).

    Tests whether an exception today changes the probability of an exception
    tomorrow. Clustered breaches are the signature of a model that adjusts to
    a volatility shock too slowly.
    """
    indicators = np.asarray(exceptions, dtype=int).ravel()
    if len(indicators) < 2:
        return np.nan, np.nan

    previous, current = indicators[:-1], indicators[1:]
    n00 = int(np.sum((previous == 0) & (current == 0)))
    n01 = int(np.sum((previous == 0) & (current == 1)))
    n10 = int(np.sum((previous == 1) & (current == 0)))
    n11 = int(np.sum((previous == 1) & (current == 1)))

    total = n00 + n01 + n10 + n11
    if total == 0 or (n01 + n11) == 0:
        # With no exceptions there is no dependence structure to reject.
        return 0.0, 1.0

    pi_01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
    pi_11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
    pi = (n01 + n11) / total

    log_null = xlogy(n00 + n10, 1.0 - pi) + xlogy(n01 + n11, pi)
    log_alt = (
        xlogy(n00, 1.0 - pi_01)
        + xlogy(n01, pi_01)
        + xlogy(n10, 1.0 - pi_11)
        + xlogy(n11, pi_11)
    )

    statistic = float(-2.0 * (log_null - log_alt))
    statistic = max(statistic, 0.0)
    return statistic, float(stats.chi2.sf(statistic, df=1))


def conditional_coverage(kupiec_statistic: float, independence_statistic: float):
    """Joint test of correct frequency and independence, chi-squared(2)."""
    if not np.isfinite(kupiec_statistic) or not np.isfinite(independence_statistic):
        return np.nan, np.nan
    statistic = float(kupiec_statistic + independence_statistic)
    return statistic, float(stats.chi2.sf(statistic, df=2))


def backtest_var(
    returns: pd.Series,
    variance_forecast: pd.Series,
    confidence: float,
    model: str = "model",
) -> VarBacktestResult:
    """Full coverage assessment of one variance forecast series."""
    frame = pd.concat(
        [returns.rename("return"), variance_forecast.rename("variance")], axis=1
    ).dropna()
    if frame.empty:
        raise ValueError(f"No overlapping observations for {model}")

    var_threshold = value_at_risk(frame["variance"], confidence)
    exceptions = (frame["return"].to_numpy() < -var_threshold).astype(int)

    n = len(frame)
    x = int(exceptions.sum())
    p = 1.0 - confidence

    kupiec_statistic, kupiec_p = kupiec_pof(n, x, p)
    independence_statistic, independence_p = christoffersen_independence(exceptions)
    cc_statistic, cc_p = conditional_coverage(kupiec_statistic, independence_statistic)

    return VarBacktestResult(
        model=model,
        confidence=confidence,
        n_observations=n,
        n_exceptions=x,
        expected_exceptions=n * p,
        exception_rate=x / n,
        kupiec_statistic=kupiec_statistic,
        kupiec_p_value=kupiec_p,
        christoffersen_statistic=independence_statistic,
        christoffersen_p_value=independence_p,
        conditional_coverage_statistic=cc_statistic,
        conditional_coverage_p_value=cc_p,
        mean_var=float(np.mean(var_threshold)),
    )


def exception_indicators(
    returns: pd.Series, variance_forecast: pd.Series, confidence: float
) -> pd.Series:
    """Dated 0/1 exception series, for plotting breach clustering."""
    frame = pd.concat(
        [returns.rename("return"), variance_forecast.rename("variance")], axis=1
    ).dropna()
    threshold = value_at_risk(frame["variance"], confidence)
    return pd.Series(
        (frame["return"].to_numpy() < -threshold).astype(int),
        index=frame.index,
        name="exception",
    )


def run_var_backtests(
    returns: pd.Series,
    forecasts: pd.DataFrame,
    confidence_levels: Sequence[float],
    labels: Optional[Dict[str, str]] = None,
) -> List[VarBacktestResult]:
    labels = labels or {}
    results: List[VarBacktestResult] = []
    for confidence in confidence_levels:
        for key in forecasts.columns:
            results.append(
                backtest_var(
                    returns=returns,
                    variance_forecast=forecasts[key],
                    confidence=confidence,
                    model=labels.get(key, key),
                )
            )
    return results


def var_results_to_frame(
    results: Sequence[VarBacktestResult], alpha: float = 0.05
) -> pd.DataFrame:
    return pd.DataFrame([r.to_dict(alpha) for r in results])
