"""Diebold-Mariano test of equal predictive accuracy.

The test asks whether the mean loss differential between two forecasts is
distinguishable from zero. Because forecast errors on financial series are
serially correlated and heteroskedastic, the sample mean is scaled by a
Newey-West long-run variance rather than the naive standard error, and the
Harvey-Leybourne-Newbold finite-sample correction is applied before referring
the statistic to a t-distribution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .metrics import loss_series


def newey_west_variance(differential: np.ndarray, lags: int) -> float:
    """HAC estimate of the long-run variance of the loss differential.

    Bartlett weights guarantee a non-negative estimate. The result is the
    long-run variance of a single observation, so the variance of the sample
    mean is this quantity divided by the sample size.
    """
    n = len(differential)
    if n == 0:
        return np.nan

    centred = differential - differential.mean()
    variance = float(np.dot(centred, centred) / n)

    for lag in range(1, lags + 1):
        if lag >= n:
            break
        weight = 1.0 - lag / (lags + 1.0)
        autocovariance = float(np.dot(centred[lag:], centred[:-lag]) / n)
        variance += 2.0 * weight * autocovariance

    return variance


def schwert_lags(n: int) -> int:
    """Automatic truncation lag, floor(4 (n/100)^(2/9))."""
    if n <= 1:
        return 0
    return int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))


@dataclass
class DieboldMarianoResult:
    """Outcome of one pairwise comparison."""

    model_a: str
    model_b: str
    loss: str
    n_observations: int
    mean_loss_a: float
    mean_loss_b: float
    mean_differential: float
    statistic: float
    p_value: float
    lags: int
    horizon: int
    harvey_corrected: bool

    @property
    def favoured(self) -> str:
        """Model with the lower average loss, irrespective of significance."""
        return self.model_a if self.mean_differential < 0 else self.model_b

    def verdict(self, alpha: float = 0.05) -> str:
        if not np.isfinite(self.p_value):
            return "undetermined"
        if self.p_value >= alpha:
            return "no significant difference"
        return f"{self.favoured} significantly more accurate"

    def to_dict(self) -> Dict[str, object]:
        return {
            "model_a": self.model_a,
            "model_b": self.model_b,
            "loss": self.loss,
            "n_observations": self.n_observations,
            "mean_loss_a": self.mean_loss_a,
            "mean_loss_b": self.mean_loss_b,
            "mean_differential": self.mean_differential,
            "dm_statistic": self.statistic,
            "p_value": self.p_value,
            "lags": self.lags,
            "favoured": self.favoured,
        }


def diebold_mariano(
    loss_a,
    loss_b,
    horizon: int = 1,
    lags: Optional[int] = None,
    harvey_correction: bool = True,
    model_a: str = "A",
    model_b: str = "B",
    loss_name: str = "loss",
) -> DieboldMarianoResult:
    """Two-sided test of :math:`H_0: E[L_A - L_B] = 0`.

    A negative statistic favours ``model_a``. Passing the two loss series
    rather than the forecasts keeps the test agnostic about which criterion is
    being compared.
    """
    a = np.asarray(loss_a, dtype=float).ravel()
    b = np.asarray(loss_b, dtype=float).ravel()
    if a.shape != b.shape:
        raise ValueError(f"Loss series lengths differ: {a.shape} vs {b.shape}")

    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    n = len(a)
    if n < 10:
        raise ValueError(f"Diebold-Mariano needs at least 10 paired observations, received {n}")

    differential = a - b
    mean_differential = float(differential.mean())

    if lags is None:
        # The theoretical truncation for an h-step forecast is h - 1, which is
        # zero at the one-day horizon. Misspecified variance models leave
        # autocorrelated losses regardless, so the data-driven rule is used as
        # a floor to avoid understating the standard error.
        lags = max(horizon - 1, schwert_lags(n))

    long_run_variance = newey_west_variance(differential, lags)
    if not np.isfinite(long_run_variance) or long_run_variance <= 0:
        # Two forecasts that produce an identical loss path are trivially
        # indistinguishable; anything else with zero dispersion is degenerate.
        statistic, p_value = (0.0, 1.0) if mean_differential == 0.0 else (np.nan, np.nan)
    else:
        statistic = mean_differential / np.sqrt(long_run_variance / n)

        if harvey_correction:
            # Harvey, Leybourne and Newbold (1997): the raw statistic is
            # oversized in finite samples, particularly at short horizons.
            factor = (n + 1.0 - 2.0 * horizon + horizon * (horizon - 1.0) / n) / n
            statistic *= np.sqrt(max(factor, 1e-12))
            p_value = 2.0 * (1.0 - stats.t.cdf(abs(statistic), df=n - 1))
        else:
            p_value = 2.0 * (1.0 - stats.norm.cdf(abs(statistic)))

    return DieboldMarianoResult(
        model_a=model_a,
        model_b=model_b,
        loss=loss_name,
        n_observations=n,
        mean_loss_a=float(a.mean()),
        mean_loss_b=float(b.mean()),
        mean_differential=mean_differential,
        statistic=float(statistic),
        p_value=float(p_value),
        lags=int(lags),
        horizon=horizon,
        harvey_corrected=harvey_correction,
    )


def pairwise_tests(
    proxy: pd.Series,
    forecasts: pd.DataFrame,
    loss: str = "qlike",
    horizon: int = 1,
    lags: Optional[int] = None,
    harvey_correction: bool = True,
    labels: Optional[Dict[str, str]] = None,
) -> List[DieboldMarianoResult]:
    """Run the test on every unordered pair of columns in ``forecasts``."""
    labels = labels or {}
    keys = list(forecasts.columns)
    losses = {key: loss_series(proxy, forecasts[key], loss) for key in keys}

    results: List[DieboldMarianoResult] = []
    for i, key_a in enumerate(keys):
        for key_b in keys[i + 1 :]:
            paired = pd.concat(
                [losses[key_a].rename("a"), losses[key_b].rename("b")], axis=1
            ).dropna()
            if len(paired) < 10:
                continue
            results.append(
                diebold_mariano(
                    paired["a"].to_numpy(),
                    paired["b"].to_numpy(),
                    horizon=horizon,
                    lags=lags,
                    harvey_correction=harvey_correction,
                    model_a=labels.get(key_a, key_a),
                    model_b=labels.get(key_b, key_b),
                    loss_name=loss,
                )
            )
    return results


def results_to_frame(results: Sequence[DieboldMarianoResult]) -> pd.DataFrame:
    return pd.DataFrame([r.to_dict() for r in results])


def statistic_matrix(
    results: Sequence[DieboldMarianoResult],
    order: Optional[Sequence[str]] = None,
    value: str = "statistic",
) -> pd.DataFrame:
    """Square matrix of test output, antisymmetric in the statistic.

    Entry ``(i, j)`` reports the comparison of row ``i`` against column ``j``,
    so a negative statistic means the row model has the lower average loss.
    """
    names: List[str] = list(order) if order is not None else []
    if not names:
        for result in results:
            for name in (result.model_a, result.model_b):
                if name not in names:
                    names.append(name)

    matrix = pd.DataFrame(np.nan, index=names, columns=names, dtype=float)
    for result in results:
        if result.model_a not in matrix.index or result.model_b not in matrix.columns:
            continue
        if value == "statistic":
            matrix.loc[result.model_a, result.model_b] = result.statistic
            matrix.loc[result.model_b, result.model_a] = -result.statistic
        elif value == "p_value":
            matrix.loc[result.model_a, result.model_b] = result.p_value
            matrix.loc[result.model_b, result.model_a] = result.p_value
        else:
            raise ValueError(f"Unsupported matrix value: {value}")
    return matrix
