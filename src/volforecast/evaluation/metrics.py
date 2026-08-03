"""Forecast loss functions and summary accuracy statistics.

QLIKE is the primary criterion. Squared and absolute error are reported
alongside it for comparability with the applied literature, not because they
are appropriate: both rank forecasts inconsistently when the target is a noisy
proxy for the latent variance, so a reversal between the QLIKE ranking and the
MSE ranking is an expected finding rather than a contradiction.
"""

from __future__ import annotations

from typing import Callable, Dict

import numpy as np
import pandas as pd

EPS = 1e-12


def _as_array(values) -> np.ndarray:
    return np.asarray(values, dtype=float).ravel()


def align(proxy, forecast) -> tuple[np.ndarray, np.ndarray]:
    """Drop any observation where either series is missing."""
    y = _as_array(proxy)
    f = _as_array(forecast)
    if y.shape != f.shape:
        raise ValueError(f"Shape mismatch: proxy {y.shape} vs forecast {f.shape}")
    mask = np.isfinite(y) & np.isfinite(f)
    return y[mask], f[mask]


def qlike_loss(proxy, forecast) -> np.ndarray:
    r"""Element-wise QLIKE, :math:`y/\hat{y} - \ln(y/\hat{y}) - 1`.

    Non-negative, zero only at a perfect forecast, and robust to noise in
    ``proxy`` in the sense of Patton: the ranking it induces over forecasts is
    the same whether it is evaluated against the latent variance or against an
    unbiased proxy for it.
    """
    y, f = align(proxy, forecast)
    y = np.maximum(y, EPS)
    f = np.maximum(f, EPS)
    ratio = y / f
    return ratio - np.log(ratio) - 1.0


def squared_error(proxy, forecast) -> np.ndarray:
    y, f = align(proxy, forecast)
    return (y - f) ** 2


def absolute_error(proxy, forecast) -> np.ndarray:
    y, f = align(proxy, forecast)
    return np.abs(y - f)


LOSS_FUNCTIONS: Dict[str, Callable] = {
    "qlike": qlike_loss,
    "mse": squared_error,
    "mae": absolute_error,
}


def loss_series(proxy: pd.Series, forecast: pd.Series, loss: str) -> pd.Series:
    """Element-wise loss retaining the date index of the common sample."""
    if loss not in LOSS_FUNCTIONS:
        raise KeyError(f"Unknown loss '{loss}'; available: {sorted(LOSS_FUNCTIONS)}")
    frame = pd.concat([proxy.rename("proxy"), forecast.rename("forecast")], axis=1).dropna()
    values = LOSS_FUNCTIONS[loss](frame["proxy"], frame["forecast"])
    return pd.Series(values, index=frame.index, name=loss)


def mincer_zarnowitz(proxy, forecast) -> Dict[str, float]:
    r"""Regress the proxy on the forecast: :math:`y_t = a + b\hat{y}_t + u_t`.

    An unbiased, efficient forecast implies :math:`a = 0` and :math:`b = 1`.
    The slope is the more informative of the two here: a value below one is the
    signature of a forecaster that over-reacts, scaling its predictions more
    aggressively than the realised proxy warrants.
    """
    y, f = align(proxy, forecast)
    # Two coefficients plus one residual degree of freedom.
    if len(y) < 3:
        return {"mz_intercept": np.nan, "mz_slope": np.nan, "mz_r2": np.nan}

    design = np.column_stack([np.ones_like(f), f])
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ coefficients
    residuals = y - fitted

    total_ss = float(((y - y.mean()) ** 2).sum())
    residual_ss = float((residuals**2).sum())
    r2 = 1.0 - residual_ss / total_ss if total_ss > 0 else np.nan

    return {
        "mz_intercept": float(coefficients[0]),
        "mz_slope": float(coefficients[1]),
        "mz_r2": float(r2),
    }


def out_of_sample_r2(proxy, forecast) -> float:
    r"""Predictive :math:`R^2`, :math:`1 - \sum(y - \hat{y})^2 / \sum(y - \bar{y})^2`.

    Distinct from the Mincer-Zarnowitz regression :math:`R^2`, which fits a free
    intercept and slope and is therefore bounded below by zero. This version
    scores the forecast as issued, so a model that is worse than the sample mean
    of the proxy returns a negative value. It is the definition most commonly
    reported in the applied comparative literature, and it is included so that
    results here can be placed alongside that work without redefining terms.

    It inherits every weakness of squared error under a noisy proxy and is
    reported for comparability only, never as a ranking criterion.
    """
    y, f = align(proxy, forecast)
    if len(y) < 2:
        return np.nan
    total_ss = float(((y - y.mean()) ** 2).sum())
    if total_ss <= 0:
        return np.nan
    return float(1.0 - ((y - f) ** 2).sum() / total_ss)


def summarise_forecast(proxy, forecast) -> Dict[str, float]:
    """Full accuracy profile for one forecast series."""
    y, f = align(proxy, forecast)
    if len(y) == 0:
        return {}

    qlike = qlike_loss(y, f)
    squared = squared_error(y, f)
    absolute = absolute_error(y, f)

    summary = {
        "n_observations": float(len(y)),
        "qlike": float(qlike.mean()),
        "mse": float(squared.mean()),
        "rmse": float(np.sqrt(squared.mean())),
        "mae": float(absolute.mean()),
        "mean_forecast": float(f.mean()),
        "mean_proxy": float(y.mean()),
        # Mean bias on the volatility scale, which is easier to interpret than
        # a bias in squared percentage points.
        "bias_volatility": float(np.sqrt(f).mean() - np.sqrt(y).mean()),
        "forecast_std": float(f.std(ddof=1)) if len(f) > 1 else np.nan,
        # Upper reach of the forecast distribution. A model whose maximum sits
        # far below the largest realised proxy cannot respond to a crisis at
        # all, however well it performs on average, so this column distinguishes
        # a merely inaccurate forecaster from a structurally capped one.
        "max_forecast": float(f.max()),
        "p99_forecast": float(np.quantile(f, 0.99)),
        "max_proxy": float(y.max()),
        "tail_reach": float(f.max() / y.max()) if y.max() > 0 else np.nan,
    }
    summary.update(mincer_zarnowitz(y, f))
    summary["r2_oos"] = out_of_sample_r2(y, f)
    return summary
