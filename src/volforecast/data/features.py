"""Return construction, volatility proxies and the predictor matrix.

Every column produced here is measurable at the close of the session it is
indexed by. Nothing in this module shifts information backwards in time; the
one-day-ahead alignment between predictors and target is applied downstream by
:func:`build_sequences`, which is the single place where the forecast timing
convention is enforced.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd

from ..config import DataConfig, FeatureConfig
from ..utils import get_logger

logger = get_logger(__name__)

# Scaling constants for the intraday range estimators.
_PARKINSON = 1.0 / (4.0 * np.log(2.0))
_GK_SECOND_TERM = 2.0 * np.log(2.0) - 1.0


@dataclass
class Dataset:
    """Aligned view of everything the models consume.

    Attributes
    ----------
    prices:
        Cleaned OHLCV frame indexed by session date.
    returns:
        Log returns in percentage points, one observation per session after the
        first. The GARCH family is estimated directly on this series.
    proxy:
        Realised variance proxy (squared returns) in squared percentage points.
        This is the forecast target and the evaluation benchmark.
    features:
        Predictor matrix for the sequence model. Begins later than ``returns``
        because the rolling statistics require a warm-up window.
    """

    prices: pd.DataFrame
    returns: pd.Series
    proxy: pd.Series
    features: pd.DataFrame

    @property
    def feature_names(self) -> List[str]:
        return list(self.features.columns)

    def common_index(self) -> pd.DatetimeIndex:
        """Dates for which a return, a proxy and a feature row all exist."""
        return self.features.index.intersection(self.proxy.index)


def compute_log_returns(prices: pd.DataFrame, price_column: str, scale: float) -> pd.Series:
    close = prices[price_column]
    returns = np.log(close).diff() * scale
    returns.name = "return"
    return returns.dropna()


def _parkinson_variance(prices: pd.DataFrame, scale: float) -> pd.Series:
    log_range = np.log(prices["high"] / prices["low"]) * scale
    return _PARKINSON * log_range.pow(2)


def _garman_klass_variance(prices: pd.DataFrame, scale: float) -> pd.Series:
    log_hl = np.log(prices["high"] / prices["low"]) * scale
    log_co = np.log(prices["close"] / prices["open"]) * scale
    variance = 0.5 * log_hl.pow(2) - _GK_SECOND_TERM * log_co.pow(2)
    # The estimator is unbiased but not bounded below; clip at a small positive
    # value so that logarithms downstream remain defined.
    return variance.clip(lower=1e-8)


def ewma_variance(proxy: pd.Series, lam: float) -> pd.Series:
    """RiskMetrics filter s2_t = lam * s2_{t-1} + (1 - lam) * r_t^2.

    The value at date ``t`` conditions on information through ``t`` inclusive,
    which makes it a valid predictor for ``t + 1``.
    """
    filtered = proxy.ewm(alpha=1.0 - lam, adjust=False).mean()
    filtered.name = "ewma_variance"
    return filtered


class FeatureEngineer:
    """Builds the predictor matrix consumed by the sequence model."""

    def __init__(self, data_config: DataConfig, feature_config: FeatureConfig):
        self.data_config = data_config
        self.feature_config = feature_config

    def build(self, prices: pd.DataFrame) -> Dataset:
        scale = self.data_config.return_scale
        floor = self.data_config.min_variance_proxy
        returns = compute_log_returns(prices, self.data_config.price_column, scale)

        raw_proxy = returns.pow(2)
        n_floored = int((raw_proxy < floor).sum())
        if n_floored:
            logger.info(
                "Applied the %.0e proxy floor to %d of %d sessions (%.2f%%)",
                floor,
                n_floored,
                len(raw_proxy),
                100.0 * n_floored / len(raw_proxy),
            )
        proxy = raw_proxy.clip(lower=floor).rename("proxy")

        aligned_prices = prices.loc[returns.index]
        columns = {}

        columns["return"] = returns
        columns["abs_return"] = returns.abs()
        columns["proxy"] = proxy
        columns["log_proxy"] = np.log(proxy)
        # Signed shock magnitude, active only on down days: the feature through
        # which the network can learn an asymmetric response.
        columns["negative_shock"] = proxy.where(returns < 0.0, 0.0)

        for window in self.feature_config.realised_vol_windows:
            realised = proxy.rolling(window).mean().pow(0.5)
            columns[f"realised_vol_{window}"] = realised

        short, long = self._vol_ratio_windows()
        columns["vol_ratio"] = (
            columns[f"realised_vol_{short}"] / columns[f"realised_vol_{long}"]
        )

        columns["ewma_variance"] = ewma_variance(proxy, self.feature_config.ewma_lambda)
        columns["parkinson_variance"] = _parkinson_variance(aligned_prices, scale)
        columns["garman_klass_variance"] = _garman_klass_variance(aligned_prices, scale)

        overnight = np.log(
            aligned_prices["open"] / aligned_prices[self.data_config.price_column].shift(1)
        ) * scale
        columns["overnight_gap"] = overnight

        volume = aligned_prices["volume"].replace(0.0, np.nan)
        volume_ma = volume.rolling(self.feature_config.volume_ma_window).mean()
        columns["volume_ratio"] = np.log(volume / volume_ma)

        features = pd.DataFrame(columns).replace([np.inf, -np.inf], np.nan)
        n_before = len(features)
        features = features.dropna()
        logger.info(
            "Feature matrix: %d columns, %d rows retained after a %d-row warm-up",
            features.shape[1],
            features.shape[0],
            n_before - features.shape[0],
        )

        return Dataset(
            prices=prices,
            returns=returns,
            proxy=proxy,
            features=features,
        )

    def _vol_ratio_windows(self) -> Tuple[int, int]:
        windows = sorted(self.feature_config.realised_vol_windows)
        return windows[0], windows[-1]


def build_sequences(
    features: pd.DataFrame,
    target: pd.Series,
    sequence_length: int,
    target_dates: pd.DatetimeIndex,
) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Assemble supervised sequences under a strict one-day-ahead convention.

    For a target date ``d`` the input window is the ``sequence_length`` feature
    rows that end on the session immediately preceding ``d``. The forecast for
    ``d`` is therefore formed at the close of ``d - 1`` and never touches a
    quantity observed on ``d`` itself.

    Returns the input tensor of shape ``(n, sequence_length, n_features)``, the
    target vector, and the subset of ``target_dates`` that could be served.
    """
    if sequence_length < 1:
        raise ValueError("sequence_length must be at least 1")

    feature_values = features.to_numpy(dtype=np.float32)
    feature_index = features.index

    windows: List[np.ndarray] = []
    targets: List[float] = []
    served: List[pd.Timestamp] = []

    for date in target_dates:
        if date not in target.index:
            continue
        # Index of the last feature row strictly before the target date.
        end = feature_index.searchsorted(date, side="left") - 1
        if end < sequence_length - 1:
            continue
        start = end - sequence_length + 1
        windows.append(feature_values[start : end + 1])
        targets.append(float(target.loc[date]))
        served.append(date)

    if not windows:
        return (
            np.empty((0, sequence_length, feature_values.shape[1]), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            pd.DatetimeIndex([]),
        )

    return (
        np.stack(windows).astype(np.float32),
        np.asarray(targets, dtype=np.float32),
        pd.DatetimeIndex(served),
    )
