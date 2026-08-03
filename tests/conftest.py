import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from volforecast.config import DataConfig, FeatureConfig  # noqa: E402
from volforecast.data.calendar import trading_sessions  # noqa: E402


@pytest.fixture
def data_config(tmp_path) -> DataConfig:
    return DataConfig(
        raw_path=tmp_path / "prices.parquet",
        processed_dir=tmp_path / "processed",
        symbol="TEST",
        price_column="close",
        start_date="2015-01-02",
        end_date="2019-12-31",
        return_scale=100.0,
        min_variance_proxy=1e-4,
    )


@pytest.fixture
def feature_config() -> FeatureConfig:
    return FeatureConfig(
        sequence_length=5,
        realised_vol_windows=[5, 21],
        ewma_lambda=0.94,
        volume_ma_window=21,
    )


@pytest.fixture
def synthetic_prices() -> pd.DataFrame:
    """A GARCH-like price series on a real NYSE calendar."""
    sessions = trading_sessions("2015-01-02", "2019-12-31")
    rng = np.random.default_rng(20240101)

    n = len(sessions)
    variance = np.empty(n)
    shocks = np.empty(n)
    variance[0] = 1.0
    for t in range(n):
        if t > 0:
            variance[t] = 0.05 + 0.10 * shocks[t - 1] ** 2 + 0.85 * variance[t - 1]
        shocks[t] = np.sqrt(variance[t]) * rng.standard_normal()

    close = 100.0 * np.exp(np.cumsum(shocks / 100.0))
    intraday = np.abs(rng.standard_normal(n)) * close * 0.004
    open_ = close * (1.0 + rng.standard_normal(n) * 0.001)

    frame = pd.DataFrame(
        {
            "date": sessions,
            "open": open_,
            "high": np.maximum(close, open_) + intraday,
            "low": np.minimum(close, open_) - intraday,
            "close": close,
            "volume": rng.uniform(5e7, 1.5e8, n),
        }
    )
    return frame
