"""Common interface shared by the econometric and neural forecasters.

The walk-forward engine drives every model through the same two-method
contract. Keeping the loop outside the models is what makes the comparison
structurally fair: no model can widen its own information set, because the
engine is the only component that decides which dates a model may see.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional

import pandas as pd

from ..data.features import Dataset


class ModelFitError(Exception):
    """Raised when a model cannot be estimated on the supplied window."""


class VolatilityModel(ABC):
    """A one-day-ahead conditional variance forecaster.

    Implementations must honour a single invariant: neither :meth:`fit` nor
    :meth:`predict` may read any observation dated after the ``train_end`` or
    ``target_date`` boundary they are given.
    """

    key: str = "base"
    label: str = "Base"
    family: str = "unspecified"

    def __init__(self) -> None:
        self._fitted = False
        self._train_end: Optional[pd.Timestamp] = None

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def train_end(self) -> Optional[pd.Timestamp]:
        """Last session included in the most recent estimation window."""
        return self._train_end

    @abstractmethod
    def fit(self, dataset: Dataset, train_end: pd.Timestamp) -> None:
        """Estimate parameters on observations up to and including ``train_end``."""

    @abstractmethod
    def predict(self, dataset: Dataset, target_date: pd.Timestamp) -> float:
        """Return the variance forecast for ``target_date``.

        The forecast is formed at the close of the session preceding
        ``target_date`` and must not use any quantity observed on
        ``target_date`` itself.
        """

    def parameters_snapshot(self) -> Dict[str, float]:
        """Estimated parameters at the current fit, for the audit trail."""
        return {}

    def __repr__(self) -> str:
        state = "fitted" if self._fitted else "unfitted"
        return f"{type(self).__name__}(key={self.key!r}, {state})"


def previous_session(index: pd.DatetimeIndex, date: pd.Timestamp) -> pd.Timestamp:
    """Last date in ``index`` that falls strictly before ``date``."""
    position = index.searchsorted(date, side="left")
    if position == 0:
        raise ModelFitError(f"No observation precedes {date.date()}")
    return index[position - 1]
