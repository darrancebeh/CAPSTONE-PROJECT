"""Segmentation of the out-of-sample period into market regimes.

Regime boundaries are fixed in configuration from documented macroeconomic
events rather than inferred from the realised volatility of the evaluation
sample. An endogenous rule, such as splitting on a volatility quantile, would
label each day using information that a forecaster standing at that date could
not have had, and would flatter whichever model reacts fastest after the fact.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import pandas as pd

from ..config import Regime

UNCLASSIFIED = "Unclassified"


def assign_regimes(index: pd.DatetimeIndex, regimes: Sequence[Regime]) -> pd.Series:
    """Label each date with the regime whose window contains it."""
    labels = pd.Series(UNCLASSIFIED, index=index, dtype="object", name="regime")
    for regime in regimes:
        start = pd.Timestamp(regime.start)
        end = pd.Timestamp(regime.end)
        if start > end:
            raise ValueError(f"Regime '{regime.name}' has start after end")
        mask = (index >= start) & (index <= end)
        overlap = labels.loc[mask] != UNCLASSIFIED
        if overlap.any():
            clashing = sorted(set(labels.loc[mask][overlap]))
            raise ValueError(f"Regime '{regime.name}' overlaps with {clashing}")
        labels.loc[mask] = regime.name
    return labels


def regime_order(regimes: Sequence[Regime]) -> List[str]:
    """Chronological regime names, with the catch-all bucket appended."""
    return [regime.name for regime in regimes] + [UNCLASSIFIED]


def regime_spans(labels: pd.Series) -> Dict[str, Dict[str, object]]:
    """Observation count and date range for every regime present."""
    spans: Dict[str, Dict[str, object]] = {}
    for name, group in labels.groupby(labels, sort=False):
        spans[str(name)] = {
            "n_observations": int(len(group)),
            "start": group.index[0].date().isoformat(),
            "end": group.index[-1].date().isoformat(),
        }
    return spans
