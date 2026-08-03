"""Expanding-window walk-forward backtesting engine.

The engine owns the information boundary. For every out-of-sample date it
computes the last session that a forecaster standing at that date could have
observed, and passes that boundary to the model. Models receive the full
:class:`Dataset` but are contractually forbidden from reading past the
boundary; :meth:`WalkForwardBacktester.audit_no_lookahead` verifies that the
contract holds by re-running a sample of forecasts against a truncated dataset
and checking that the numbers are unchanged.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..config import Config
from ..data.features import Dataset
from ..models.base import ModelFitError, VolatilityModel, previous_session
from ..utils import get_logger
from .regimes import assign_regimes, regime_spans

logger = get_logger(__name__)


@dataclass
class ModelRunSummary:
    """Per-model bookkeeping for one walk-forward pass."""

    key: str
    label: str
    family: str
    n_forecasts: int
    n_refits: int
    n_failures: int
    fit_seconds: float
    predict_seconds: float

    @property
    def total_seconds(self) -> float:
        return self.fit_seconds + self.predict_seconds

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "family": self.family,
            "n_forecasts": self.n_forecasts,
            "n_refits": self.n_refits,
            "n_failures": self.n_failures,
            "fit_seconds": round(self.fit_seconds, 3),
            "predict_seconds": round(self.predict_seconds, 3),
            "total_seconds": round(self.total_seconds, 3),
        }


@dataclass
class BacktestResult:
    """Everything a downstream consumer needs to score the run."""

    forecasts: pd.DataFrame
    proxy: pd.Series
    returns: pd.Series
    regimes: pd.Series
    summaries: List[ModelRunSummary] = field(default_factory=list)
    fit_log: pd.DataFrame = field(default_factory=pd.DataFrame)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def model_keys(self) -> List[str]:
        return list(self.forecasts.columns)

    def labels(self) -> Dict[str, str]:
        return {s.key: s.label for s in self.summaries}


class WalkForwardBacktester:
    """Drives every model over a common out-of-sample window."""

    def __init__(self, config: Config, dataset: Dataset):
        self.config = config
        self.dataset = dataset
        self.session_index = dataset.returns.index
        self.evaluation_dates = self._resolve_evaluation_dates()

    def _resolve_evaluation_dates(self) -> pd.DatetimeIndex:
        split_date = pd.Timestamp(self.config.split.initial_train_end)
        candidates = self.dataset.common_index()
        dates = candidates[candidates > split_date]

        if len(dates) == 0:
            raise ValueError(
                f"No out-of-sample dates after {split_date.date()}; check the split configuration"
            )

        first_train = self.session_index[self.session_index <= split_date]
        minimum = self.config.split.min_train_observations
        if len(first_train) < minimum:
            raise ValueError(
                f"Initial training window holds {len(first_train)} observations, "
                f"below the configured minimum of {minimum}"
            )

        logger.info(
            "Out-of-sample window: %s to %s (%d sessions); initial training window %d sessions",
            dates[0].date(),
            dates[-1].date(),
            len(dates),
            len(first_train),
        )
        return dates

    def run(self, models: Sequence[VolatilityModel]) -> BacktestResult:
        forecasts: Dict[str, pd.Series] = {}
        summaries: List[ModelRunSummary] = []
        fit_records: List[Dict[str, Any]] = []

        for model in models:
            series, summary, records = self._run_single(model)
            forecasts[model.key] = series
            summaries.append(summary)
            fit_records.extend(records)

        panel = pd.DataFrame(forecasts, index=self.evaluation_dates)
        panel.index.name = "date"

        proxy = self.dataset.proxy.loc[self.evaluation_dates]
        returns = self.dataset.returns.loc[self.evaluation_dates]
        regimes = assign_regimes(self.evaluation_dates, self.config.regimes)

        metadata = self._build_metadata(summaries, regimes)
        fit_log = pd.DataFrame(fit_records)

        return BacktestResult(
            forecasts=panel,
            proxy=proxy,
            returns=returns,
            regimes=regimes,
            summaries=summaries,
            fit_log=fit_log,
            metadata=metadata,
        )

    def _run_single(self, model: VolatilityModel):
        cadence = max(int(self.config.backtest.cadence(model.key)), 1)
        values = np.full(len(self.evaluation_dates), np.nan, dtype=float)
        records: List[Dict[str, Any]] = []

        fit_seconds = 0.0
        predict_seconds = 0.0
        n_refits = 0
        n_failures = 0
        steps_since_refit = 0

        logger.info("Running %s (refit every %d session(s))", model.label, cadence)

        for position, target_date in enumerate(self.evaluation_dates):
            history_end = previous_session(self.session_index, target_date)
            if history_end >= target_date:
                raise RuntimeError(
                    f"Information boundary violated: history_end {history_end} >= target {target_date}"
                )

            needs_refit = not model.is_fitted or steps_since_refit == 0
            if needs_refit:
                start = time.perf_counter()
                try:
                    model.fit(self.dataset, train_end=history_end)
                    n_refits += 1
                    records.append(
                        {
                            "model": model.key,
                            "refit_date": history_end,
                            "n_train_observations": int(
                                len(self.session_index[self.session_index <= history_end])
                            ),
                            **model.parameters_snapshot(),
                        }
                    )
                except ModelFitError as exc:
                    logger.warning("%s: fit failed at %s (%s)", model.label, history_end.date(), exc)
                    n_failures += 1
                finally:
                    fit_seconds += time.perf_counter() - start

            steps_since_refit = (steps_since_refit + 1) % cadence

            if not model.is_fitted:
                continue

            start = time.perf_counter()
            try:
                values[position] = model.predict(self.dataset, target_date)
            except ModelFitError as exc:
                logger.warning(
                    "%s: forecast failed for %s (%s)", model.label, target_date.date(), exc
                )
                n_failures += 1
            finally:
                predict_seconds += time.perf_counter() - start

            if (position + 1) % 250 == 0:
                logger.info(
                    "  %s: %d/%d sessions", model.label, position + 1, len(self.evaluation_dates)
                )

        series = pd.Series(values, index=self.evaluation_dates, name=model.key)
        n_valid = int(series.notna().sum())
        if n_valid == 0:
            raise RuntimeError(f"{model.label} produced no usable forecasts")

        summary = ModelRunSummary(
            key=model.key,
            label=model.label,
            family=model.family,
            n_forecasts=n_valid,
            n_refits=n_refits,
            n_failures=n_failures,
            fit_seconds=fit_seconds,
            predict_seconds=predict_seconds,
        )
        logger.info(
            "%s complete: %d forecasts, %d refits, %.1fs total",
            model.label,
            n_valid,
            n_refits,
            summary.total_seconds,
        )
        return series, summary, records

    def _build_metadata(
        self, summaries: Sequence[ModelRunSummary], regimes: pd.Series
    ) -> Dict[str, Any]:
        return {
            "config_name": self.config.name,
            "random_seed": self.config.random_seed,
            "symbol": self.config.data.symbol,
            "sample_start": self.session_index[0],
            "sample_end": self.session_index[-1],
            "initial_train_end": self.config.split.initial_train_end,
            "evaluation_start": self.evaluation_dates[0],
            "evaluation_end": self.evaluation_dates[-1],
            "n_evaluation_days": len(self.evaluation_dates),
            "sequence_length": self.config.features.sequence_length,
            "n_features": self.dataset.features.shape[1],
            "feature_names": self.dataset.feature_names,
            "refit_cadence": self.config.backtest.refit_every,
            "regime_spans": regime_spans(regimes),
            "models": [s.to_dict() for s in summaries],
        }

    def audit_no_lookahead(
        self,
        model: VolatilityModel,
        sample_dates: Optional[Sequence[pd.Timestamp]] = None,
        n_checkpoints: int = 5,
    ) -> pd.DataFrame:
        """Re-forecast a handful of dates from a truncated dataset.

        A model that peeks at future observations will produce a different
        number once everything after the information boundary is physically
        removed from the inputs. The model is refitted on the truncated data,
        so this checks the whole estimate-then-forecast path rather than the
        forecast step alone.
        """
        if sample_dates is None:
            positions = np.unique(
                np.linspace(0, len(self.evaluation_dates) - 1, num=n_checkpoints, dtype=int)
            )
            sample_dates = [self.evaluation_dates[p] for p in positions]

        rows = []
        for target_date in sample_dates:
            history_end = previous_session(self.session_index, target_date)

            model.fit(self.dataset, train_end=history_end)
            full = model.predict(self.dataset, target_date)

            truncated = Dataset(
                prices=self.dataset.prices.loc[:history_end],
                returns=self.dataset.returns.loc[:history_end],
                proxy=self.dataset.proxy.loc[:history_end],
                features=self.dataset.features.loc[:history_end],
            )
            model.fit(truncated, train_end=history_end)
            restricted = model.predict(truncated, target_date)

            rows.append(
                {
                    "model": model.key,
                    "target_date": target_date,
                    "forecast_full_dataset": full,
                    "forecast_truncated_dataset": restricted,
                    "absolute_difference": abs(full - restricted),
                }
            )

        return pd.DataFrame(rows)
