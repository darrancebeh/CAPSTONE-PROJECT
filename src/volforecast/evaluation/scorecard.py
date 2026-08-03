"""Assembly of the full evaluation scorecard from a completed backtest."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..backtest.engine import BacktestResult
from ..backtest.regimes import UNCLASSIFIED
from ..config import Config
from ..utils import get_logger
from .diebold_mariano import pairwise_tests, results_to_frame, statistic_matrix
from .metrics import LOSS_FUNCTIONS, loss_series, summarise_forecast
from .var_backtest import run_var_backtests, var_results_to_frame

logger = get_logger(__name__)


class Evaluator:
    """Turns a forecast panel into the tables reported in the results chapter."""

    def __init__(self, config: Config, result: BacktestResult):
        self.config = config
        self.result = result
        self.labels = result.labels() or {k: k for k in result.forecasts.columns}
        self.model_order = [self.labels[k] for k in result.forecasts.columns]

    # ------------------------------------------------------------------
    # Accuracy tables
    # ------------------------------------------------------------------
    def overall_metrics(self) -> pd.DataFrame:
        rows = []
        for key in self.result.forecasts.columns:
            summary = summarise_forecast(self.result.proxy, self.result.forecasts[key])
            summary["model"] = self.labels[key]
            summary["key"] = key
            rows.append(summary)

        frame = pd.DataFrame(rows).set_index("model")
        frame = frame.loc[self.model_order]
        frame["qlike_rank"] = frame["qlike"].rank(method="min").astype(int)
        frame["mse_rank"] = frame["mse"].rank(method="min").astype(int)
        return frame

    def rank_comparison(self) -> pd.DataFrame:
        """Model ranking under each criterion, side by side.

        Included because the comparative literature routinely ranks volatility
        models on RMSE, MAE or predictive R-squared and reports whichever
        ordering results as *the* ordering. Where those criteria disagree with
        each other on the same forecasts, that disagreement is evidence about
        the criteria rather than about the models, and it is the reason this
        project treats QLIKE as the criterion of record.
        """
        overall = self.overall_metrics()
        ranks = pd.DataFrame(index=overall.index)

        # Lower is better for the loss columns, higher is better for fit.
        for column in ("qlike", "mse", "rmse", "mae"):
            ranks[column.upper()] = overall[column].rank(method="min").astype(int)
        for column in ("r2_oos", "mz_r2"):
            ranks[column] = overall[column].rank(method="min", ascending=False).astype(int)

        ranks = ranks.rename(columns={"r2_oos": "R2 (out-of-sample)", "mz_r2": "R2 (MZ)"})
        ranks["Rank spread"] = ranks.max(axis=1) - ranks.min(axis=1)
        return ranks

    def regime_metrics(self, loss: str = "qlike") -> pd.DataFrame:
        """Average loss by model and regime, plus the full-sample column."""
        if loss not in LOSS_FUNCTIONS:
            raise KeyError(f"Unknown loss '{loss}'")

        regimes = self.result.regimes
        columns: Dict[str, pd.Series] = {}

        for name in self._regime_names():
            mask = regimes == name
            if mask.sum() == 0:
                continue
            values = {}
            for key in self.result.forecasts.columns:
                series = loss_series(
                    self.result.proxy.loc[mask], self.result.forecasts[key].loc[mask], loss
                )
                values[self.labels[key]] = float(series.mean()) if len(series) else np.nan
            columns[name] = pd.Series(values)

        full = {}
        for key in self.result.forecasts.columns:
            series = loss_series(self.result.proxy, self.result.forecasts[key], loss)
            full[self.labels[key]] = float(series.mean())
        columns["Full sample"] = pd.Series(full)

        frame = pd.DataFrame(columns)
        return frame.loc[self.model_order]

    def regime_detail(self) -> pd.DataFrame:
        """Every accuracy statistic, for every model, within every regime."""
        rows = []
        regimes = self.result.regimes
        for name in self._regime_names() + ["Full sample"]:
            mask = pd.Series(True, index=regimes.index) if name == "Full sample" else regimes == name
            if mask.sum() == 0:
                continue
            for key in self.result.forecasts.columns:
                summary = summarise_forecast(
                    self.result.proxy.loc[mask], self.result.forecasts[key].loc[mask]
                )
                summary["regime"] = name
                summary["model"] = self.labels[key]
                rows.append(summary)
        frame = pd.DataFrame(rows)
        front = ["regime", "model", "n_observations", "qlike", "mse", "rmse", "mae"]
        ordered = front + [c for c in frame.columns if c not in front]
        return frame[ordered]

    def regime_observation_counts(self) -> pd.DataFrame:
        realised = np.sqrt(self.result.proxy)
        rows = []
        for name in self._regime_names():
            mask = self.result.regimes == name
            if mask.sum() == 0:
                continue
            rows.append(
                {
                    "regime": name,
                    "n_days": int(mask.sum()),
                    "start": self.result.regimes.index[mask][0].date().isoformat(),
                    "end": self.result.regimes.index[mask][-1].date().isoformat(),
                    "annualised_vol_pct": float(
                        np.sqrt(self.result.proxy.loc[mask].mean() * 252.0)
                    ),
                    "mean_abs_return_pct": float(realised.loc[mask].mean()),
                    "worst_day_pct": float(self.result.returns.loc[mask].min()),
                }
            )
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Significance testing
    # ------------------------------------------------------------------
    def dm_results(self, loss: str = "qlike", regime: Optional[str] = None):
        proxy = self.result.proxy
        forecasts = self.result.forecasts
        if regime is not None:
            mask = self.result.regimes == regime
            proxy = proxy.loc[mask]
            forecasts = forecasts.loc[mask]

        return pairwise_tests(
            proxy=proxy,
            forecasts=forecasts,
            loss=loss,
            horizon=1,
            lags=self.config.evaluation.hac_lags,
            harvey_correction=self.config.evaluation.harvey_correction,
            labels=self.labels,
        )

    def dm_table(self, loss: str = "qlike") -> pd.DataFrame:
        frame = results_to_frame(self.dm_results(loss=loss))
        alpha = self.config.evaluation.var_test_significance
        frame["significant"] = frame["p_value"] < alpha
        return frame

    def dm_matrix(self, loss: str = "qlike", value: str = "statistic") -> pd.DataFrame:
        return statistic_matrix(self.dm_results(loss=loss), order=self.model_order, value=value)

    def dm_by_regime(self, loss: str = "qlike") -> pd.DataFrame:
        rows = []
        alpha = self.config.evaluation.var_test_significance
        for name in self._regime_names():
            if (self.result.regimes == name).sum() < 20:
                continue
            for outcome in self.dm_results(loss=loss, regime=name):
                record = outcome.to_dict()
                record["regime"] = name
                record["significant"] = outcome.p_value < alpha
                rows.append(record)
        if not rows:
            return pd.DataFrame()
        frame = pd.DataFrame(rows)
        front = ["regime", "model_a", "model_b", "n_observations"]
        return frame[front + [c for c in frame.columns if c not in front]]

    # ------------------------------------------------------------------
    # Risk management assessment
    # ------------------------------------------------------------------
    def var_table(self) -> pd.DataFrame:
        results = run_var_backtests(
            returns=self.result.returns,
            forecasts=self.result.forecasts,
            confidence_levels=self.config.evaluation.var_confidence_levels,
            labels=self.labels,
        )
        return var_results_to_frame(results, alpha=self.config.evaluation.var_test_significance)

    # ------------------------------------------------------------------
    def runtime_table(self) -> pd.DataFrame:
        frame = pd.DataFrame([s.to_dict() for s in self.result.summaries])
        if frame.empty:
            return frame
        frame = frame.rename(columns={"label": "model"}).set_index("model")
        return frame.drop(columns=["key"])

    def build_all(self) -> Dict[str, pd.DataFrame]:
        tables = {
            "overall_metrics": self.overall_metrics(),
            "rank_comparison": self.rank_comparison(),
            "regime_qlike": self.regime_metrics("qlike"),
            "regime_mse": self.regime_metrics("mse"),
            "regime_detail": self.regime_detail(),
            "regime_profile": self.regime_observation_counts(),
            "dm_qlike": self.dm_table("qlike"),
            "dm_mse": self.dm_table("mse"),
            "dm_matrix_qlike": self.dm_matrix("qlike"),
            "dm_pvalue_matrix_qlike": self.dm_matrix("qlike", value="p_value"),
            "dm_by_regime_qlike": self.dm_by_regime("qlike"),
            "var_backtests": self.var_table(),
            "runtime": self.runtime_table(),
        }
        return {name: table for name, table in tables.items() if table is not None}

    def save(self, tables: Dict[str, pd.DataFrame], directory: Optional[Path] = None) -> None:
        directory = Path(directory or self.config.backtest.table_dir)
        directory.mkdir(parents=True, exist_ok=True)
        for name, table in tables.items():
            if table.empty:
                continue
            include_index = not isinstance(table.index, pd.RangeIndex)
            table.to_csv(directory / f"{name}.csv", index=include_index)
        logger.info("Wrote %d result tables to %s", len(tables), directory)

    def _regime_names(self) -> List[str]:
        configured = [regime.name for regime in self.config.regimes]
        present = set(self.result.regimes.unique())
        names = [name for name in configured if name in present]
        if UNCLASSIFIED in present:
            names.append(UNCLASSIFIED)
        return names
