"""Persistence layer for backtest output.

Forecast panels are written as Parquet so that the dashboard and the analysis
scripts can reload a completed run without re-estimating anything. Run
metadata is written alongside as JSON so that any cached panel can be traced
back to the configuration that produced it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from ..utils import get_logger

logger = get_logger(__name__)

FORECAST_FILE = "forecasts.parquet"
FIT_LOG_FILE = "fit_log.parquet"
METADATA_FILE = "run_metadata.json"


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (pd.Timestamp,)):
        return value.date().isoformat()
    if isinstance(value, Path):
        return str(value)
    return value


class ForecastStore:
    """Reads and writes the artefacts of a single backtest run."""

    def __init__(self, directory: Path):
        self.directory = Path(directory)

    @property
    def forecast_path(self) -> Path:
        return self.directory / FORECAST_FILE

    @property
    def fit_log_path(self) -> Path:
        return self.directory / FIT_LOG_FILE

    @property
    def metadata_path(self) -> Path:
        return self.directory / METADATA_FILE

    def exists(self) -> bool:
        return self.forecast_path.exists() and self.metadata_path.exists()

    def save(
        self,
        forecasts: pd.DataFrame,
        metadata: Dict[str, Any],
        fit_log: Optional[pd.DataFrame] = None,
    ) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)

        panel = forecasts.copy()
        panel.index.name = "date"
        panel.to_parquet(self.forecast_path)

        if fit_log is not None and not fit_log.empty:
            fit_log.to_parquet(self.fit_log_path, index=False)

        with open(self.metadata_path, "w", encoding="utf-8") as handle:
            json.dump(_jsonable(metadata), handle, indent=2)

        logger.info("Saved backtest artefacts to %s", self.directory)

    def load_forecasts(self) -> pd.DataFrame:
        if not self.forecast_path.exists():
            raise FileNotFoundError(
                f"No cached forecasts at {self.forecast_path}. Run the pipeline first."
            )
        panel = pd.read_parquet(self.forecast_path)
        panel.index = pd.DatetimeIndex(panel.index)
        return panel

    def load_metadata(self) -> Dict[str, Any]:
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"No run metadata at {self.metadata_path}")
        with open(self.metadata_path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def load_fit_log(self) -> pd.DataFrame:
        if not self.fit_log_path.exists():
            return pd.DataFrame()
        return pd.read_parquet(self.fit_log_path)
