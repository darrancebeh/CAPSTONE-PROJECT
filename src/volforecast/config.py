"""Typed configuration objects loaded from the project YAML file."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def _resolve(path: str) -> Path:
    """Interpret a configured path relative to the project root."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


@dataclass(frozen=True)
class DataConfig:
    raw_path: Path
    processed_dir: Path
    symbol: str
    price_column: str
    start_date: str
    end_date: str
    return_scale: float
    min_variance_proxy: float

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "DataConfig":
        return cls(
            raw_path=_resolve(raw["raw_path"]),
            processed_dir=_resolve(raw["processed_dir"]),
            symbol=raw["symbol"],
            price_column=raw["price_column"],
            start_date=raw["start_date"],
            end_date=raw["end_date"],
            return_scale=float(raw["return_scale"]),
            min_variance_proxy=float(raw["min_variance_proxy"]),
        )


@dataclass(frozen=True)
class FeatureConfig:
    sequence_length: int
    realised_vol_windows: List[int]
    ewma_lambda: float
    volume_ma_window: int

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "FeatureConfig":
        return cls(
            sequence_length=int(raw["sequence_length"]),
            realised_vol_windows=[int(w) for w in raw["realised_vol_windows"]],
            ewma_lambda=float(raw["ewma_lambda"]),
            volume_ma_window=int(raw["volume_ma_window"]),
        )


@dataclass(frozen=True)
class SplitConfig:
    initial_train_end: str
    min_train_observations: int

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "SplitConfig":
        return cls(
            initial_train_end=raw["initial_train_end"],
            min_train_observations=int(raw["min_train_observations"]),
        )


@dataclass(frozen=True)
class BacktestConfig:
    refit_every: Dict[str, int]
    forecast_dir: Path
    table_dir: Path
    figure_dir: Path

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "BacktestConfig":
        return cls(
            refit_every={k: int(v) for k, v in raw["refit_every"].items()},
            forecast_dir=_resolve(raw["forecast_dir"]),
            table_dir=_resolve(raw["table_dir"]),
            figure_dir=_resolve(raw["figure_dir"]),
        )

    def cadence(self, model_key: str) -> int:
        return self.refit_every.get(model_key, 1)


@dataclass(frozen=True)
class Regime:
    name: str
    start: str
    end: str


@dataclass(frozen=True)
class EvaluationConfig:
    primary_loss: str
    hac_lags: Optional[int]
    harvey_correction: bool
    var_confidence_levels: List[float]
    var_test_significance: float

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "EvaluationConfig":
        return cls(
            primary_loss=raw["primary_loss"],
            hac_lags=None if raw.get("hac_lags") is None else int(raw["hac_lags"]),
            harvey_correction=bool(raw.get("harvey_correction", True)),
            var_confidence_levels=[float(c) for c in raw["var_confidence_levels"]],
            var_test_significance=float(raw["var_test_significance"]),
        )


@dataclass(frozen=True)
class Config:
    name: str
    random_seed: int
    data: DataConfig
    features: FeatureConfig
    split: SplitConfig
    backtest: BacktestConfig
    models: Dict[str, Any]
    regimes: List[Regime]
    evaluation: EvaluationConfig
    raw: Dict[str, Any] = field(repr=False, default_factory=dict)

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "Config":
        path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
        with open(path, "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)

        return cls(
            name=raw["project"]["name"],
            random_seed=int(raw["project"]["random_seed"]),
            data=DataConfig.from_dict(raw["data"]),
            features=FeatureConfig.from_dict(raw["features"]),
            split=SplitConfig.from_dict(raw["split"]),
            backtest=BacktestConfig.from_dict(raw["backtest"]),
            models=raw["models"],
            regimes=[Regime(**item) for item in raw["regimes"]],
            evaluation=EvaluationConfig.from_dict(raw["evaluation"]),
            raw=raw,
        )

    def ensure_output_dirs(self) -> None:
        for directory in (
            self.data.processed_dir,
            self.backtest.forecast_dir,
            self.backtest.table_dir,
            self.backtest.figure_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
