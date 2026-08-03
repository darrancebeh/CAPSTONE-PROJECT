"""Ingestion and validation of the raw daily OHLCV feed."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from ..config import DataConfig
from ..utils import get_logger
from .calendar import trading_sessions

logger = get_logger(__name__)

REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")
PRICE_COLUMNS = ("open", "high", "low", "close")


class DataValidationError(Exception):
    """Raised when the raw feed violates a non-recoverable integrity rule."""


@dataclass
class DataQualityReport:
    """Audit trail of everything the ingestion layer found and repaired."""

    n_rows_raw: int = 0
    n_rows_clean: int = 0
    start_date: Optional[pd.Timestamp] = None
    end_date: Optional[pd.Timestamp] = None
    expected_sessions: int = 0
    missing_sessions: List[str] = field(default_factory=list)
    unexpected_sessions: List[str] = field(default_factory=list)
    duplicate_dates_dropped: int = 0
    forward_filled_prices: int = 0
    zero_volume_days: int = 0
    ohlc_violations: int = 0

    def to_dict(self) -> Dict[str, object]:
        return {
            "rows_raw": self.n_rows_raw,
            "rows_clean": self.n_rows_clean,
            "start_date": None if self.start_date is None else self.start_date.date().isoformat(),
            "end_date": None if self.end_date is None else self.end_date.date().isoformat(),
            "expected_sessions": self.expected_sessions,
            "missing_sessions": len(self.missing_sessions),
            "unexpected_sessions": len(self.unexpected_sessions),
            "duplicate_dates_dropped": self.duplicate_dates_dropped,
            "forward_filled_prices": self.forward_filled_prices,
            "zero_volume_days": self.zero_volume_days,
            "ohlc_violations": self.ohlc_violations,
        }

    def summary(self) -> str:
        lines = [f"{k}: {v}" for k, v in self.to_dict().items()]
        return "\n".join(lines)


class MarketDataLoader:
    """Loads, validates and repairs the daily OHLCV series for one symbol.

    Repairs are limited to operations that cannot introduce look-ahead bias:
    duplicate removal, chronological ordering, and forward-filling price levels
    across sessions that the exchange calendar says should exist but that the
    vendor did not deliver. Volume is filled with zero rather than carried
    forward, since a carried-forward volume would fabricate traded activity.
    """

    def __init__(self, config: DataConfig):
        self.config = config
        self.report = DataQualityReport()

    def load(self) -> pd.DataFrame:
        frame = self._read(self.config.raw_path)
        frame = self._standardise(frame)
        frame = self._restrict_to_sample(frame)
        frame = self._deduplicate(frame)
        self._check_integrity(frame)
        frame = self._align_to_calendar(frame)

        self.report.n_rows_clean = len(frame)
        self.report.start_date = frame.index[0]
        self.report.end_date = frame.index[-1]
        self.report.zero_volume_days = int((frame["volume"] <= 0).sum())

        logger.info(
            "Loaded %s: %d sessions from %s to %s",
            self.config.symbol,
            len(frame),
            self.report.start_date.date(),
            self.report.end_date.date(),
        )
        return frame

    @staticmethod
    def _read(path: Path) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"Raw data file not found: {path}")
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        if path.suffix == ".csv":
            return pd.read_csv(path)
        raise ValueError(f"Unsupported raw data format: {path.suffix}")

    def _standardise(self, frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.copy()
        frame.columns = [str(c).strip().lower() for c in frame.columns]
        self.report.n_rows_raw = len(frame)

        if "date" in frame.columns:
            frame["date"] = pd.to_datetime(frame["date"])
            frame = frame.set_index("date")
        else:
            frame.index = pd.to_datetime(frame.index)
            frame.index.name = "date"

        missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
        if missing:
            raise DataValidationError(f"Raw feed is missing required columns: {missing}")

        if "symbol" in frame.columns:
            symbols = set(frame["symbol"].unique())
            if symbols != {self.config.symbol}:
                raise DataValidationError(
                    f"Expected a single-symbol feed for {self.config.symbol}, found {sorted(symbols)}"
                )

        keep = [c for c in REQUIRED_COLUMNS if c in frame.columns]
        frame = frame[keep].astype(float)
        # Normalise away any intraday timestamp component.
        frame.index = frame.index.normalize()
        return frame.sort_index()

    def _restrict_to_sample(self, frame: pd.DataFrame) -> pd.DataFrame:
        start = pd.Timestamp(self.config.start_date)
        end = pd.Timestamp(self.config.end_date)
        return frame.loc[(frame.index >= start) & (frame.index <= end)]

    def _deduplicate(self, frame: pd.DataFrame) -> pd.DataFrame:
        duplicated = frame.index.duplicated(keep="last")
        self.report.duplicate_dates_dropped = int(duplicated.sum())
        if self.report.duplicate_dates_dropped:
            logger.warning("Dropped %d duplicate timestamps", self.report.duplicate_dates_dropped)
        return frame.loc[~duplicated]

    def _check_integrity(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            raise DataValidationError("No observations remain after applying the sample window")

        if not frame.index.is_monotonic_increasing:
            raise DataValidationError("Timestamps are not in ascending order after sorting")

        prices = frame[list(PRICE_COLUMNS)]
        if (prices <= 0).any().any():
            raise DataValidationError("Non-positive price levels detected in the raw feed")

        violations = (
            (frame["high"] < frame["low"])
            | (frame["high"] < frame["open"])
            | (frame["high"] < frame["close"])
            | (frame["low"] > frame["open"])
            | (frame["low"] > frame["close"])
        )
        self.report.ohlc_violations = int(violations.sum())
        if self.report.ohlc_violations:
            raise DataValidationError(
                f"{self.report.ohlc_violations} bars violate the high/low envelope"
            )

        if (frame["volume"] < 0).any():
            raise DataValidationError("Negative volume detected in the raw feed")

    def _align_to_calendar(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Reindex onto the exchange calendar and repair absent sessions."""
        expected = trading_sessions(frame.index[0], frame.index[-1])
        self.report.expected_sessions = len(expected)

        missing = expected.difference(frame.index)
        unexpected = frame.index.difference(expected)
        self.report.missing_sessions = [d.date().isoformat() for d in missing]
        self.report.unexpected_sessions = [d.date().isoformat() for d in unexpected]

        if len(unexpected):
            # Bars on non-session dates indicate a vendor error rather than a
            # gap; they are kept but surfaced so the anomaly is visible.
            logger.warning(
                "%d bars fall on dates the exchange calendar marks as closed (e.g. %s)",
                len(unexpected),
                unexpected[0].date(),
            )

        if not len(missing):
            return frame

        logger.warning(
            "%d expected sessions absent from the feed; forward-filling price levels",
            len(missing),
        )
        combined = frame.reindex(frame.index.union(expected)).sort_index()
        gaps = combined[list(PRICE_COLUMNS)].isna().any(axis=1)
        self.report.forward_filled_prices = int(gaps.sum())

        combined[list(PRICE_COLUMNS)] = combined[list(PRICE_COLUMNS)].ffill()
        combined["volume"] = combined["volume"].fillna(0.0)

        # A leading gap cannot be filled from history and must be discarded.
        return combined.dropna(subset=list(PRICE_COLUMNS))


def load_market_data(config: DataConfig) -> tuple[pd.DataFrame, DataQualityReport]:
    loader = MarketDataLoader(config)
    frame = loader.load()
    return frame, loader.report
