"""Model implementations and the registry used to build them from config."""

from __future__ import annotations

from typing import Dict, List

from ..config import Config
from .base import ModelFitError, VolatilityModel, previous_session
from .garch import Egarch11, Garch11, GjrGarch11
from .losses import MseLoss, QlikeLoss, build_loss
from .lstm import LstmVolatilityModel, TrainingHistory, VolatilityLSTM

__all__ = [
    "VolatilityModel",
    "ModelFitError",
    "previous_session",
    "Garch11",
    "Egarch11",
    "GjrGarch11",
    "QlikeLoss",
    "MseLoss",
    "build_loss",
    "VolatilityLSTM",
    "LstmVolatilityModel",
    "TrainingHistory",
    "build_model_suite",
]


def build_model_suite(config: Config) -> List[VolatilityModel]:
    """Instantiate every model in the comparison, in presentation order."""
    garch_settings: Dict = config.models["garch_family"]
    lstm_settings: Dict = config.models["lstm"]

    distribution = garch_settings.get("distribution", "normal")
    mean = garch_settings.get("mean", "Constant")

    def make_lstm(loss_name: str, key: str, label: str) -> LstmVolatilityModel:
        return LstmVolatilityModel(
            loss_name=loss_name,
            sequence_length=config.features.sequence_length,
            hidden_size=int(lstm_settings["hidden_size"]),
            num_layers=int(lstm_settings["num_layers"]),
            dropout=float(lstm_settings["dropout"]),
            learning_rate=float(lstm_settings["learning_rate"]),
            weight_decay=float(lstm_settings["weight_decay"]),
            batch_size=int(lstm_settings["batch_size"]),
            max_epochs=int(lstm_settings["max_epochs"]),
            patience=int(lstm_settings["patience"]),
            validation_fraction=float(lstm_settings["validation_fraction"]),
            grad_clip=float(lstm_settings["grad_clip"]),
            min_variance=float(lstm_settings["min_variance"]),
            seed=config.random_seed,
            key=key,
            label=label,
        )

    return [
        Garch11(distribution=distribution, mean=mean),
        Egarch11(distribution=distribution, mean=mean),
        GjrGarch11(distribution=distribution, mean=mean),
        make_lstm("qlike", "lstm_qlike", "LSTM (QLIKE)"),
        make_lstm("mse", "lstm_mse", "LSTM (MSE)"),
    ]
