"""LSTM conditional variance forecaster with a pluggable econometric loss.

The network is deliberately small. With roughly one thousand training
sequences at the first refit and a single noisy target, capacity is not the
binding constraint; the binding constraint is the signal-to-noise ratio of the
squared-return proxy. The architecture is held fixed across both training
objectives so that the QLIKE-versus-MSE comparison isolates the loss function
rather than the model.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..data.features import Dataset, build_sequences
from ..utils import get_logger
from .base import ModelFitError, VolatilityModel, previous_session
from .losses import build_loss

logger = get_logger(__name__)


@dataclass
class TrainingHistory:
    """Diagnostics retained from the most recent estimation."""

    train_end: Optional[pd.Timestamp] = None
    n_train: int = 0
    n_validation: int = 0
    epochs_run: int = 0
    best_epoch: int = 0
    best_validation_loss: float = float("nan")
    final_train_loss: float = float("nan")
    train_losses: List[float] = field(default_factory=list)
    validation_losses: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "train_end": None if self.train_end is None else self.train_end.date().isoformat(),
            "n_train": self.n_train,
            "n_validation": self.n_validation,
            "epochs_run": self.epochs_run,
            "best_epoch": self.best_epoch,
            "best_validation_loss": self.best_validation_loss,
            "final_train_loss": self.final_train_loss,
        }


class StandardScaler:
    """Column-wise standardisation fitted strictly on the training window."""

    def __init__(self) -> None:
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None

    def fit(self, values: np.ndarray) -> "StandardScaler":
        self.mean_ = values.mean(axis=0)
        scale = values.std(axis=0)
        # A constant column carries no information; mapping it to zero is
        # preferable to dividing by an arbitrarily small number.
        self.scale_ = np.where(scale < 1e-12, 1.0, scale)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Scaler has not been fitted")
        return (values - self.mean_) / self.scale_


class VolatilityLSTM(nn.Module):
    """Recurrent encoder with a strictly positive variance head.

    The final activation is a softplus offset by ``min_variance``. A ReLU would
    admit an exactly zero forecast, at which QLIKE and its gradient are both
    undefined; softplus is smooth everywhere and bounded away from zero once
    offset, so the optimisation graph stays well defined even when the network
    is confident that the next session will be quiet.

    Because the recurrent state is bounded by its tanh activations, the head's
    pre-activation cannot exceed ``sum(|W|) + b``, so the forecast carries an
    implied upper limit of ``softplus(sum(|W|) + b)``. That limit is a property
    of this parameterisation and is stated as a constraint of the study rather
    than treated as an incidental detail; Section 4.7 of the report quantifies
    where it lands and what it costs.
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 32,
        num_layers: int = 2,
        dropout: float = 0.2,
        min_variance: float = 1e-3,
    ):
        super().__init__()
        self.min_variance = min_variance

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)

        self._initialise_recurrent_weights()

    def _initialise_recurrent_weights(self) -> None:
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
                # Bias the forget gate towards remembering. Volatility is
                # strongly persistent, so a network that starts by discarding
                # its cell state has to unlearn that before it can fit at all.
                hidden = param.numel() // 4
                param.data[hidden : 2 * hidden].fill_(1.0)

    def set_output_bias(self, target_mean: float) -> None:
        """Centre the untrained network on the unconditional variance.

        Inverting the softplus at the mean of the training target means the
        first forward pass already predicts a plausible variance level, which
        matters under QLIKE: the loss is unbounded as the forecast approaches
        zero, so a default-initialised head can start in a region with very
        large gradients.
        """
        offset = max(target_mean - self.min_variance, 1e-6)
        # Numerically stable inverse of softplus.
        bias = offset + math.log(-math.expm1(-offset)) if offset < 20.0 else offset
        nn.init.zeros_(self.head.weight)
        nn.init.constant_(self.head.bias, bias)

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(sequences)
        last_step = self.dropout(output[:, -1, :])
        raw = self.head(last_step).squeeze(-1)
        return nn.functional.softplus(raw) + self.min_variance

    def implied_ceiling(self) -> float:
        """Largest variance this network can output, given its fitted weights.

        Reported in Section 4.7 so that the limitation is stated as a measured
        quantity rather than left for a reader to infer.
        """
        with torch.no_grad():
            reach = float(self.head.weight.abs().sum() + self.head.bias)
        return float(math.log1p(math.exp(min(reach, 30.0)))) + self.min_variance


class LstmVolatilityModel(VolatilityModel):
    """Walk-forward wrapper around :class:`VolatilityLSTM`.

    Each refit rebuilds the supervised sequences from the expanding training
    window, refits the feature scaler on that window alone, reinitialises the
    network and trains from scratch with early stopping on a chronologically
    held-out tail. Training from scratch rather than warm-starting keeps every
    refit independent of the order in which earlier refits happened to
    converge.
    """

    family = "Deep learning"

    def __init__(
        self,
        loss_name: str = "qlike",
        sequence_length: int = 21,
        hidden_size: int = 32,
        num_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        batch_size: int = 64,
        max_epochs: int = 150,
        patience: int = 15,
        validation_fraction: float = 0.15,
        grad_clip: float = 1.0,
        min_variance: float = 1e-3,
        seed: int = 42,
        key: Optional[str] = None,
        label: Optional[str] = None,
    ):
        super().__init__()
        self.loss_name = loss_name.lower()
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.validation_fraction = validation_fraction
        self.grad_clip = grad_clip
        self.min_variance = min_variance
        self.seed = seed

        self.key = key or f"lstm_{self.loss_name}"
        self.label = label or f"LSTM ({self.loss_name.upper()})"

        self.device = torch.device("cpu")
        self.criterion = build_loss(self.loss_name)
        self.network: Optional[VolatilityLSTM] = None
        self.scaler: Optional[StandardScaler] = None
        self.history = TrainingHistory()
        self._refit_count = 0
        self._feature_names: List[str] = []

    def _window_seed(self, train_end: pd.Timestamp) -> int:
        stamp = int(pd.Timestamp(train_end).strftime("%Y%m%d"))
        return (self.seed * 1_000_003 + stamp) % (2**31 - 1)

    def fit(self, dataset: Dataset, train_end: pd.Timestamp) -> None:
        candidate_dates = dataset.common_index()
        target_dates = candidate_dates[candidate_dates <= train_end]

        inputs, targets, served = build_sequences(
            features=dataset.features,
            target=dataset.proxy,
            sequence_length=self.sequence_length,
            target_dates=target_dates,
        )
        if len(served) < 200:
            raise ModelFitError(
                f"{self.label} needs at least 200 training sequences, received {len(served)}"
            )

        n_validation = max(int(round(len(served) * self.validation_fraction)), 1)
        n_train = len(served) - n_validation
        if n_train < 100:
            raise ModelFitError(f"{self.label} training split is too small ({n_train} sequences)")

        # The scaler sees only feature rows dated on or before ``train_end``.
        self.scaler = StandardScaler().fit(
            dataset.features.loc[:train_end].to_numpy(dtype=np.float64)
        )
        self._feature_names = list(dataset.features.columns)

        scaled = self.scaler.transform(inputs.reshape(-1, inputs.shape[-1])).reshape(inputs.shape)
        scaled = scaled.astype(np.float32)

        x_train = torch.from_numpy(scaled[:n_train])
        y_train = torch.from_numpy(targets[:n_train])
        x_validation = torch.from_numpy(scaled[n_train:])
        y_validation = torch.from_numpy(targets[n_train:])

        # The seed is a deterministic function of the information boundary
        # rather than of a refit counter. Two refits on the same window are
        # then bit-identical regardless of how many refits preceded them,
        # which is what makes the look-ahead audit a meaningful test.
        window_seed = self._window_seed(train_end)
        generator = torch.Generator().manual_seed(window_seed)
        torch.manual_seed(window_seed)

        self.network = VolatilityLSTM(
            n_features=inputs.shape[-1],
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            min_variance=self.min_variance,
        ).to(self.device)
        self.network.set_output_bias(float(y_train.mean()))

        loader = DataLoader(
            TensorDataset(x_train, y_train),
            batch_size=self.batch_size,
            shuffle=True,
            generator=generator,
            drop_last=False,
        )
        optimiser = torch.optim.Adam(
            self.network.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        best_state = copy.deepcopy(self.network.state_dict())
        best_validation = float("inf")
        best_epoch = 0
        epochs_without_improvement = 0
        history = TrainingHistory(
            train_end=train_end, n_train=n_train, n_validation=n_validation
        )

        for epoch in range(1, self.max_epochs + 1):
            self.network.train()
            batch_losses = []
            for batch_x, batch_y in loader:
                optimiser.zero_grad(set_to_none=True)
                prediction = self.network(batch_x)
                loss = self.criterion(prediction, batch_y)
                if not torch.isfinite(loss):
                    raise ModelFitError(f"{self.label} produced a non-finite loss at epoch {epoch}")
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.grad_clip)
                optimiser.step()
                batch_losses.append(float(loss.detach()))

            validation_loss = self._evaluate(x_validation, y_validation)
            history.train_losses.append(float(np.mean(batch_losses)))
            history.validation_losses.append(validation_loss)
            history.epochs_run = epoch

            if validation_loss < best_validation - 1e-9:
                best_validation = validation_loss
                best_epoch = epoch
                best_state = copy.deepcopy(self.network.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.patience:
                    break

        self.network.load_state_dict(best_state)
        self.network.eval()

        history.best_epoch = best_epoch
        history.best_validation_loss = best_validation
        history.final_train_loss = history.train_losses[-1] if history.train_losses else float("nan")
        self.history = history

        self._fitted = True
        self._train_end = train_end
        self._refit_count += 1

        logger.info(
            "%s refit through %s: %d sequences, best epoch %d/%d, validation %s = %.5f",
            self.label,
            train_end.date(),
            n_train,
            best_epoch,
            history.epochs_run,
            self.loss_name.upper(),
            best_validation,
        )

    @torch.no_grad()
    def _evaluate(self, inputs: torch.Tensor, targets: torch.Tensor) -> float:
        if self.network is None:
            raise ModelFitError("Network has not been initialised")
        self.network.eval()
        predictions = self.network(inputs)
        return float(self.criterion(predictions, targets))

    @torch.no_grad()
    def predict(self, dataset: Dataset, target_date: pd.Timestamp) -> float:
        if self.network is None or self.scaler is None:
            raise ModelFitError(f"{self.label} must be fitted before predicting")

        history_end = previous_session(dataset.features.index, target_date)
        window = dataset.features.loc[:history_end].iloc[-self.sequence_length :]
        if len(window) < self.sequence_length:
            raise ModelFitError(
                f"{self.label} needs {self.sequence_length} feature rows before "
                f"{target_date.date()}, found {len(window)}"
            )

        scaled = self.scaler.transform(window.to_numpy(dtype=np.float64)).astype(np.float32)
        tensor = torch.from_numpy(scaled).unsqueeze(0)

        self.network.eval()
        variance = float(self.network(tensor).item())
        if not np.isfinite(variance) or variance <= 0.0:
            raise ModelFitError(
                f"{self.label} produced a non-positive variance forecast for {target_date.date()}"
            )
        return variance

    def parameters_snapshot(self) -> Dict[str, float]:
        if self.network is None:
            return {}
        n_params = sum(p.numel() for p in self.network.parameters())
        return {
            "n_parameters": float(n_params),
            "best_epoch": float(self.history.best_epoch),
            "best_validation_loss": float(self.history.best_validation_loss),
        }
