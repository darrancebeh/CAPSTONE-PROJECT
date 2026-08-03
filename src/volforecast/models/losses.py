"""Loss functions used for neural network optimisation.

The QLIKE implementation here is the differentiable counterpart of the
evaluation metric in :mod:`volforecast.evaluation.metrics`. Both are kept in
the same functional form so that the quantity the network minimises during
training is exactly the quantity it is scored on out of sample.
"""

from __future__ import annotations

import torch
from torch import nn


class QlikeLoss(nn.Module):
    r"""Quasi-likelihood loss for variance forecasts.

    .. math::
        L(y, \hat{y}) = \frac{y}{\hat{y}} - \ln\!\left(\frac{y}{\hat{y}}\right) - 1

    The loss is non-negative and attains zero only at :math:`\hat{y} = y`. Its
    derivative with respect to the forecast is :math:`(\hat{y} - y)/\hat{y}^2`,
    so the penalty on an under-forecast grows without bound as the forecast
    approaches zero while the penalty on an over-forecast is bounded by
    :math:`1/\hat{y}`. That asymmetry is the property that makes the loss a
    member of the proxy-robust Bregman family, and it is the reason the network
    trained on it cannot buy a low score by systematically understating risk.

    Parameters
    ----------
    eps:
        Floor applied to both arguments before the ratio is formed. It guards
        the logarithm against a degenerate zero proxy without materially
        altering the gradient at any economically meaningful variance level.
    reduction:
        ``"mean"``, ``"sum"`` or ``"none"``.
    """

    def __init__(self, eps: float = 1e-8, reduction: str = "mean"):
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(f"Unsupported reduction: {reduction}")
        self.eps = eps
        self.reduction = reduction

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prediction = prediction.clamp_min(self.eps)
        target = target.clamp_min(self.eps)

        ratio = target / prediction
        loss = ratio - torch.log(ratio) - 1.0

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class MseLoss(nn.Module):
    """Squared error on the variance level.

    Retained as the ablation baseline: it is the objective that applied deep
    learning work ordinarily minimises, and it is not robust to noise in the
    variance proxy.
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = (prediction - target).pow(2)
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def build_loss(name: str, **kwargs) -> nn.Module:
    """Instantiate a training objective by name."""
    key = name.strip().lower()
    if key == "qlike":
        return QlikeLoss(**kwargs)
    if key == "mse":
        return MseLoss(**kwargs)
    raise ValueError(f"Unknown loss function: {name}")
