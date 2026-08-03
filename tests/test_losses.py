"""Properties of the QLIKE objective that the methodology depends on."""

import numpy as np
import pytest
import torch

from volforecast.evaluation.metrics import qlike_loss
from volforecast.models.losses import MseLoss, QlikeLoss, build_loss


def test_qlike_is_zero_only_at_a_perfect_forecast():
    proxy = np.array([0.5, 1.0, 4.0, 12.0])
    assert qlike_loss(proxy, proxy) == pytest.approx(np.zeros(4), abs=1e-12)
    assert np.all(qlike_loss(proxy, proxy * 1.5) > 0)
    assert np.all(qlike_loss(proxy, proxy * 0.5) > 0)


def test_qlike_penalises_under_forecasts_more_than_over_forecasts():
    """The asymmetry is the reason the loss is preferred for risk work.

    Halving the forecast and doubling it are equally wrong in ratio terms, but
    only the first understates risk, and QLIKE charges more for it.
    """
    proxy = np.array([1.0])
    under = qlike_loss(proxy, proxy * 0.5)[0]
    over = qlike_loss(proxy, proxy * 2.0)[0]
    assert under > over


def test_qlike_diverges_as_the_forecast_approaches_zero():
    proxy = np.array([1.0])
    losses = [qlike_loss(proxy, np.array([f]))[0] for f in (1e-1, 1e-2, 1e-3, 1e-4)]
    assert all(later > earlier for earlier, later in zip(losses, losses[1:]))
    assert losses[-1] > 1000.0


def test_torch_implementation_matches_the_numpy_metric():
    """Training objective and evaluation metric must be the same function."""
    rng = np.random.default_rng(7)
    proxy = rng.gamma(shape=2.0, scale=0.5, size=256)
    forecast = rng.gamma(shape=2.0, scale=0.5, size=256)

    reference = qlike_loss(proxy, forecast).mean()
    computed = QlikeLoss()(
        torch.tensor(forecast, dtype=torch.float64),
        torch.tensor(proxy, dtype=torch.float64),
    )
    assert float(computed) == pytest.approx(reference, rel=1e-10)


def test_qlike_gradient_points_towards_the_target():
    """d/dyhat = (yhat - y) / yhat^2, so the sign must flip at yhat = y."""
    proxy = torch.tensor([2.0], dtype=torch.float64)

    for forecast_value, expected_sign in ((1.0, -1.0), (4.0, 1.0)):
        forecast = torch.tensor([forecast_value], dtype=torch.float64, requires_grad=True)
        QlikeLoss()(forecast, proxy).backward()
        gradient = float(forecast.grad)
        assert np.sign(gradient) == expected_sign

        analytic = (forecast_value - 2.0) / forecast_value**2
        assert gradient == pytest.approx(analytic, rel=1e-9)


def test_qlike_gradient_magnitude_is_larger_for_under_forecasts():
    proxy = torch.tensor([1.0], dtype=torch.float64)
    gradients = {}
    for name, value in (("under", 0.5), ("over", 2.0)):
        forecast = torch.tensor([value], dtype=torch.float64, requires_grad=True)
        QlikeLoss()(forecast, proxy).backward()
        gradients[name] = abs(float(forecast.grad))
    assert gradients["under"] > gradients["over"]


def test_mse_treats_symmetric_errors_identically():
    proxy = torch.tensor([4.0], dtype=torch.float64)
    low = float(MseLoss()(torch.tensor([2.0], dtype=torch.float64), proxy))
    high = float(MseLoss()(torch.tensor([6.0], dtype=torch.float64), proxy))
    assert low == pytest.approx(high)


def test_loss_reductions_and_factory():
    proxy = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    forecast = torch.tensor([1.5, 1.5, 1.5], dtype=torch.float64)

    elementwise = QlikeLoss(reduction="none")(forecast, proxy)
    assert elementwise.shape == (3,)
    assert float(QlikeLoss(reduction="mean")(forecast, proxy)) == pytest.approx(
        float(elementwise.mean())
    )
    assert float(QlikeLoss(reduction="sum")(forecast, proxy)) == pytest.approx(
        float(elementwise.sum())
    )

    assert isinstance(build_loss("qlike"), QlikeLoss)
    assert isinstance(build_loss("MSE"), MseLoss)
    with pytest.raises(ValueError):
        build_loss("huber")
