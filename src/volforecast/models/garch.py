"""GARCH-family conditional variance models.

All three specifications are estimated by Gaussian quasi-maximum likelihood.
Daily equity returns are leptokurtic, so the Gaussian likelihood is
deliberately misspecified with respect to the innovation distribution; under
the standard regularity conditions the QML estimator of the variance
parameters remains consistent and asymptotically normal, and it keeps the
distributional assumptions identical across the econometric baselines and the
neural network, which makes no distributional assumption at all.
"""

from __future__ import annotations

import warnings
from typing import Dict, Optional

import numpy as np
import pandas as pd
from arch import arch_model
from arch.univariate.base import ARCHModelResult

from ..data.features import Dataset
from ..utils import get_logger
from .base import ModelFitError, VolatilityModel, previous_session

logger = get_logger(__name__)


class GarchFamilyModel(VolatilityModel):
    """Shared estimation and forecasting logic for the ``arch`` specifications.

    Subclasses supply the volatility process arguments. The class separates
    estimation from filtering: parameters are re-optimised only on refit dates,
    while forecasts on intervening dates are produced by running the variance
    recursion forward over the newly observed returns with the parameters held
    fixed. That mirrors how a desk actually operates a calibrated model and
    keeps the refit cadence a free parameter rather than a hidden assumption.
    """

    family = "GARCH family"
    vol_kwargs: Dict[str, object] = {}

    def __init__(self, distribution: str = "normal", mean: str = "Constant"):
        super().__init__()
        self.distribution = distribution
        self.mean = mean
        self._params: Optional[pd.Series] = None
        self._result: Optional[ARCHModelResult] = None

    def _build(self, returns: np.ndarray):
        return arch_model(
            returns,
            mean=self.mean,
            dist=self.distribution,
            rescale=False,
            **self.vol_kwargs,
        )

    def fit(self, dataset: Dataset, train_end: pd.Timestamp) -> None:
        returns = dataset.returns.loc[:train_end]
        if len(returns) < 100:
            raise ModelFitError(
                f"{self.label} needs at least 100 observations, received {len(returns)}"
            )

        model = self._build(returns.to_numpy(dtype=float))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                result = model.fit(disp="off", show_warning=False, options={"maxiter": 500})
            except Exception as exc:  # pragma: no cover - optimiser instability
                if self._params is None:
                    raise ModelFitError(f"{self.label} failed to converge: {exc}") from exc
                # Fall back to the previous parameter vector rather than
                # abandoning the walk-forward run on a single bad window.
                logger.warning(
                    "%s failed at %s; retaining previous estimates", self.label, train_end.date()
                )
                # The cached result object belongs to an earlier window, so it
                # must be discarded: predictions from here are produced by
                # filtering the retained parameters over the current history.
                self._result = None
                self._train_end = train_end
                return

        self._params = result.params
        self._result = result
        self._fitted = True
        self._train_end = train_end

    def predict(self, dataset: Dataset, target_date: pd.Timestamp) -> float:
        if self._params is None:
            raise ModelFitError(f"{self.label} must be fitted before predicting")

        history_end = previous_session(dataset.returns.index, target_date)
        returns = dataset.returns.loc[:history_end]

        if self._result is not None and self._train_end == history_end:
            forecast = self._result.forecast(horizon=1, reindex=False)
        else:
            # Parameters are held at their last estimated values and the
            # variance recursion is filtered forward over the returns observed
            # since the refit. No re-optimisation takes place here.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fixed = self._build(returns.to_numpy(dtype=float)).fix(self._params.to_numpy())
            forecast = fixed.forecast(horizon=1, reindex=False)

        variance = float(np.asarray(forecast.variance)[-1, 0])
        if not np.isfinite(variance) or variance <= 0.0:
            raise ModelFitError(
                f"{self.label} produced a non-positive variance forecast for {target_date.date()}"
            )
        return variance

    def parameters_snapshot(self) -> Dict[str, float]:
        if self._params is None:
            return {}
        return {str(k): float(v) for k, v in self._params.items()}


class Garch11(GarchFamilyModel):
    r"""Bollerslev GARCH(1,1).

    .. math::
        \sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2

    Symmetric by construction: the squared innovation discards the sign of the
    shock, so a fall and a rally of equal magnitude imply the same forecast.
    """

    key = "garch"
    label = "GARCH(1,1)"
    vol_kwargs = {"vol": "GARCH", "p": 1, "q": 1}


class Egarch11(GarchFamilyModel):
    r"""Nelson EGARCH(1,1).

    The recursion is specified on :math:`\ln \sigma_t^2`, which guarantees a
    positive variance without constraining the parameters, and includes a
    signed term whose coefficient measures the asymmetric response directly.
    """

    key = "egarch"
    label = "EGARCH(1,1)"
    vol_kwargs = {"vol": "EGARCH", "p": 1, "o": 1, "q": 1}


class GjrGarch11(GarchFamilyModel):
    r"""Glosten-Jagannathan-Runkle GARCH(1,1,1).

    .. math::
        \sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2
                     + \gamma I[\epsilon_{t-1} < 0]\epsilon_{t-1}^2
                     + \beta \sigma_{t-1}^2

    A positive :math:`\gamma` means negative shocks raise next-day variance by
    :math:`(\alpha + \gamma)\epsilon_{t-1}^2` against :math:`\alpha
    \epsilon_{t-1}^2` for positive shocks of the same size.
    """

    key = "gjr_garch"
    label = "GJR-GARCH(1,1,1)"
    vol_kwargs = {"vol": "GARCH", "p": 1, "o": 1, "q": 1}
