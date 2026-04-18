"""Quantile Regression Neural Network baseline (Koenker 2005).

Predicts multiple quantiles simultaneously via pinball loss.
Distribution-free — no Gaussian assumption.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from automl_package.enums import TaskType, UncertaintyMethod
from automl_package.models.base_pytorch import PyTorchModelBase
from automl_package.utils.distributions import QuantileDistribution


_DEFAULT_QUANTILES = (0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975)


class _QuantileNetwork(nn.Module):
    """MLP producing one output per quantile level."""

    def __init__(self, input_size: int, hidden_size: int, n_hidden: int, n_quantiles: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_size
        for _ in range(n_hidden):
            layers.extend([nn.Linear(in_dim, hidden_size), nn.ReLU(), nn.Dropout(0.1)])
            in_dim = hidden_size
        layers.append(nn.Linear(hidden_size, n_quantiles))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # (B, Q)


class QuantileRegressionNN(PyTorchModelBase):
    """Quantile Regression NN for distribution-free uncertainty estimation.

    Predicts multiple quantiles simultaneously. The predicted quantiles
    define a non-parametric predictive distribution.

    Args:
        quantiles: Tuple of quantile levels to predict.
        hidden_size: Width of hidden layers.
        n_hidden: Number of hidden layers.
    """

    def __init__(
        self,
        quantiles: tuple[float, ...] = _DEFAULT_QUANTILES,
        hidden_size: int = 64,
        n_hidden: int = 2,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("task_type", TaskType.REGRESSION)
        kwargs.setdefault("uncertainty_method", UncertaintyMethod.PROBABILISTIC)
        super().__init__(**kwargs)
        self.quantiles = tuple(sorted(quantiles))
        self.hidden_size = hidden_size
        self.n_hidden = n_hidden

    @property
    def name(self) -> str:
        return f"QR-NN(Q={len(self.quantiles)})"

    def build_model(self) -> None:
        self.model = _QuantileNetwork(self.input_size, self.hidden_size, self.n_hidden, len(self.quantiles))
        self.model.to(self.device)

    @staticmethod
    def _pinball_loss(y_true: torch.Tensor, y_pred: torch.Tensor, taus: torch.Tensor) -> torch.Tensor:
        """Vectorized pinball loss across all quantiles."""
        diff = y_true.unsqueeze(-1) - y_pred  # (B, Q)
        return torch.mean(torch.maximum(taus * diff, (taus - 1) * diff))

    def _fit_single(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        forced_iterations: int | None = None,
        forward_pass_kwargs: dict[str, Any] | None = None,
    ) -> tuple[int, list[float]]:
        self.input_size = x_train.shape[1]
        self.build_model()
        self._setup_optimizers(self.model)

        n_epochs = forced_iterations or self.n_epochs
        device = self.device
        train_losses: list[float] = []

        x_t = torch.tensor(x_train, dtype=torch.float32, device=device)
        y_t = torch.tensor(y_train, dtype=torch.float32, device=device).ravel()
        taus = torch.tensor(self.quantiles, dtype=torch.float32, device=device)

        for epoch in range(n_epochs):
            self.model.train()
            self.optimizer.zero_grad()
            q_pred = self.model(x_t)
            loss = self._pinball_loss(y_t, q_pred, taus)
            loss.backward()
            self.optimizer.step()
            train_losses.append(loss.item())

        return n_epochs, train_losses

    def predict(self, x: np.ndarray | pd.DataFrame, filter_data: bool = True) -> np.ndarray:
        """Returns median (τ=0.5) prediction."""
        if filter_data:
            x = self._filter_predict_data(x)
        x = np.asarray(x, dtype=np.float32)
        self.model.eval()
        with torch.no_grad():
            q_pred = self.model(torch.tensor(x, dtype=torch.float32, device=self.device))
        q_pred_np = q_pred.cpu().numpy()
        # Return median quantile (interpolate if 0.5 not in quantiles)
        return np.array([np.interp(0.5, self.quantiles, q_pred_np[i]) for i in range(len(x))])

    def predict_uncertainty(self, x: np.ndarray | pd.DataFrame, filter_data: bool = True) -> np.ndarray:
        """Uncertainty from IQR: std ≈ IQR / 1.35."""
        if filter_data:
            x = self._filter_predict_data(x)
        x = np.asarray(x, dtype=np.float32)
        self.model.eval()
        with torch.no_grad():
            q_pred = self.model(torch.tensor(x, dtype=torch.float32, device=self.device))
        q_pred_np = q_pred.cpu().numpy()
        q25 = np.array([np.interp(0.25, self.quantiles, q_pred_np[i]) for i in range(len(x))])
        q75 = np.array([np.interp(0.75, self.quantiles, q_pred_np[i]) for i in range(len(x))])
        return np.maximum((q75 - q25) / 1.35, 1e-9)

    def predict_quantiles(self, x: np.ndarray | pd.DataFrame, filter_data: bool = True) -> np.ndarray:
        """Returns all predicted quantile values, shape (N, Q)."""
        if filter_data:
            x = self._filter_predict_data(x)
        x = np.asarray(x, dtype=np.float32)
        self.model.eval()
        with torch.no_grad():
            q_pred = self.model(torch.tensor(x, dtype=torch.float32, device=self.device))
        return q_pred.cpu().numpy()

    def predict_distribution(self, x: np.ndarray | pd.DataFrame, filter_data: bool = True) -> QuantileDistribution:
        """Returns a QuantileDistribution for evaluation with scoring/calibration."""
        q_vals = self.predict_quantiles(x, filter_data)
        return QuantileDistribution(np.array(self.quantiles), q_vals)

    def get_hyperparameter_search_space(self) -> dict[str, Any]:
        return {
            "hidden_size": ("categorical", [32, 64, 128, 256]),
            "n_hidden": ("int", 1, 4),
            "learning_rate": ("float", 1e-4, 1e-2, True),
        }

    def get_classifier_predictions(self, x: np.ndarray | pd.DataFrame, y_true_original: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError("QR-NN does not support get_classifier_predictions.")

    def get_shap_explainer_info(self) -> dict[str, Any]:
        """Returns empty dict — SHAP not supported for QR-NN."""
        return {}
