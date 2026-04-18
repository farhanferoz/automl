"""Mixture Density Network baseline (Bishop 1994).

Neural network outputting parameters of a Gaussian mixture model.
Closest neural-network baseline to ProbReg for multimodal predictions.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from automl_package.enums import TaskType, UncertaintyMethod
from automl_package.models.base_pytorch import PyTorchModelBase
from automl_package.utils.distributions import MixtureOfGaussiansDistribution
from automl_package.utils.pytorch_utils import get_device


class _MDNNetwork(nn.Module):
    """MDN architecture: shared trunk → (weights, means, log_vars)."""

    def __init__(self, input_size: int, hidden_size: int, n_hidden: int, n_components: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_size
        for _ in range(n_hidden):
            layers.extend([nn.Linear(in_dim, hidden_size), nn.ReLU(), nn.Dropout(0.1)])
            in_dim = hidden_size
        self.trunk = nn.Sequential(*layers)

        self.weight_head = nn.Linear(hidden_size, n_components)
        self.mean_head = nn.Linear(hidden_size, n_components)
        self.log_var_head = nn.Linear(hidden_size, n_components)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        weights = torch.softmax(self.weight_head(h), dim=-1)  # (B, K)
        means = self.mean_head(h)  # (B, K)
        log_vars = self.log_var_head(h)  # (B, K)
        return weights, means, log_vars


class MixtureDensityNetwork(PyTorchModelBase):
    """Mixture Density Network for probabilistic regression.

    Outputs a K-component Gaussian mixture per input, trained via
    negative log-likelihood of the mixture.

    Args:
        n_components: Number of Gaussian mixture components.
        hidden_size: Width of hidden layers in the trunk.
        n_hidden: Number of hidden layers in the trunk.
    """

    def __init__(
        self,
        n_components: int = 5,
        hidden_size: int = 64,
        n_hidden: int = 2,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("task_type", TaskType.REGRESSION)
        kwargs.setdefault("uncertainty_method", UncertaintyMethod.PROBABILISTIC)
        super().__init__(**kwargs)
        self.n_components = n_components
        self.hidden_size = hidden_size
        self.n_hidden = n_hidden

    @property
    def name(self) -> str:
        return f"MDN(K={self.n_components})"

    def build_model(self) -> None:
        self.model = _MDNNetwork(self.input_size, self.hidden_size, self.n_hidden, self.n_components)
        self.model.to(self.device)

    def _compute_loss(self, outputs: tuple[torch.Tensor, ...], targets: torch.Tensor) -> torch.Tensor:
        """Negative log-likelihood of the mixture."""
        weights, means, log_vars = outputs
        targets = targets.view(-1, 1).expand_as(means)  # (B, K)
        vars_ = torch.exp(log_vars).clamp(min=1e-6)
        # Log of Gaussian PDF per component
        log_component = -0.5 * (torch.log(2 * torch.pi * vars_) + (targets - means) ** 2 / vars_)
        # Log of mixture: log sum_k w_k * N(y; mu_k, var_k)
        log_mix = torch.logsumexp(torch.log(weights.clamp(min=1e-30)) + log_component, dim=-1)
        return -log_mix.mean()

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

        for epoch in range(n_epochs):
            self.model.train()
            self.optimizer.zero_grad()
            outputs = self.model(x_t)
            loss = self._compute_loss(outputs, y_t)
            loss.backward()
            self.optimizer.step()
            train_losses.append(loss.item())

        return n_epochs, train_losses

    def predict(self, x: np.ndarray | pd.DataFrame, filter_data: bool = True) -> np.ndarray:
        if filter_data:
            x = self._filter_predict_data(x)
        x = np.asarray(x, dtype=np.float32)
        self.model.eval()
        with torch.no_grad():
            x_t = torch.tensor(x, dtype=torch.float32, device=self.device)
            weights, means, log_vars = self.model(x_t)
        # Mixture mean
        return (weights * means).sum(dim=-1).cpu().numpy()

    def predict_uncertainty(self, x: np.ndarray | pd.DataFrame, filter_data: bool = True) -> np.ndarray:
        if filter_data:
            x = self._filter_predict_data(x)
        x = np.asarray(x, dtype=np.float32)
        self.model.eval()
        with torch.no_grad():
            x_t = torch.tensor(x, dtype=torch.float32, device=self.device)
            weights, means, log_vars = self.model(x_t)
            weights_np = weights.cpu().numpy()
            means_np = means.cpu().numpy()
            stds_np = torch.exp(0.5 * log_vars).cpu().numpy()
        # Law of total variance
        dist = MixtureOfGaussiansDistribution(weights_np, means_np, stds_np)
        return np.sqrt(dist.variance)

    def predict_distribution(self, x: np.ndarray | pd.DataFrame, filter_data: bool = True) -> MixtureOfGaussiansDistribution:
        """Returns the full mixture distribution for richer evaluation."""
        if filter_data:
            x = self._filter_predict_data(x)
        x = np.asarray(x, dtype=np.float32)
        self.model.eval()
        with torch.no_grad():
            x_t = torch.tensor(x, dtype=torch.float32, device=self.device)
            weights, means, log_vars = self.model(x_t)
        return MixtureOfGaussiansDistribution(
            weights.cpu().numpy(),
            means.cpu().numpy(),
            torch.exp(0.5 * log_vars).cpu().numpy(),
        )

    def get_hyperparameter_search_space(self) -> dict[str, Any]:
        return {
            "n_components": ("int", 2, 20),
            "hidden_size": ("categorical", [32, 64, 128, 256]),
            "n_hidden": ("int", 1, 4),
            "learning_rate": ("float", 1e-4, 1e-2, True),
        }

    def get_classifier_predictions(self, x: np.ndarray | pd.DataFrame, y_true_original: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError("MDN does not support get_classifier_predictions.")

    def get_shap_explainer_info(self) -> dict[str, Any]:
        """Returns empty dict — SHAP not supported for MDN."""
        return {}
