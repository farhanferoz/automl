"""FT-Transformer baseline (Gorishniy et al., 2021, NeurIPS).

Tabular transformer that tokenizes each feature into an embedding and
applies self-attention. Wrapped as a baseline — not integrated into the
core package architecture.

Install: pip install rtdl-num-embeddings (or rtdl)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from automl_package.enums import TaskType, UncertaintyMethod
from automl_package.models.base_pytorch import PyTorchModelBase


class _FTTransformerNet(nn.Module):
    """Simple FT-Transformer-style architecture for tabular data.

    Uses learned feature embeddings + CLS token + transformer encoder.
    Produces (mean, log_var) for probabilistic regression.
    """

    def __init__(self, n_features: int, d_model: int, n_heads: int, n_layers: int, dropout: float) -> None:
        super().__init__()
        self.feature_embeddings = nn.Linear(1, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, 2)  # mean + log_var
        self.n_features = n_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, F) → (B, F, 1) → (B, F, d_model)
        tokens = self.feature_embeddings(x.unsqueeze(-1))
        # Prepend CLS token
        cls = self.cls_token.expand(x.size(0), -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)  # (B, F+1, d_model)
        out = self.transformer(tokens)
        cls_out = out[:, 0, :]  # CLS token output
        return self.head(cls_out)  # (B, 2)


class FTTransformerModel(PyTorchModelBase):
    """FT-Transformer wrapper for tabular regression with uncertainty.

    Args:
        d_model: Embedding dimension.
        n_heads: Number of attention heads.
        n_layers: Number of transformer encoder layers.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("task_type", TaskType.REGRESSION)
        kwargs.setdefault("uncertainty_method", UncertaintyMethod.PROBABILISTIC)
        super().__init__(**kwargs)
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout_val = dropout

    @property
    def name(self) -> str:
        return f"FT-Transformer(d={self.d_model},L={self.n_layers})"

    def build_model(self) -> None:
        self.model = _FTTransformerNet(
            n_features=self.input_size,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            dropout=self.dropout_val,
        )
        self.model.to(self.device)

    def _fit_single(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        forced_iterations: int | None = None,
        forward_pass_kwargs: dict[str, Any] | None = None,
    ) -> tuple[int, list[float]]:
        from automl_package.utils.losses import nll_loss

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
            loss = nll_loss(outputs, y_t)
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
            out = self.model(torch.tensor(x, dtype=torch.float32, device=self.device))
        return out[:, 0].cpu().numpy()

    def predict_uncertainty(self, x: np.ndarray | pd.DataFrame, filter_data: bool = True) -> np.ndarray:
        if filter_data:
            x = self._filter_predict_data(x)
        x = np.asarray(x, dtype=np.float32)
        self.model.eval()
        with torch.no_grad():
            out = self.model(torch.tensor(x, dtype=torch.float32, device=self.device))
        return torch.exp(0.5 * out[:, 1]).cpu().numpy()

    def get_hyperparameter_search_space(self) -> dict[str, Any]:
        return {
            "d_model": ("categorical", [32, 64, 128]),
            "n_heads": ("categorical", [2, 4, 8]),
            "n_layers": ("int", 1, 4),
            "learning_rate": ("float", 1e-4, 1e-2, True),
            "dropout_val": ("float", 0.0, 0.3),
        }

    def get_classifier_predictions(self, x: np.ndarray | pd.DataFrame, y_true_original: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError("FT-Transformer does not support get_classifier_predictions.")

    def get_shap_explainer_info(self) -> dict[str, Any]:
        """Returns empty dict — SHAP not supported for FT-Transformer."""
        return {}
