"""FT-Transformer baseline (Gorishniy et al., 2021, NeurIPS).

Tabular transformer that tokenizes each feature into an embedding and
applies self-attention. Wrapped as a baseline -- not integrated into the
core package architecture.

Training recipe (updated post Phase 9 fragility audit):
- Pre-norm layers (norm_first=True) — standard stability fix.
- Mini-batch Adam with weight_decay (1e-5) instead of full-batch training.
- Linear warmup over ``warmup_epochs`` then constant LR.
- Gradient clipping at ``grad_clip_norm`` (default 1.0).
- Optional validation-driven early stopping via ``early_stopping_rounds``.

The original minimal wrapper trained full-batch with no warmup and lost every
UCI dataset. These additions are standard best practice for tabular transformers.
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

    Uses learned feature embeddings + CLS token + pre-norm transformer encoder.
    Produces (mean, log_var) for probabilistic regression.
    """

    def __init__(self, n_features: int, d_model: int, n_heads: int, n_layers: int, dropout: float) -> None:
        super().__init__()
        self.feature_embeddings = nn.Linear(1, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 2)  # mean + log_var
        self.n_features = n_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.feature_embeddings(x.unsqueeze(-1))
        cls = self.cls_token.expand(x.size(0), -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        out = self.transformer(tokens)
        cls_out = self.final_norm(out[:, 0, :])
        return self.head(cls_out)


class FTTransformerModel(PyTorchModelBase):
    """FT-Transformer wrapper for tabular regression with uncertainty.

    Args:
        d_model: Embedding dimension.
        n_heads: Number of attention heads.
        n_layers: Number of transformer encoder layers.
        dropout: Dropout rate.
        weight_decay: L2 on AdamW; ~1e-5 is standard for tabular transformers.
        warmup_epochs: Linear warmup over this many epochs before constant LR.
        grad_clip_norm: Gradient clipping norm.
    """

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        weight_decay: float = 1e-5,
        warmup_epochs: int = 5,
        grad_clip_norm: float = 1.0,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("task_type", TaskType.REGRESSION)
        kwargs.setdefault("uncertainty_method", UncertaintyMethod.PROBABILISTIC)
        super().__init__(**kwargs)
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout_val = dropout
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.grad_clip_norm = grad_clip_norm

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

    def _setup_optimizers(self, model: nn.Module) -> None:
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay,
        )

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
        if self.random_seed is not None:
            torch.manual_seed(self.random_seed)
        self.build_model()
        self._setup_optimizers(self.model)

        n_epochs = forced_iterations or self.n_epochs
        base_lr = self.learning_rate
        warmup = max(1, int(self.warmup_epochs))
        device = self.device

        x_t = torch.tensor(x_train, dtype=torch.float32, device=device)
        y_t = torch.tensor(y_train, dtype=torch.float32, device=device).ravel()
        if x_val is not None:
            x_v = torch.tensor(x_val, dtype=torch.float32, device=device)
            y_v = torch.tensor(y_val, dtype=torch.float32, device=device).ravel()
        else:
            x_v = y_v = None

        batch_size = max(32, min(256, x_t.size(0)))
        n_batches = max(1, x_t.size(0) // batch_size)

        train_losses: list[float] = []
        best_val = float("inf")
        patience = 0
        best_state: dict[str, torch.Tensor] | None = None
        best_epoch = 0

        for epoch in range(n_epochs):
            # Linear warmup to base_lr across first `warmup` epochs.
            lr_scale = min(1.0, (epoch + 1) / warmup)
            for g in self.optimizer.param_groups:
                g["lr"] = base_lr * lr_scale

            self.model.train()
            perm = torch.randperm(x_t.size(0), device=device)
            epoch_loss = 0.0
            for bi in range(n_batches):
                idx = perm[bi * batch_size : (bi + 1) * batch_size]
                xb, yb = x_t[idx], y_t[idx]
                self.optimizer.zero_grad()
                out = self.model(xb)
                loss = nll_loss(out, yb)
                loss.backward()
                if self.grad_clip_norm is not None and self.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.optimizer.step()
                epoch_loss += float(loss.item())
            train_losses.append(epoch_loss / n_batches)

            if x_v is not None:
                self.model.eval()
                with torch.no_grad():
                    val_out = self.model(x_v)
                    v = float(nll_loss(val_out, y_v).item())
                if v < best_val - 1e-6:
                    best_val = v
                    best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
                    best_epoch = epoch
                    patience = 0
                else:
                    patience += 1
                    if self.early_stopping_rounds is not None and patience >= self.early_stopping_rounds:
                        break

        if best_state is not None:
            self.model.load_state_dict(best_state)
            return best_epoch + 1, train_losses

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
            "weight_decay": ("float", 1e-7, 1e-3, True),
            "warmup_epochs": ("int", 1, 20),
        }

    def get_classifier_predictions(self, x: np.ndarray | pd.DataFrame, y_true_original: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError("FT-Transformer does not support get_classifier_predictions.")

    def get_shap_explainer_info(self) -> dict[str, Any]:
        """Returns empty dict - SHAP not supported for FT-Transformer."""
        return {}
