"""Width-flexible neural network: shared trunk, prefix-masked per-input width, per-width heads.

Ports the G-WIDTH certified `SharedTrunkPerWidthHeadNet` architecture
(`automl_package/examples/nested_width_net.py:222`) into the package's `PyTorchModelBase` model
family: ONE shared trunk `Linear(input_size, w_max)` (`w_max = max(widths)`), a per-width prefix
mask (`hidden[:, w:] = 0`) applied before EACH width's OWN dedicated readout head
`Linear(w_max, output_size)`. `docs/width_mse_2026-07-16/verdict_variable_width_mse.md` certified
this specific arm (shared trunk + per-width heads) reaches the per-region noise floor on every
seed, unlike a fully-shared readout (`NestedWidthNet`), which plateaus 3.7-5.9x above floor --
localizing that earlier failure to the shared READOUT, not the shared trunk.

Training: sum of per-width losses every step (one optimizer step scores every configured width) --
the certified joint regime from
`automl_package/examples/kdropout_converged_width_experiment.py::_train_kdropout_to_convergence`'s
SANDWICH-schedule inner loop (`total_loss = total_loss + _width_loss(...)`), specialised here to
sum ALL configured widths every step rather than a sampled subset (this model's `widths` is a
small fixed tuple, e.g. 4 values, not the 1..w_max per-node granularity that loop subsamples via
the SANDWICH schedule). Wired into `PyTorchModelBase`'s existing generic `_fit_single` via a
`multi_width_loss` criterion wrapper (see `build_model`/`_multi_width_criterion`) rather than an
overridden training loop, so batching / early stopping / regularization are reused unmodified.

`inference_mode="routed"` (per-input width selection) is implemented via `fit_router()` +
`DistilledCapacityRouter` (`automl_package/models/common/distilled_router.py`, capacity-programme
Task F3): call `fit_router(x_val, y_val)` after `.fit()`, then `predict(x,
inference_mode="routed")`. `width_selection_method=DISTILLED` is a separate, still-unwired flag
that continues to raise `NotImplementedError` at construction (see
`TestFlexibleWidthNNValidation` in `tests/test_flexible_width_network.py`) -- routing does not
require it. Only MSE (regression) and CE/BCE (classification) tasks are supported here, mirroring
`FlexibleHiddenLayersNN.build_model`'s task_type/output_size criterion-selection dispatch
(`automl_package/models/flexible_neural_network.py`); its PROBABILISTIC branch is intentionally
out of scope for this port (Task F2 spec).
"""

from collections.abc import Callable
from typing import Any, ClassVar

import numpy as np
import torch
import torch.nn as nn

from automl_package.enums import ActivationFunction, ExplainerType, TaskType, UncertaintyMethod, WidthSelectionMethod
from automl_package.models.base_pytorch import PyTorchModelBase
from automl_package.models.common.distilled_router import DEFAULT_TOLERANCE, DistilledCapacityRouter
from automl_package.utils.convergence import DEFAULT_PATIENCE, ConvergenceTracker
from automl_package.utils.pytorch_utils import get_activation_function_map

_BINARY_CLASSIFICATION_THRESHOLD = 0.5


def _multi_width_criterion(base_criterion: nn.Module) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Wraps a per-width loss into the certified sum-over-widths training objective (see module docstring).

    `stacked_outputs` is `(len(widths), N, output_size)` -- `FlexibleWidthNNModule.forward`'s
    output -- so this single callable, handed to `PyTorchModelBase._fit_single` as `self.criterion`,
    reproduces "one optimizer step scores every width" without that loop needing to know widths exist.
    """

    def multi_width_loss(stacked_outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        total = stacked_outputs.new_zeros(())
        for width_outputs in stacked_outputs:
            total = total + base_criterion(width_outputs, targets)
        return total

    return multi_width_loss


class FlexibleWidthNN(PyTorchModelBase):
    """A PyTorch neural network with per-input selectable width (active hidden units).

    A single shared trunk maps input to a `w_max`-dimensional hidden representation; each
    configured width `w` reads that representation through its OWN dedicated head after zeroing
    hidden units `w:` (the prefix mask). See module docstring for the certified architecture and
    training-regime provenance.
    """

    _defaults: ClassVar[dict[str, Any]] = {
        "widths": (16, 32, 48, 64),
        "activation": ActivationFunction.RELU,
        "width_selection_method": WidthSelectionMethod.NONE,
    }

    def __init__(self, **kwargs: Any) -> None:
        """Initializes the FlexibleWidthNN."""
        for key, value in FlexibleWidthNN._defaults.items():
            kwargs.setdefault(key, value)
        super().__init__(**kwargs)

        self.widths = tuple(sorted({int(w) for w in self.widths}))
        if not self.widths or any(w <= 0 for w in self.widths):
            raise ValueError(f"widths must be a non-empty tuple of positive integers, got {self.widths}.")
        if self.width_selection_method == WidthSelectionMethod.DISTILLED:
            raise NotImplementedError(
                "width_selection_method=WidthSelectionMethod.DISTILLED requires DistilledCapacityRouter "
                "(task F3, docs/plans/capacity_programme/flexnn-core.md), which has not landed yet. "
                "Use WidthSelectionMethod.NONE and call predict(x, width=...) with an explicit width."
            )

    @property
    def name(self) -> str:
        """Returns the name of the model."""
        return "FlexibleWidthNN"

    class FlexibleWidthNNModule(nn.Module):
        """Internal PyTorch nn.Module: shared trunk + per-width prefix-masked heads.

        For width `w`, `forward_width` zeros the trunk's hidden units `w:` before applying head
        `w`'s own `Linear(w_max, output_size)` -- the SAME prefix-masking mechanism as
        `nested_width_net.SharedTrunkPerWidthHeadNet.forward_width`, generalized to arbitrary
        `input_size`/`output_size` and an explicit `widths` tuple rather than the 1..w_max node
        granularity of the research script.
        """

        def __init__(self, outer_instance: "FlexibleWidthNN") -> None:
            """Builds the shared trunk plus one dedicated head per configured width."""
            super().__init__()
            self.widths = outer_instance.widths
            self.w_max = max(self.widths)
            self.trunk_linear = nn.Linear(outer_instance.input_size, self.w_max)
            self.activation = outer_instance.activation()
            self.heads = nn.ModuleDict({str(w): nn.Linear(self.w_max, outer_instance.output_size) for w in self.widths})

        def hidden(self, x: torch.Tensor) -> torch.Tensor:
            """`(N, input_size) -> (N, w_max)` post-activation trunk representation, shared across all widths."""
            return self.activation(self.trunk_linear(x))

        def forward_width(self, x: torch.Tensor, width: int) -> torch.Tensor:
            """Prediction at one fixed configured width: prefix-mask the trunk, then width's own head."""
            key = str(width)
            if key not in self.heads:
                raise ValueError(f"width={width} is not one of the configured widths {self.widths}")
            h = self.hidden(x)
            mask = torch.zeros_like(h)
            mask[:, :width] = 1.0
            return self.heads[key](h * mask)

        def forward_at_widths(self, x: torch.Tensor, widths: torch.Tensor) -> torch.Tensor:
            """Per-sample forced width (e.g. routed by a `DistilledCapacityRouter`), bucketed by width.

            `widths` is a `(N,)` LongTensor whose values are each one of `self.widths`.
            """
            out_features = next(iter(self.heads.values())).out_features
            result = torch.zeros(x.size(0), out_features, device=x.device, dtype=x.dtype)
            for width in torch.unique(widths).tolist():
                mask = widths == width
                result[mask] = self.forward_width(x[mask], width)
            return result

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Every configured width's output, stacked: `(len(widths), N, output_size)`.

            Paired with `_multi_width_criterion`'s wrapper, this lets the generic
            `PyTorchModelBase._fit_single` (unmodified) train ALL widths every step via one
            `self.model(batch_x)` / `self.criterion(outputs, batch_y)` call.
            """
            return torch.stack([self.forward_width(x, w) for w in self.widths], dim=0)

    def _fit_single(self, *args: Any, **kwargs: Any) -> tuple[int, list[float]]:
        """Wraps the generic joint-width training loop to expose `convergence_summary_`.

        The base loop trains every configured width jointly each step (sum-over-widths criterion,
        see module docstring) and returns ONE val-loss trajectory -- the SUMMED loss across all
        widths, not a per-width breakdown. A genuine per-width trajectory would require overriding
        the shared `PyTorchModelBase._fit_single` loop itself, out of scope here, so this is a single
        COMBINED convergence summary of the joint trajectory, not one per width.
        """
        best_epoch, val_loss_history = super()._fit_single(*args, **kwargs)
        patience = self.early_stopping_rounds if self.early_stopping_rounds is not None else DEFAULT_PATIENCE
        tracker = ConvergenceTracker(patience=patience, min_delta=0.0)
        for i, v in enumerate(val_loss_history, start=1):
            tracker.update(i, v)
        self.convergence_summary_ = tracker.result(final_epoch=len(val_loss_history)).summary()
        return best_epoch, val_loss_history

    def build_model(self) -> None:
        """Builds the internal PyTorch nn.Module and the sum-over-widths criterion."""
        if isinstance(self.activation, ActivationFunction):
            activation_function_map = get_activation_function_map()
            self.activation = activation_function_map.get(self.activation)
            if self.activation is None:
                raise ValueError(f"Unsupported activation function: {self.activation}")

        self.model = self.FlexibleWidthNNModule(self).to(self.device)

        if self.task_type == TaskType.REGRESSION:
            base_criterion: nn.Module = nn.MSELoss()
        elif self.task_type == TaskType.CLASSIFICATION:
            base_criterion = nn.BCEWithLogitsLoss() if self.output_size == 1 else nn.CrossEntropyLoss()
        else:
            raise ValueError("task_type must be 'regression' or 'classification'")
        self.criterion = _multi_width_criterion(base_criterion)

    def predict(self, x: np.ndarray, filter_data: bool = True, width: int | None = None, inference_mode: str = "fixed") -> np.ndarray:
        """Makes predictions with the FlexibleWidthNN.

        Args:
            x: Input features.
            filter_data: Whether to filter input columns to those seen at fit time.
            width: Which configured width to predict at (`inference_mode="fixed"` only). Defaults
                to the largest configured width (`max(self.widths)`) when omitted.
            inference_mode: `"fixed"` (default) predicts at `width` via that width's own head.
                `"routed"` uses a `DistilledCapacityRouter` fitted via `fit_router()` to pick a
                per-sample width post-hoc (capacity-programme Task F3; MASTER Decision 13).
        """
        if inference_mode not in ("fixed", "routed"):
            raise ValueError(f"Unknown inference_mode: {inference_mode!r}")
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        if inference_mode == "routed" and getattr(self, "capacity_router_", None) is None:
            raise RuntimeError("No router fitted; call fit_router() before predict(inference_mode='routed').")

        if inference_mode == "fixed":
            width = max(self.widths) if width is None else width
            if width not in self.widths:
                raise ValueError(f"width={width} is not one of the configured widths {self.widths}")

        if filter_data:
            x = self._filter_predict_data(x)
        x_array = x.values if hasattr(x, "values") else x
        x_tensor = torch.tensor(x_array, dtype=torch.float32).to(self.device)

        self.model.eval()
        with torch.no_grad():
            if inference_mode == "routed":
                widths_np = np.array([capacity[0] for capacity in self.capacity_router_.route(x_array)], dtype=np.int64)
                widths_tensor = torch.as_tensor(widths_np, dtype=torch.long, device=self.device)
                final_output = self.model.forward_at_widths(x_tensor, widths_tensor)
            else:
                final_output = self.model.forward_width(x_tensor, width)
            if self.task_type == TaskType.CLASSIFICATION:
                predictions = (
                    (torch.sigmoid(final_output) > _BINARY_CLASSIFICATION_THRESHOLD).cpu().numpy().astype(int)
                    if self.output_size == 1
                    else torch.argmax(final_output, dim=1).cpu().numpy()
                )
            else:
                predictions = final_output.cpu().numpy()
        return predictions.flatten()

    def fit_router(
        self,
        x_val: np.ndarray,
        y_val: np.ndarray,
        capacity_grid: list[tuple[int, ...]] | None = None,
        tolerance: float = DEFAULT_TOLERANCE,
        cost_fn: Callable[[tuple[int, ...]], float] | None = None,
    ) -> DistilledCapacityRouter:
        """Fits a `DistilledCapacityRouter` post-hoc for `predict(..., inference_mode="routed")`.

        Same pattern as `FlexibleHiddenLayersNN.fit_router` (capacity-programme Task F3): `eval_fn`
        forces every sample through a fixed width via `FlexibleWidthNNModule.forward_width` and
        scores it against `y_val` with squared error (regression) or 0/1 error (classification).

        Args:
            x_val: held-out inputs.
            y_val: held-out targets.
            capacity_grid: candidate widths as 1-tuples, e.g. `[(16,), (32,)]`. Defaults to every
                configured width in `self.widths`.
            tolerance: cheapest-within-tolerance labeling tolerance (see `DistilledCapacityRouter`).
            cost_fn: `cost_fn(capacity) -> float`. Defaults to `executed_flops` from
                `automl_package/examples/capacity_accounting.py` (S2 accounting) at the routed
                width. Imported inside this method, not at module top -- see
                `FlexibleHiddenLayersNN.fit_router` for why (the same load-time circular import
                this class would otherwise create with `capacity_accounting.py`).

        Returns:
            The fitted `DistilledCapacityRouter` (also stored as `self.capacity_router_`).
        """
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        if capacity_grid is None:
            capacity_grid = [(width,) for width in self.widths]

        y_val_arr = np.asarray(y_val)

        def eval_fn(x: np.ndarray, capacity: tuple[int, ...]) -> np.ndarray:
            width = capacity[0]
            x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device)
            self.model.eval()
            with torch.no_grad():
                raw_output = self.model.forward_width(x_tensor, width)
            if self.task_type == TaskType.CLASSIFICATION:
                if self.output_size == 1:
                    pred = (torch.sigmoid(raw_output) > _BINARY_CLASSIFICATION_THRESHOLD).float().squeeze(1).cpu().numpy()
                else:
                    pred = torch.argmax(raw_output, dim=1).float().cpu().numpy()
                return (pred != y_val_arr).astype(np.float64)
            return (raw_output.squeeze(1).cpu().numpy() - y_val_arr) ** 2

        if cost_fn is None:
            from automl_package.examples.capacity_accounting import executed_flops  # noqa: PLC0415, I001 -- avoids a load-time circular import (capacity_accounting.py imports this class)

            def cost_fn(capacity: tuple[int, ...]) -> float:
                return float(executed_flops(self.model, capacity[0]))

        router = DistilledCapacityRouter(device=self.device)
        router.fit(eval_fn=eval_fn, x_val=x_val, y_val=y_val, capacity_grid=capacity_grid, tolerance=tolerance, cost_fn=cost_fn)
        self.capacity_router_ = router
        return router

    def predict_uncertainty(self, x: np.ndarray, filter_data: bool = True) -> np.ndarray:
        """Estimates prediction uncertainty for regression (fixed width, WD1).

        `CONSTANT` and `BINNED_RESIDUAL_STD` delegate straight to the inherited
        (`PyTorchModelBase` / `BaseModel`) implementation: `CONSTANT` only reads
        `self._train_residual_std`, and `BINNED_RESIDUAL_STD` only calls `self.predict()`
        (overridden above) -- neither ever touches `FlexibleWidthNNModule.forward`'s raw
        `(len(widths), N, output_size)` stacked-over-widths output, so nothing about this
        architecture breaks them.

        `MC_DROPOUT` and `PROBABILISTIC` raise explicitly instead of silently mis-indexing
        that stacked tensor (the WD1 defect): `FlexibleWidthNNModule` has no dropout layer
        (`dropout_rate` is never wired into it, unlike `FlexibleHiddenLayersNN`), so MC
        sampling would be deterministic and silently report zero uncertainty; and its heads
        emit only the raw regression target, never a `(mean, log_variance)` pair, so there is
        no variance to read out for `PROBABILISTIC` (see module docstring: that branch is out
        of scope for this port).
        """
        if not self.is_regression_model:
            raise ValueError("predict_uncertainty is only available for regression models.")
        if self.uncertainty_method == UncertaintyMethod.MC_DROPOUT:
            raise NotImplementedError(
                "uncertainty_method=UncertaintyMethod.MC_DROPOUT is not supported by FlexibleWidthNN: "
                "FlexibleWidthNNModule has no dropout layer, so MC sampling would be deterministic and "
                "silently report zero uncertainty. Use CONSTANT or BINNED_RESIDUAL_STD instead."
            )
        if self.uncertainty_method == UncertaintyMethod.PROBABILISTIC:
            raise NotImplementedError(
                "uncertainty_method=UncertaintyMethod.PROBABILISTIC is not supported by FlexibleWidthNN: "
                "its heads emit only the raw regression target, never a (mean, log_variance) pair (see "
                "module docstring -- the PROBABILISTIC branch is out of scope for this port). Use CONSTANT "
                "or BINNED_RESIDUAL_STD instead."
            )
        return super().predict_uncertainty(x, filter_data=filter_data)

    def get_params(self) -> dict[str, Any]:
        """Gets the parameters of the model."""
        params = super().get_params()
        params.update(
            {
                "widths": self.widths,
                "activation": self.activation,
                "width_selection_method": self.width_selection_method,
                "trustworthy": getattr(self, "convergence_summary_", {}).get("trustworthy", False),
            }
        )
        return params

    def get_shap_explainer_info(self) -> dict[str, Any]:
        """Gets the SHAP explainer type and the model to be explained (at `w_max`, via `forward`)."""
        return {"explainer_type": ExplainerType.DEEP, "model": self.get_internal_model()}
