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

`capacity_selection=CapacitySelection.PER_INPUT` (per-input width selection) is implemented via
`fit_router()` + `DistilledCapacityRouter` (`automl_package/models/common/distilled_router.py`,
capacity-programme Task FP-3): call `fit_router(x_val, y_val)` after `.fit()`, then plain
`predict(x)` -- no caller flag needed, the constructor's `capacity_selection` already says which
path to take. Only MSE (regression) and CE/BCE (classification) tasks are supported here, mirroring
`FlexibleHiddenLayersNN.build_model`'s task_type/output_size criterion-selection dispatch
(`automl_package/models/flexible_neural_network.py`); its PROBABILISTIC branch is intentionally
out of scope for this port (Task F2 spec).

`capacity_selection=CapacitySelection.GLOBAL_CHEAP` (W-SHARED, capacity-programme Task WSEL-3) is
the same two-call shape: call `fit_global_selector(x_val, y_val)` after `.fit()`, then plain
`predict(x)` -- no caller flag. It scores every configured width's held-out error on `(x_val,
y_val)` and hands that curve to the shared `cheapest_within_tolerance` selector
(`automl_package/utils/capacity_selection.py`) at twice a bootstrap standard error (`width.md`
Section 1's global-arm selection rule; MASTER Decision 18) -- picking ONE width for the whole
dataset, not per input. Do not confuse this with `PER_INPUT`: that labels every row independently
via `DistilledCapacityRouter`; this reads one held-out curve and returns a single index.
"""

from collections.abc import Callable
from typing import Any, ClassVar

import numpy as np
import torch
import torch.nn as nn

from automl_package.enums import ActivationFunction, CapacitySelection, ExplainerType, TaskType, UncertaintyMethod
from automl_package.models.base_pytorch import PyTorchModelBase
from automl_package.models.flexnn.routing import DEFAULT_TOLERANCE, DistilledCapacityRouter
from automl_package.utils.capacity_accounting import _linear_macs, executed_flops
from automl_package.utils.capacity_selection import DEFAULT_N_BOOT, cheapest_within_tolerance
from automl_package.utils.convergence import DEFAULT_PATIENCE, ConvergenceTracker
from automl_package.utils.numerics import calculate_binned_stats
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
        "capacity_selection": CapacitySelection.FIXED,
    }

    def __init__(self, **kwargs: Any) -> None:
        """Initializes the FlexibleWidthNN."""
        if "inference_mode" in kwargs:
            raise TypeError(
                "inference_mode is not a constructor parameter -- FlexibleWidthNN no longer accepts "
                "it (capacity-programme FP-3: removed from predict() too, clean break, no shim). Use "
                "capacity_selection=CapacitySelection.PER_INPUT and call fit_router() instead."
            )
        for key, value in FlexibleWidthNN._defaults.items():
            kwargs.setdefault(key, value)
        super().__init__(**kwargs)

        self.widths = tuple(sorted({int(w) for w in self.widths}))
        if not self.widths or any(w <= 0 for w in self.widths):
            raise ValueError(f"widths must be a non-empty tuple of positive integers, got {self.widths}.")

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

        The `UncertaintyMethod.CONSTANT` residual std that `PyTorchModelBase._fit_single` computes
        at the end of training is handled by this class's `_fit_residual_std` override below
        (capacity-programme FP-10), not here -- no special-casing needed in this wrapper.
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

    def _predictions_from_output(self, final_output: torch.Tensor) -> np.ndarray:
        """Shared postprocessing: raw net output -> flat predictions (classification argmax/threshold, else raw)."""
        if self.task_type == TaskType.CLASSIFICATION:
            predictions = (
                (torch.sigmoid(final_output) > _BINARY_CLASSIFICATION_THRESHOLD).cpu().numpy().astype(int)
                if self.output_size == 1
                else torch.argmax(final_output, dim=1).cpu().numpy()
            )
        else:
            predictions = final_output.cpu().numpy()
        return predictions.flatten()

    def _predict_at_explicit_width(self, x: np.ndarray, filter_data: bool, width: int) -> np.ndarray:
        """Predicts at ONE explicit width, bypassing the `capacity_selection` gate entirely.

        FP-3/FP-10 side effect, not part of the public API: several generic `BaseModel`/mixin call
        sites outside this class's write set -- this class's own `_fit_residual_std` and
        `_predict_for_scoring` overrides below, `BinnedUncertaintyMixin.calibrate_uncertainty`/
        `predict_binned_uncertainty` (`models/common/mixins.py:99,112`) -- need SOME concrete
        prediction with NO width, regardless of whether this instance is configured `FIXED` or
        `PER_INPUT` (no router exists yet during `.fit()` in either case). `predict()`'s public
        contract now correctly raises on a caller-omitted width (FP-3.b.4); this helper is what
        those internal callers use instead, always at `max(self.widths)` -- matching this class's
        pre-FP-3 implicit default for exactly this internal bookkeeping use, without weakening the
        public contract.
        """
        if filter_data:
            x = self._filter_predict_data(x)
        x_array = x.values if hasattr(x, "values") else x
        x_tensor = torch.tensor(x_array, dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            final_output = self.model.forward_width(x_tensor, width)
        return self._predictions_from_output(final_output)

    def _fit_residual_std(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """Computes the `UncertaintyMethod.CONSTANT` residual std at `max(self.widths)`.

        Overrides `PyTorchModelBase._fit_residual_std` (capacity-programme FP-10): the base
        implementation calls the public `self.predict(x_train, filter_data=False)`, which now
        raises under `CapacitySelection.FIXED` when no `width` is given (FP-3.b.4). This override
        uses `_predict_at_explicit_width` instead, at `max(self.widths)` -- matching this class's
        pre-FP-3 implicit default for this internal bookkeeping use. Replaces the earlier
        signature-introspection-based workaround that lived in `_fit_single`.
        """
        y_pred_train = self._predict_at_explicit_width(x_train, filter_data=False, width=max(self.widths))
        self._train_residual_std = np.std(np.asarray(y_train, dtype=np.float32) - y_pred_train)
        if np.isnan(self._train_residual_std):
            self._train_residual_std = 0.0

    def _predict_for_scoring(self, x: np.ndarray, filter_data: bool = True) -> np.ndarray:
        """Internal, non-caller-facing prediction path for CV/HPO/evaluation bookkeeping (FP-10).

        Overrides `BaseModel._predict_for_scoring`. Generic machinery -- `BaseModel`'s CV folds
        (`base.py:353`, `:444`), the HPO objective (`base.py:372`), `evaluate()` (`base.py:513`),
        and `utils/data_handler.py:102`'s log-scale check -- calls this polymorphically with no
        width, expecting a concrete prediction regardless of `capacity_selection`. Under
        `CapacitySelection.FIXED` the public `predict()` contract correctly raises when `width` is
        omitted (FP-3.b.4); this override predicts at `max(self.widths)` instead, via
        `_predict_at_explicit_width` -- same "internal bookkeeping, not the public API" shape as
        `_fit_residual_std` above. Under `PER_INPUT`, `predict()` needs no width, so this just
        delegates to it. Under `GLOBAL_CHEAP` (WSEL-3), this predicts at `max(self.widths)` too,
        UNCONDITIONALLY -- this bookkeeping path runs from inside `.fit()`'s own CV/HPO machinery,
        strictly before `fit_global_selector()`'s second call could have happened, so no
        `selected_width_` exists yet; matches `ProbabilisticRegressionModel._predict_for_scoring`'s
        identical ruling for its own `GLOBAL_CHEAP` case (an unselected forward, not the caller
        gate).
        """
        if self.capacity_selection in (CapacitySelection.FIXED, CapacitySelection.GLOBAL_CHEAP):
            return self._predict_at_explicit_width(x, filter_data=filter_data, width=max(self.widths))
        return self.predict(x, filter_data=filter_data)

    def predict(self, x: np.ndarray, filter_data: bool = True, width: int | None = None) -> np.ndarray:
        """Makes predictions with the FlexibleWidthNN.

        Args:
            x: Input features.
            filter_data: Whether to filter input columns to those seen at fit time.
            width: Which configured width to predict at, under `CapacitySelection.FIXED` (the
                default). Must be given explicitly -- no implicit default to the largest
                configured width. Must be omitted under `CapacitySelection.PER_INPUT` and
                `CapacitySelection.GLOBAL_CHEAP`; both choose the width themselves.

        Under `CapacitySelection.PER_INPUT`, routes with a `DistilledCapacityRouter` fitted via
        `fit_router()` -- no caller flag needed (capacity-programme Task FP-3; MASTER Decision 13).
        Under `CapacitySelection.GLOBAL_CHEAP` (W-SHARED, WSEL-3), predicts at the ONE width
        `fit_global_selector()` chose for the whole dataset -- also no caller flag needed.
        """
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")

        if self.capacity_selection == CapacitySelection.PER_INPUT:
            if width is not None:
                raise ValueError("width must not be passed under CapacitySelection.PER_INPUT: the fitted router chooses per input.")
            if getattr(self, "capacity_router_", None) is None:
                raise RuntimeError("No router fitted; call fit_router() before predict() under CapacitySelection.PER_INPUT.")
        elif self.capacity_selection == CapacitySelection.GLOBAL_CHEAP:
            if width is not None:
                raise ValueError("width must not be passed under CapacitySelection.GLOBAL_CHEAP: fit_global_selector() already chose one for the whole dataset.")
            if getattr(self, "selected_width_", None) is None:
                raise RuntimeError("No width selected; call fit_global_selector() before predict() under CapacitySelection.GLOBAL_CHEAP.")
            width = self.selected_width_
        else:
            if width is None:
                raise ValueError("width must be specified under CapacitySelection.FIXED (no implicit default to the largest configured width).")
            if width not in self.widths:
                raise ValueError(f"width={width} is not one of the configured widths {self.widths}")

        if filter_data:
            x = self._filter_predict_data(x)
        x_array = x.values if hasattr(x, "values") else x
        x_tensor = torch.tensor(x_array, dtype=torch.float32).to(self.device)

        self.model.eval()
        with torch.no_grad():
            if self.capacity_selection == CapacitySelection.PER_INPUT:
                widths_np = np.array([capacity[0] for capacity in self.capacity_router_.route(x_array)], dtype=np.int64)
                widths_tensor = torch.as_tensor(widths_np, dtype=torch.long, device=self.device)
                final_output = self.model.forward_at_widths(x_tensor, widths_tensor)
            else:
                final_output = self.model.forward_width(x_tensor, width)
        return self._predictions_from_output(final_output)

    def _per_sample_error_at_width(self, x: np.ndarray, y: np.ndarray, width: int) -> np.ndarray:
        """Per-sample held-out error at ONE width: squared error (regression) or 0/1 error (classification).

        Shared scoring primitive behind both selection mechanisms this class exposes -- `fit_router`'s
        per-input `eval_fn` (`CapacitySelection.PER_INPUT`) and `fit_global_selector`'s held-out error
        curve (`CapacitySelection.GLOBAL_CHEAP`, WSEL-3) -- so the two selectors agree on what "error
        at a width" means and neither re-derives it.

        Args:
            x: `(N, in_dim)` inputs.
            y: `(N,)` targets, already `np.asarray`'d by the caller.
            width: one of `self.widths`.

        Returns:
            `(N,)` per-sample error at `width`, lower is better.
        """
        x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device)
        self.model.eval()
        with torch.no_grad():
            raw_output = self.model.forward_width(x_tensor, width)
        if self.task_type == TaskType.CLASSIFICATION:
            if self.output_size == 1:
                pred = (torch.sigmoid(raw_output) > _BINARY_CLASSIFICATION_THRESHOLD).float().squeeze(1).cpu().numpy()
            else:
                pred = torch.argmax(raw_output, dim=1).float().cpu().numpy()
            return (pred != y).astype(np.float64)
        return (raw_output.squeeze(1).cpu().numpy() - y) ** 2

    def fit_router(
        self,
        x_val: np.ndarray,
        y_val: np.ndarray,
        capacity_grid: list[tuple[int, ...]] | None = None,
        tolerance: float = DEFAULT_TOLERANCE,
        cost_fn: Callable[[tuple[int, ...]], float] | None = None,
    ) -> DistilledCapacityRouter:
        """Fits a `DistilledCapacityRouter` post-hoc for `predict()` under `CapacitySelection.PER_INPUT`.

        Same pattern as `FlexibleHiddenLayersNN.fit_router` (capacity-programme Task F3): `eval_fn`
        forces every sample through a fixed width via `_per_sample_error_at_width` and scores it
        against `y_val` with squared error (regression) or 0/1 error (classification).

        Args:
            x_val: held-out inputs.
            y_val: held-out targets.
            capacity_grid: candidate widths as 1-tuples, e.g. `[(16,), (32,)]`. Defaults to every
                configured width in `self.widths`.
            tolerance: cheapest-within-tolerance labeling tolerance (see `DistilledCapacityRouter`).
            cost_fn: `cost_fn(capacity) -> float`. Defaults to `automl_package.utils.capacity_accounting
                .executed_flops` at the routed width, dispatched on `FlexibleWidthNNModule` via the
                registration at the bottom of this file.

        Returns:
            The fitted `DistilledCapacityRouter` (also stored as `self.capacity_router_`).
        """
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        if capacity_grid is None:
            capacity_grid = [(width,) for width in self.widths]

        y_val_arr = np.asarray(y_val)

        def eval_fn(x: np.ndarray, capacity: tuple[int, ...]) -> np.ndarray:
            return self._per_sample_error_at_width(x, y_val_arr, capacity[0])

        if cost_fn is None:

            def cost_fn(capacity: tuple[int, ...]) -> float:
                return float(executed_flops(self.model, capacity[0]))

        router = DistilledCapacityRouter(device=self.device)
        router.fit(eval_fn=eval_fn, x_val=x_val, y_val=y_val, capacity_grid=capacity_grid, tolerance=tolerance, cost_fn=cost_fn)
        self.capacity_router_ = router
        return router

    def fit_global_selector(
        self,
        x_val: np.ndarray,
        y_val: np.ndarray,
        n_boot: int = DEFAULT_N_BOOT,
        seed: int | None = None,
    ) -> int:
        """Picks ONE width for the whole dataset (W-SHARED, capacity-programme Task WSEL-3).

        Scores every configured width on the held-out `(x_val, y_val)` selection set via
        `_per_sample_error_at_width` (the same scoring primitive `fit_router` uses), then feeds the
        resulting `(N, len(self.widths))` held-out error curve to the shared
        `automl_package.utils.capacity_selection.cheapest_within_tolerance` selector -- the smallest
        width whose held-out error is not meaningfully worse than the best width's, "meaningfully" =
        exceeding twice a bootstrap-estimated standard error (`width.md` Section 1's global-arm
        selection rule; MASTER Decision 18). Neither the tolerance rule nor its bootstrap
        standard-error estimator is re-derived here -- both are imported from FP-9's shared module,
        per WSEL-3's own doctrine.

        Stores the chosen width as `self.selected_width_`; `predict()` then uses it with no caller
        flag under `CapacitySelection.GLOBAL_CHEAP`, mirroring `fit_router()`'s two-call
        `CapacitySelection.PER_INPUT` pattern -- call this AFTER `.fit()`, on a selection set
        disjoint from what `.fit()` trained on.

        Args:
            x_val: held-out inputs, disjoint from `fit()`'s training data.
            y_val: held-out targets.
            n_boot: bootstrap resamples for the selection rule's paired-difference standard-error
                estimate (see `cheapest_within_tolerance`).
            seed: RNG seed for the bootstrap resample -- determinism. Defaults to `self.random_seed`
                (or `0` if unset), matching `ProbabilisticRegressionModel.fit_global_selector`'s
                convention for the same selection rule.

        Returns:
            The selected width (also stored as `self.selected_width_`, and the full curve as
            `self.global_selector_curve_`).

        Raises:
            RuntimeError: the model has not been fitted yet.
        """
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")

        x_arr = x_val.values if hasattr(x_val, "values") else np.asarray(x_val)
        y_val_arr = np.asarray(y_val)
        error_table = np.stack([self._per_sample_error_at_width(x_arr, y_val_arr, width) for width in self.widths], axis=1)

        resolved_seed = seed if seed is not None else (self.random_seed or 0)
        idx = cheapest_within_tolerance(error_table, n_boot=n_boot, seed=resolved_seed)
        selected_width = self.widths[idx]

        self.selected_width_ = selected_width
        self.global_selector_curve_ = {
            "widths": list(self.widths),
            "mean_error": error_table.mean(axis=0).tolist(),
            "n_selection": len(y_val_arr),
            "selected_width": selected_width,
        }
        return selected_width

    def predict_uncertainty(self, x: np.ndarray, filter_data: bool = True) -> np.ndarray:
        """Estimates prediction uncertainty for regression (fixed width, WD1).

        `CONSTANT` and `BINNED_RESIDUAL_STD` delegate straight to the inherited
        (`PyTorchModelBase` / `BaseModel`) implementation: `CONSTANT` only reads
        `self._train_residual_std`, and `BINNED_RESIDUAL_STD` calls
        `self.predict_binned_uncertainty()` (overridden below, alongside
        `calibrate_uncertainty()`, to use `_predict_at_explicit_width` instead of the mixin's
        generic `self.predict(x)` -- same FP-3 reason as `_fit_single`'s residual-std fix above)
        -- neither ever touches `FlexibleWidthNNModule.forward`'s raw
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

    def calibrate_uncertainty(self, x: np.ndarray, y: np.ndarray, n_bins: int = 10) -> None:
        """Calibrates `BINNED_RESIDUAL_STD` uncertainty (`BinnedUncertaintyMixin`, `models/common/mixins.py:97`).

        Overridden ONLY to swap the mixin's generic `self.predict(x)` (no width -- would raise
        under `CapacitySelection.FIXED` per FP-3.b.4) for `_predict_at_explicit_width` at
        `max(self.widths)`. Logic otherwise identical to the mixin's.
        """
        predictions = self._predict_at_explicit_width(x, filter_data=True, width=max(self.widths))
        residuals = y.flatten() - predictions.flatten()
        bin_edges, stats = calculate_binned_stats(probas=predictions, values=residuals, n_bins=n_bins, aggregations={"std": np.std}, min_value=-np.inf, max_value=np.inf)
        self._binned_uncertainty_edges = bin_edges
        self._binned_uncertainty_lookup = stats["std"]

    def predict_binned_uncertainty(self, x: np.ndarray) -> np.ndarray:
        """Looks up calibrated `BINNED_RESIDUAL_STD` uncertainty (`models/common/mixins.py:107`).

        Overridden ONLY to swap the mixin's generic `self.predict(x)` for
        `_predict_at_explicit_width` at `max(self.widths)` -- same reason as `calibrate_uncertainty`
        above; logic otherwise identical to the mixin's.
        """
        if self._binned_uncertainty_edges is None or self._binned_uncertainty_lookup is None or len(self._binned_uncertainty_lookup) == 0:
            raise RuntimeError("Uncertainty model has not been calibrated. Call `calibrate_uncertainty` first.")
        predictions = self._predict_at_explicit_width(x, filter_data=True, width=max(self.widths))
        bin_indices = np.searchsorted(self._binned_uncertainty_edges[1:], predictions.flatten(), side="left")
        bin_indices = np.clip(bin_indices, 0, len(self._binned_uncertainty_edges) - 2)
        stds = self._binned_uncertainty_lookup[bin_indices]
        fallback = float(np.nanmean(self._binned_uncertainty_lookup)) if np.any(~np.isnan(self._binned_uncertainty_lookup)) else 0.0
        return np.nan_to_num(stds, nan=fallback)

    def get_params(self) -> dict[str, Any]:
        """Gets the parameters of the model."""
        params = super().get_params()
        params.update(
            {
                "widths": self.widths,
                "activation": self.activation,
                "capacity_selection": self.capacity_selection,
                "trustworthy": getattr(self, "convergence_summary_", {}).get("trustworthy", False),
            }
        )
        return params

    def get_shap_explainer_info(self) -> dict[str, Any]:
        """Gets the SHAP explainer type and the model to be explained (at `w_max`, via `forward`)."""
        return {"explainer_type": ExplainerType.DEEP, "model": self.get_internal_model()}


# ---------------------------------------------------------------------------
# executed_flops registration (capacity-programme Task FP-1) -- self-registered here, not from
# `automl_package/utils/capacity_accounting.py`, to avoid a circular import: that module would
# otherwise need to import this file's class while this file imports its `executed_flops`.
# ---------------------------------------------------------------------------


@executed_flops.register(FlexibleWidthNN.FlexibleWidthNNModule)
def _executed_flops_flexible_width_module(net: FlexibleWidthNN.FlexibleWidthNNModule, config: int) -> int:
    """Routed width-`config` MACs for `FlexibleWidthNN`.

    Same shared-trunk-prefix + per-width-head shape and slicing argument as
    `nested_width_net.SharedTrunkPerWidthHeadNet` (`FlexibleWidthNN`'s module docstring states this
    class ports that exact architecture) -- reuses the identical formula shape with this class's
    own attribute names (`trunk_linear`, `heads[str(w)]`).
    """
    w = config
    if w not in net.widths:
        raise ValueError(f"config={w} is not one of the configured widths {net.widths}")
    head = net.heads[str(w)]
    return _linear_macs(net.trunk_linear.in_features, w) + _linear_macs(w, head.out_features)
