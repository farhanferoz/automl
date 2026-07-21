"""Probabilistic Regression model implemented in PyTorch."""

from collections.abc import Callable
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from automl_package.enums import (
    BoundaryRegularizationMethod,
    CapacitySelection,
    ExplainerType,
    NClassesRegularization,
    NClassesSelectionMethod,
    ProbabilisticRegressionOptimizationStrategy,
    ProbRegLossType,
    RegressionStrategy,
    UncertaintyMethod,
)
from automl_package.logger import logger
from automl_package.models.architectures.probabilistic_regression_net import ProbabilisticRegressionNet
from automl_package.models.base_pytorch import PyTorchModelBase
from automl_package.models.common.distilled_router import DEFAULT_TOLERANCE, DistilledCapacityRouter
from automl_package.models.common.losses import calculate_combined_loss
from automl_package.models.common.middle_class_penalty_mixin import MiddleClassPenaltyMixin
from automl_package.models.common.mixins import BoundaryLossMixin
from automl_package.models.common.penalties import apply_additional_penalties
from automl_package.models.selection_strategies.base_selection_strategy import DIRECT_REGRESSION_K_SENTINEL
from automl_package.utils.capacity_selection import cheapest_within_tolerance
from automl_package.utils.data_handler import create_train_val_split
from automl_package.utils.distributions import MixtureOfGaussiansDistribution
from automl_package.utils.losses import masked_cross_entropy_loss, mdn_nll
from automl_package.utils.numerics import calculate_class_value_ranges, create_bins
from automl_package.utils.ordering_loss import ordering_loss as ordering_loss_fn
from automl_package.utils.plotting import plot_nn_probability_mappers
from automl_package.utils.transforms import symexp, symlog


class ProbabilisticRegressionModel(PyTorchModelBase, BoundaryLossMixin, MiddleClassPenaltyMixin):
    """A PyTorch-based probabilistic regression model that directly learns both mean and variance.

    ProbReg is a CLASSIFIER over k classes with per-class (mean, log_var) regression heads,
    combined via the law of total variance -- not a Gaussian mixture model in the
    latent-component sense.

    Recommended dynamic-k path (capacity-programme Task F9, MASTER Decision 13 -- selection is
    DISTILLED, never in-training): train with `n_classes_selection_method=NESTED` (per-sample k
    drawn as a training schedule, ported from `_capacity_ladder_nested.py`), then construct with
    `capacity_selection=CapacitySelection.PER_INPUT`, call `fit_router()`, and `predict(x)` routes
    per-input k with no caller flag (capacity-programme Task FP-3). The other dynamic strategies --
    SOFT_GATING, GUMBEL_SOFTMAX, STE, REINFORCE, and `NClassesRegularization` (K_PENALTY/ELBO) --
    remain fully functional but are labeled COMPARISON ARMS, not the recommended path.

    Warning:
        Dynamic n_classes selection (`n_classes_selection_method != NONE`) combined with an
        optimization_strategy that activates classification cross-entropy (anything other than
        REGRESSION_ONLY or GRADIENT_STOP) is NOT validated: class boundaries are precomputed
        independently per k, so per-k re-binned CE targets redefine class identity across k
        (a node-0 conflict). The validated recipe of record is dynamic n_classes selection with
        `optimization_strategy=REGRESSION_ONLY` (see
        docs/plans/width_dial_2026-07-11/cascade_execution_plan_2026-07-11.md §3.1).
    """

    _defaults: ClassVar[dict[str, Any]] = {
        "input_size": None,
        "n_classes": 3,
        "n_classes_inf": float("inf"),
        "max_n_classes_for_probabilistic_path": 10,
        "base_classifier_params": None,
        "regression_head_params": None,
        "direct_regression_head_params": None,
        "regression_strategy": RegressionStrategy.SEPARATE_HEADS,
        "n_classes_selection_method": NClassesSelectionMethod.NONE,
        "gumbel_tau": 0.5,
        "n_classes_predictor_learning_rate": 0.001,
        "optimization_strategy": ProbabilisticRegressionOptimizationStrategy.REGRESSION_ONLY,
        "boundary_regularization_method": BoundaryRegularizationMethod.NONE,
        "boundary_loss_weight": 1.0,
        "use_monotonic_constraints": False,
        "constrain_middle_class": True,
        "use_middle_class_nll_penalty": False,
        "n_classes_regularization": NClassesRegularization.NONE,
        "k_penalty_weight": 0.01,
        # ELBO prior configuration for n_classes_regularization == ELBO.
        # bypass_prior_prob: prior probability p(bypass) for the direct-regression mode,
        #   applied as a separate Bernoulli KL. Default 0.5 (symmetric / least-informative).
        # k_prior_type: prior over probabilistic modes k ∈ {2..k_max}, conditional on not-bypass.
        #   "uniform" (default) — pure concentration penalty.
        #   "geometric" — prefers low k; step size controlled by k_prior_geometric_lambda.
        "bypass_prior_prob": 0.5,
        "k_prior_type": "uniform",
        "k_prior_geometric_lambda": 0.2,
        "loss_type": "nll",
        "beta": 0.5,
        "target_transform": None,
        "prob_reg_loss_type": ProbRegLossType.GAUSSIAN_LTV,
        "use_anchored_heads": False,
        # Ordering-constraint identifiability (see docs/probreg_identifiability_research.md §3.3, §7.7).
        # None = auto: enable (weight=1.0) only for the (SEPARATE_HEADS, Gaussian-LTV, REGRESSION_ONLY)
        # triple — the single configuration where the penalty measurably helps. Redundant under
        # CE_STOP_GRAD, harmful under MDN; auto-resolution sets it to 0 there. Pass a float to
        # override (e.g. 0.0 to disable everywhere).
        "ordering_constraint_weight": None,
        "ordering_constraint_margin": 0.0,
        "ordering_top_decile_fraction": 0.1,
        "capacity_selection": CapacitySelection.FIXED,
        # Fraction of fit()'s (x, y) held out for CapacitySelection.GLOBAL_CHEAP/GLOBAL_SWEEP's
        # internal k-selection step (capacity-programme FP-3.e: a constructor parameter, never a
        # baked-in constant -- PA's task text, PB measures the right value across
        # {5, 10, 15, 25, 40}%). Unused under FIXED/PER_INPUT.
        "selection_fraction": 0.15,
    }

    def __init__(self, **kwargs: Any) -> None:
        """Initializes the ProbabilisticRegressionModel."""
        if "inference_mode" in kwargs:
            raise TypeError(
                "inference_mode is not a constructor parameter -- ProbabilisticRegressionModel no "
                "longer accepts it (capacity-programme FP-3: removed from predict()/"
                "predict_uncertainty() too, clean break, no shim). Use "
                "capacity_selection=CapacitySelection.PER_INPUT and call fit_router() instead."
            )
        for key, value in self._defaults.items():
            kwargs.setdefault(key, value)

        if kwargs.get("boundary_regularization_method") == BoundaryRegularizationMethod.PENALTY and kwargs.get("uncertainty_method") == UncertaintyMethod.PROBABILISTIC:
            raise ValueError(
                "The PENALTY boundary method is not compatible with the PROBABILISTIC uncertainty method from a pure statistical perspective. Use the HARDSIGMOID method instead."
            )

        if (
            kwargs.get("optimization_strategy") != ProbabilisticRegressionOptimizationStrategy.REGRESSION_ONLY
            and kwargs.get("uncertainty_method") != UncertaintyMethod.PROBABILISTIC
        ):
            logger.warning(
                f"Selected optimization_strategy requires a probabilistic uncertainty method. "
                f"Overriding uncertainty_method from {kwargs.get('uncertainty_method').value} to {UncertaintyMethod.PROBABILISTIC.value}."
            )
            kwargs["uncertainty_method"] = UncertaintyMethod.PROBABILISTIC

        output_size = 2 if kwargs.get("uncertainty_method") == UncertaintyMethod.PROBABILISTIC else 1
        kwargs["output_size"] = output_size

        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)

        if isinstance(self.regression_strategy, str):
            self.regression_strategy = RegressionStrategy[self.regression_strategy.upper()]
        if isinstance(self.n_classes_selection_method, str):
            self.n_classes_selection_method = NClassesSelectionMethod[self.n_classes_selection_method.upper()]
        if isinstance(self.optimization_strategy, str):
            self.optimization_strategy = ProbabilisticRegressionOptimizationStrategy[self.optimization_strategy.upper()]
        if isinstance(self.n_classes_regularization, str):
            self.n_classes_regularization = NClassesRegularization[self.n_classes_regularization.upper()]
        if isinstance(self.prob_reg_loss_type, str):
            self.prob_reg_loss_type = ProbRegLossType[self.prob_reg_loss_type.upper()]
        if isinstance(self.capacity_selection, str):
            self.capacity_selection = CapacitySelection[self.capacity_selection.upper()]

        if self.n_classes_selection_method == NClassesSelectionMethod.NESTED and self.uncertainty_method != UncertaintyMethod.PROBABILISTIC:
            raise ValueError(
                "n_classes_selection_method=NESTED requires uncertainty_method=PROBABILISTIC: the nested-k "
                "scheme is a strictly-probabilistic Gaussian-NLL training objective "
                "(automl_package/examples/_capacity_ladder_nested.py), not defined for constant/MC-dropout heads."
            )

        if self.use_monotonic_constraints and self.regression_strategy != RegressionStrategy.SEPARATE_HEADS:
            logger.warning("Monotonic constraints are only supported for the 'SEPARATE_HEADS' regression strategy.")
            self.use_monotonic_constraints = False

        if self.use_anchored_heads and self.regression_strategy != RegressionStrategy.SEPARATE_HEADS:
            logger.warning("use_anchored_heads=True has no effect for regression_strategy=%s; anchoring is only defined for SEPARATE_HEADS. Ignoring.", self.regression_strategy)
            self.use_anchored_heads = False

        if self.ordering_constraint_weight is None:
            recommended_combo = (
                self.regression_strategy == RegressionStrategy.SEPARATE_HEADS
                and self.prob_reg_loss_type == ProbRegLossType.GAUSSIAN_LTV
                and self.optimization_strategy == ProbabilisticRegressionOptimizationStrategy.REGRESSION_ONLY
            )
            self.ordering_constraint_weight = 1.0 if recommended_combo else 0.0
        elif self.ordering_constraint_weight > 0.0 and self.prob_reg_loss_type == ProbRegLossType.MDN:
            logger.warning(
                "ordering_constraint_weight=%.3f with MDN loss is empirically harmful "
                "(see docs/probreg_identifiability_research.md §9.1 cells E vs F). "
                "Pass ordering_constraint_weight=0.0 to silence this warning.",
                self.ordering_constraint_weight,
            )

        if self.constrain_middle_class and self.regression_strategy == RegressionStrategy.SINGLE_HEAD_FINAL_OUTPUT:
            logger.warning("The `constrain_middle_class` option is not supported for the SINGLE_HEAD_FINAL_OUTPUT strategy.")
            self.constrain_middle_class = False

        if self.use_middle_class_nll_penalty:
            if self.regression_strategy == RegressionStrategy.SINGLE_HEAD_FINAL_OUTPUT:
                logger.warning("The `use_middle_class_nll_penalty` option is not supported for the SINGLE_HEAD_FINAL_OUTPUT strategy.")
                self.use_middle_class_nll_penalty = False
            elif self.n_classes % 2 == 0:
                logger.warning(f"The `use_middle_class_nll_penalty` option is enabled, but n_classes is {self.n_classes} (an even number). This option will have no effect.")

        self.base_classifier_params = self.base_classifier_params if self.base_classifier_params is not None else {}
        self.regression_head_params = self.regression_head_params if self.regression_head_params is not None else {}
        self.direct_regression_head_params = self.direct_regression_head_params if self.direct_regression_head_params is not None else {}

        self.direct_regression = self.n_classes_selection_method == NClassesSelectionMethod.NONE and self.n_classes >= self.n_classes_inf
        self.is_composite_regression_model = not self.direct_regression

        if self.direct_regression:
            logger.info(f"Number of classes ({self.n_classes}) >= n_classes_inf ({self.n_classes_inf}). Using direct regression mode.")
        elif self.n_classes_selection_method == NClassesSelectionMethod.NONE:
            logger.info(f"Using probabilistic regression mode with fixed {self.n_classes} classes.")
        else:
            logger.info(f"Using probabilistic regression mode with dynamic n_classes selection via {self.n_classes_selection_method.value}.")

        self.precomputed_class_boundaries = {}
        self.class_value_ranges_ = {}

    def fit(self, x: np.ndarray | pd.DataFrame, y: np.ndarray | pd.DataFrame | pd.Series, timestamps: np.ndarray | None = None) -> None:
        """Fits the model, honoring `self.capacity_selection` end-to-end (capacity-programme PA).

        Under `CapacitySelection.FIXED`/`PER_INPUT` this is exactly `PyTorchModelBase.fit`
        (PER_INPUT still needs an explicit follow-up `fit_router()` call, unchanged from FP-3).
        Under `CapacitySelection.GLOBAL_CHEAP`/`GLOBAL_SWEEP`, a single `fit(x, y)` call also
        performs the held-out global-k selection those modes need -- see `fit_global_selector`
        (M1) / `fit_sweep_selector` (M3) for the mechanism and `_split_off_selection_set` for how
        `self.selection_fraction` of `(x, y)` is carved out and held out from training for that
        selection read.
        """
        if self.capacity_selection == CapacitySelection.GLOBAL_CHEAP:
            self._fit_global_cheap(x, y)
            return
        if self.capacity_selection == CapacitySelection.GLOBAL_SWEEP:
            self._fit_global_sweep(x, y)
            return
        super().fit(x, y, timestamps=timestamps)

    def _split_off_selection_set(self, x: np.ndarray | pd.DataFrame, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Splits `(x, y)` into a fit portion and a held-out selection portion.

        Returns `(x_fit, y_fit, x_sel, y_sel)`, sized `(1 - selection_fraction)` /
        `selection_fraction`, for `CapacitySelection.GLOBAL_CHEAP`/`GLOBAL_SWEEP`'s internal
        k-selection step. The selection portion is held out from everything downstream -- the fit
        portion never sees it -- so the selection read is genuinely held-out, not merely
        early-stopping validation (which draws from `self.validation_fraction`, a separate,
        unrelated knob).

        Uses `self.split_strategy`/`self.random_seed`, the same split machinery `fit()` itself
        uses (`create_train_val_split`), with `validation_fraction=self.selection_fraction` and
        `test_fraction=0`.
        """
        x_arr = x.values if hasattr(x, "values") else np.asarray(x)
        y_arr = np.asarray(y)
        train_indices, sel_indices, _ = create_train_val_split(
            x=x_arr,
            validation_fraction=self.selection_fraction,
            test_fraction=0.0,
            split_strategy=self.split_strategy,
            timestamps=None,
            random_state=self.random_seed,
        )
        return x_arr[train_indices], y_arr[train_indices], x_arr[sel_indices], y_arr[sel_indices]

    def _fit_global_cheap(self, x: np.ndarray | pd.DataFrame, y: np.ndarray) -> None:
        """`CapacitySelection.GLOBAL_CHEAP` (M1): trains, then selects one k off the held-out remainder.

        Trains the NESTED net on `1 - self.selection_fraction` of the data, then picks ONE k for
        the whole dataset from the held-out remainder via `fit_global_selector`.
        """
        x_fit, y_fit, x_sel, y_sel = self._split_off_selection_set(x, y)
        super().fit(x_fit, y_fit)
        self.fit_global_selector(x_sel, y_sel)

    def _fit_global_sweep(self, x: np.ndarray | pd.DataFrame, y: np.ndarray) -> None:
        """`CapacitySelection.GLOBAL_SWEEP` (M3): trains one ordinary model per k, keeps the winner.

        Trains a SEPARATE ORDINARY model (`NClassesSelectionMethod.NONE`) per candidate k on
        `1 - self.selection_fraction` of the data, scores each on the held-out remainder, and
        keeps the winner -- see `fit_sweep_selector` for the selection rule (the SAME rule
        `fit_global_selector` uses).
        """
        x_fit, y_fit, x_sel, y_sel = self._split_off_selection_set(x, y)
        self.fit_sweep_selector(x_fit, y_fit, x_sel, y_sel)

    @property
    def name(self) -> str:
        """Returns the name of the model."""
        return f"ProbabilisticRegression_{self.regression_strategy.value}"

    @property
    def regression_heads(self) -> nn.ModuleList:
        """Returns the regression heads module list."""
        return self.model.regression_module.heads

    def _calculate_custom_loss(self, model_outputs: tuple, y_true: torch.Tensor, include_boundary_loss: bool = True) -> torch.Tensor:
        """Calculates the loss for the ProbabilisticRegressionModel."""
        final_predictions, classifier_logits_out, selected_k_values, _log_prob_for_reinforce, per_head_outputs = model_outputs
        y_true_squeezed = y_true.squeeze(-1) if y_true.ndim > 1 else y_true

        # 1. Calculate main regression loss
        if self.prob_reg_loss_type == ProbRegLossType.MDN and per_head_outputs is not None and self.uncertainty_method == UncertaintyMethod.PROBABILISTIC:
            # MDN NLL: probabilities enter the likelihood directly → structural identifiability.
            # Slice before softmax so the distribution is over exactly n_classes components.
            # Under CE_STOP_GRAD/GRADIENT_STOP the classifier is not supervised by regression
            # loss; detach logits here so MDN's probs path does not become a back-door channel
            # (regression_module already received detached probs in NoneStrategy.forward).
            logits_for_mdn = classifier_logits_out
            if self.optimization_strategy in (
                ProbabilisticRegressionOptimizationStrategy.CE_STOP_GRAD,
                ProbabilisticRegressionOptimizationStrategy.GRADIENT_STOP,
            ):
                logits_for_mdn = logits_for_mdn.detach()
            probs_for_mdn = torch.softmax(logits_for_mdn[:, : self.n_classes], dim=-1)
            mus = per_head_outputs[:, : self.n_classes, 0]
            log_vars = per_head_outputs[:, : self.n_classes, 1]
            regression_loss = mdn_nll(y_true_squeezed, probs_for_mdn, mus, log_vars)
        else:
            regression_loss = calculate_combined_loss(
                predictions=final_predictions,
                y_true=y_true,
                uncertainty_method=self.uncertainty_method,
                include_boundary_loss=False,
                loss_type=self.loss_type,
                beta=self.beta,
            )

        total_loss = regression_loss
        unique_k = torch.tensor([])
        probabilistic_indices = torch.tensor([])

        # 2. Classification loss: existing strategies + CE_STOP_GRAD
        is_ce_active = self.optimization_strategy not in (
            ProbabilisticRegressionOptimizationStrategy.REGRESSION_ONLY,
            ProbabilisticRegressionOptimizationStrategy.GRADIENT_STOP,
        )
        if is_ce_active:
            # Exclude samples that took the direct-regression bypass (sentinel value).
            # Was `< self.n_classes_inf`, which fails when n_classes_inf=inf (default) because
            # the sentinel 2**30 < inf and the sentinel then flows into precomputed_class_boundaries[k].
            probabilistic_indices = torch.where(selected_k_values != DIRECT_REGRESSION_K_SENTINEL)[0]

            if probabilistic_indices.numel() > 0:
                y_true_prob = y_true_squeezed[probabilistic_indices]
                logits_prob = classifier_logits_out[probabilistic_indices]
                k_values_prob = selected_k_values[probabilistic_indices]

                y_binned_prob = torch.zeros_like(y_true_prob, dtype=torch.long)
                unique_k = torch.unique(k_values_prob)

                for k in unique_k:
                    k_int = int(k.item())
                    mask = k_values_prob == k
                    boundaries = self.precomputed_class_boundaries[k_int]
                    _, y_binned_k = create_bins(data=y_true_prob[mask].cpu().numpy(), unique_bin_edges=boundaries)
                    y_binned_prob[mask] = torch.tensor(y_binned_k, dtype=torch.long, device=self.device)

                classification_loss = masked_cross_entropy_loss(logits_prob, y_binned_prob, k_values_prob)
                total_loss += classification_loss

        # 3. Apply additional penalties (Middle Class NLL, Boundary Regularization)
        boundary_reg_active = self.boundary_regularization_method != BoundaryRegularizationMethod.NONE and include_boundary_loss
        if per_head_outputs is not None and (self.use_middle_class_nll_penalty or boundary_reg_active):
            if probabilistic_indices.numel() == 0:
                probabilistic_indices = torch.where(selected_k_values != DIRECT_REGRESSION_K_SENTINEL)[0]

            if probabilistic_indices.numel() > 0:
                y_true_prob = y_true_squeezed[probabilistic_indices]
                per_head_outputs_prob = per_head_outputs[probabilistic_indices]
                k_values_prob = selected_k_values[probabilistic_indices]

                if not torch.is_tensor(unique_k) or unique_k.numel() == 0:
                    unique_k = torch.unique(k_values_prob)

                for k in unique_k:
                    k_int = int(k.item())
                    k_mask = k_values_prob == k
                    if not torch.any(k_mask):
                        continue

                    total_loss = apply_additional_penalties(
                        total_loss=total_loss,
                        per_head_outputs=per_head_outputs_prob[k_mask],
                        y_true_squeezed=y_true_prob[k_mask],
                        model_instance=self,
                        include_boundary_loss=include_boundary_loss,
                        class_boundaries=self.precomputed_class_boundaries[k_int],
                        class_value_ranges=self.class_value_ranges_.get(k_int),
                        middle_class_dist_params=self.middle_class_dist_params_.get(k_int),
                    )


        # 3b. Ordering-constraint identifiability penalty. See
        # docs/probreg_identifiability_research.md §3.3. Only SEPARATE_HEADS has a
        # permutation symmetry to break.
        ordering_weight = getattr(self, "ordering_constraint_weight", 0.0)
        if (
            ordering_weight > 0.0
            and self.regression_strategy == RegressionStrategy.SEPARATE_HEADS
            and per_head_outputs is not None
            and classifier_logits_out is not None
        ):
            # Detach under stop-grad optimisation strategies so the ordering
            # constraint does not become a back-door gradient path into the
            # classifier (mirrors the MDN path above).
            logits_for_order = classifier_logits_out[:, : self.n_classes]
            if self.optimization_strategy in (
                ProbabilisticRegressionOptimizationStrategy.CE_STOP_GRAD,
                ProbabilisticRegressionOptimizationStrategy.GRADIENT_STOP,
            ):
                logits_for_order = logits_for_order.detach()
            head_means = per_head_outputs[:, : self.n_classes, 0]
            order_term = ordering_loss_fn(
                logits_for_order,
                head_means,
                top_decile_fraction=getattr(self, "ordering_top_decile_fraction", 0.1),
                margin=getattr(self, "ordering_constraint_margin", 0.0),
            )
            total_loss = total_loss + ordering_weight * order_term

        # 4. ELBO / k-penalty regularization for dynamic n_classes
        if self.n_classes_selection_method != NClassesSelectionMethod.NONE and self.n_classes_regularization != NClassesRegularization.NONE:
            k_probs = self.model.n_classes_strategy.mode_selection_probs
            if k_probs is not None:
                n_modes = k_probs.size(1)
                if self.n_classes_regularization == NClassesRegularization.ELBO:
                    # Split KL into two components:
                    #   (a) Bernoulli on bypass vs not-bypass with prior p(bypass) = bypass_prior_prob.
                    #   (b) Categorical on k ∈ {2..k_max} conditional on not-bypass, with a uniform
                    #       or geometric prior, weighted by per-sample (1 − p_bypass) mass.
                    # This removes two earlier issues: (1) the monolithic linspace prior placed
                    # bypass at the least-favoured end, actively suppressing the correct answer
                    # on heavy-tail data; (2) the constant logit range [3,1] became near-uniform
                    # as n_modes grew.
                    eps = 1e-8
                    p_bypass_per_sample = k_probs[:, -1]
                    p_bypass = p_bypass_per_sample.mean()
                    bypass_prior = torch.tensor(self.bypass_prior_prob, device=k_probs.device, dtype=k_probs.dtype)
                    kl_bypass = (
                        p_bypass * torch.log((p_bypass + eps) / (bypass_prior + eps))
                        + (1.0 - p_bypass) * torch.log((1.0 - p_bypass + eps) / (1.0 - bypass_prior + eps))
                    )

                    if n_modes > 1:
                        prob_mode_probs = k_probs[:, :-1]
                        prob_mode_mass = prob_mode_probs.sum(dim=-1, keepdim=True).clamp_min(eps)
                        q_given_not_bypass = prob_mode_probs / prob_mode_mass
                        n_probabilistic_modes = n_modes - 1
                        if self.k_prior_type == "geometric":
                            k_indices = torch.arange(n_probabilistic_modes, dtype=k_probs.dtype, device=k_probs.device)
                            log_prior = k_indices * torch.log(torch.tensor(1.0 - self.k_prior_geometric_lambda, device=k_probs.device, dtype=k_probs.dtype))
                            prior_logits_k = log_prior
                        else:  # "uniform"
                            prior_logits_k = torch.zeros(n_probabilistic_modes, device=k_probs.device, dtype=k_probs.dtype)
                        q_k = torch.distributions.Categorical(probs=q_given_not_bypass + eps)
                        p_k = torch.distributions.Categorical(logits=prior_logits_k)
                        kl_k_per_sample = torch.distributions.kl_divergence(q_k, p_k)
                        kl_k = ((1.0 - p_bypass_per_sample) * kl_k_per_sample).mean()
                    else:
                        kl_k = torch.zeros((), device=k_probs.device, dtype=k_probs.dtype)

                    total_loss = total_loss + kl_bypass + kl_k
                elif self.n_classes_regularization == NClassesRegularization.K_PENALTY:
                    # Weighted expected k: penalize selecting higher k values
                    k_indices = torch.arange(1, n_modes + 1, dtype=torch.float, device=k_probs.device)
                    expected_k = torch.sum(k_probs * k_indices, dim=1)
                    total_loss = total_loss + self.k_penalty_weight * expected_k.mean()

        return total_loss

    def _calculate_performance_score(self, y_true: np.ndarray, y_pred: np.ndarray, x_val: np.ndarray | None = None, y_pred_std: np.ndarray | None = None) -> float:
        """Calculates the performance score, including the classification penalty if applicable."""
        score = super()._calculate_performance_score(y_true, y_pred, x_val, y_pred_std)

        if self.optimization_strategy != ProbabilisticRegressionOptimizationStrategy.REGRESSION_ONLY:
            self.model.eval()
            with torch.no_grad():
                x_val_tensor = torch.tensor(x_val, dtype=torch.float32).to(self.device)
                y_val_tensor = torch.tensor(y_true, dtype=torch.float32).to(self.device).unsqueeze(1)

                model_outputs = self.model(x_val_tensor)
                # Add the classification/boundary loss to the regression score
                score += self._calculate_custom_loss(model_outputs, y_val_tensor, include_boundary_loss=False).item()

        return score

    def build_model(self) -> None:
        """Builds the internal PyTorch nn.Module for the ProbabilisticRegressionModel."""
        if self.n_classes_selection_method != NClassesSelectionMethod.NONE and self.max_n_classes_for_probabilistic_path >= self.n_classes_inf:
            raise ValueError("max_n_classes_for_probabilistic_path must be less than n_classes_inf when n_classes_selection_method is not NONE.")

        self.model = ProbabilisticRegressionNet(
            input_size=self.input_size,
            n_classes=self.n_classes,
            n_classes_inf=self.n_classes_inf,
            max_n_classes_for_probabilistic_path=self.max_n_classes_for_probabilistic_path,
            base_classifier_params=self.base_classifier_params,
            regression_head_params=self.regression_head_params,
            direct_regression_head_params=self.direct_regression_head_params,
            regression_strategy=self.regression_strategy,
            uncertainty_method=self.uncertainty_method,
            n_classes_selection_method=self.n_classes_selection_method,
            optimization_strategy=self.optimization_strategy,
            gumbel_tau=self.gumbel_tau,
            n_classes_predictor_learning_rate=self.n_classes_predictor_learning_rate,
            device=self.device,
            use_monotonic_constraints=self.use_monotonic_constraints,
            constrain_middle_class=self.constrain_middle_class,
            centroids=getattr(self, "_per_class_centroids", None),
            use_anchored_heads=self.use_anchored_heads,
        )
        self.model.to(self.device)

        init_val = getattr(self, "_constant_head_init_value", None)
        if init_val is not None:
            self.model.regression_module.init_middle_class_mean(init_val)

        self.criterion = self._calculate_custom_loss

    def get_hyperparameter_search_space(self) -> dict[str, Any]:
        """Defines the hyperparameter search space for ProbabilisticRegressionModel.

        Returns:
            Dict[str, Any]: A dictionary defining the hyperparameter search space.
        """
        space = super().get_hyperparameter_search_space()
        space.update(
            {
                "regression_strategy": {"type": "categorical", "choices": [s.value for s in RegressionStrategy]},
                "optimization_strategy": {"type": "categorical", "choices": [s.value for s in ProbabilisticRegressionOptimizationStrategy]},
                "base_classifier_params__hidden_layers": {"type": "int", "low": 1, "high": 2},
                "base_classifier_params__hidden_size": {"type": "int", "low": 32, "high": 64, "step": 32},
                "base_classifier_params__use_batch_norm": {"type": "categorical", "choices": [True, False]},
                "base_classifier_params__dropout_rate": {"type": "float", "low": 0.0, "high": 0.5, "step": 0.1},
                "regression_head_params__hidden_layers": {"type": "int", "low": 0, "high": 1},
                "regression_head_params__hidden_size": {"type": "int", "low": 16, "high": 32, "step": 16},
                "regression_head_params__use_batch_norm": {"type": "categorical", "choices": [True, False]},
                "regression_head_params__dropout_rate": {"type": "float", "low": 0.0, "high": 0.5, "step": 0.1},
                "boundary_regularization_method": {"type": "categorical", "choices": [e.value for e in BoundaryRegularizationMethod]},
                "boundary_loss_weight": {"type": "float", "low": 0.1, "high": 10.0, "log": True},
            }
        )

        if self.n_classes_selection_method == NClassesSelectionMethod.NONE:
            space["n_classes"] = {"type": "int", "low": 2, "high": (int(self.n_classes_inf) - 1 if self.n_classes_inf != float("inf") else 5)}
        else:
            space["max_n_classes_for_probabilistic_path"] = {"type": "int", "low": 2, "high": (int(self.n_classes_inf) - 1 if self.n_classes_inf != float("inf") else 10)}
            space["direct_regression_head_params__hidden_layers"] = {"type": "int", "low": 1, "high": 2}
            space["direct_regression_head_params__hidden_size"] = {"type": "int", "low": 32, "high": 64, "step": 32}
            space["direct_regression_head_params__use_batch_norm"] = {"type": "categorical", "choices": [True, False]}
            space["direct_regression_head_params__dropout_rate"] = {"type": "float", "low": 0.0, "high": 0.5, "step": 0.1}

            # gumbel_tau only feeds f.gumbel_softmax in GumbelSoftmaxStrategy/SteStrategy
            # (n_classes_strategies.py); n_classes_predictor_learning_rate only feeds
            # ReinforceStrategy's policy optimizer. Both are inert dead HPO dims otherwise.
            if self.n_classes_selection_method in (NClassesSelectionMethod.GUMBEL_SOFTMAX, NClassesSelectionMethod.STE):
                space["gumbel_tau"] = {"type": "float", "low": 1e-8, "high": 1.0, "log": True}
            if self.n_classes_selection_method == NClassesSelectionMethod.REINFORCE:
                space["n_classes_predictor_learning_rate"] = {"type": "float", "low": 1e-8, "high": 1e-2, "log": True}

        if self.search_space_override:
            space.update(self.search_space_override)

        return space

    def get_internal_model(self) -> Any:
        """Returns a wrapper around the internal model that is compatible with SHAP."""

        class _ShapModelWrapper(nn.Module):
            def __init__(self, model: nn.Module) -> None:
                super().__init__()
                self.model = model

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.model(x)[0]

        return _ShapModelWrapper(self.model)

    def _setup_optimizers(self, model: nn.Module) -> None:
        super()._setup_optimizers(model)
        if self.n_classes_selection_method != NClassesSelectionMethod.NONE and hasattr(self.model, "n_classes_predictor") and self.model.n_classes_predictor is not None:
            n_classes_predictor_params = self.model.n_classes_predictor.parameters()
            self.model.n_classes_strategy.setup_optimizers(n_classes_predictor_params)

    def _fit_single(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        forced_iterations: int | None = None,
        forward_pass_kwargs: dict | None = None,
    ) -> tuple[int, list[float]]:
        """Fits a single model instance.

        Args:
            x_train (np.ndarray): The training features.
            y_train (np.ndarray): The training targets.
            x_val (np.ndarray | None): The validation features.
            y_val (np.ndarray | None): The validation targets.
            forced_iterations (int | None): If provided, train for this many iterations, ignoring early stopping.
            forward_pass_kwargs (dict | None): Keyword arguments to pass to the model's forward pass.

        Returns:
            tuple[int, list[float]]: A tuple containing:
                - The number of iterations the model was trained for.
                - A list of the validation loss values for each epoch.
        """
        if self.target_transform == "symlog":
            y_train = symlog(torch.tensor(y_train, dtype=torch.float32)).numpy()
            if y_val is not None:
                y_val = symlog(torch.tensor(y_val, dtype=torch.float32)).numpy()

        is_ce_active = self.optimization_strategy not in (
            ProbabilisticRegressionOptimizationStrategy.REGRESSION_ONLY,
            ProbabilisticRegressionOptimizationStrategy.GRADIENT_STOP,
        )
        if self.n_classes_selection_method != NClassesSelectionMethod.NONE and is_ce_active:
            logger.warning(
                "Dynamic n_classes selection (%s) with optimization_strategy=%s activates "
                "classification cross-entropy: class boundaries are precomputed independently per "
                "k, so per-k re-binned CE targets redefine class identity across k (node-0 "
                "conflict). This combination is NOT validated. The validated recipe of record is "
                "optimization_strategy=REGRESSION_ONLY (see "
                "docs/plans/width_dial_2026-07-11/cascade_execution_plan_2026-07-11.md §3.1).",
                self.n_classes_selection_method.value,
                self.optimization_strategy.value,
            )

        if not self.direct_regression:
            self.precomputed_class_boundaries = {}
            self.class_value_ranges_ = {}
            middle_class_params_per_k = {}
            y_flat = y_train.flatten() if y_train.ndim > 1 else y_train
            y_min, y_max = np.min(y_flat), np.max(y_flat)

            max_k = self.max_n_classes_for_probabilistic_path if self.n_classes_selection_method != NClassesSelectionMethod.NONE else self.n_classes
            k_values = range(2, max_k + 1) if self.n_classes_selection_method != NClassesSelectionMethod.NONE else [max_k]

            for k in k_values:
                boundaries, y_binned = create_bins(data=y_flat, n_bins=k, min_value=-np.inf, max_value=np.inf)
                self.precomputed_class_boundaries[k] = boundaries

                if self.use_middle_class_nll_penalty:
                    # The mixin stores result in self.middle_class_dist_params_ as a flat dict.
                    # Capture it per k before the next iteration overwrites it.
                    self._calculate_middle_class_dist_params(y_flat, y_binned, n_classes=k)
                    middle_class_params_per_k[k] = self.middle_class_dist_params_

                if self.boundary_regularization_method != BoundaryRegularizationMethod.NONE:
                    self.class_value_ranges_[k] = calculate_class_value_ranges(y_flat=y_flat, y_binned=y_binned, k=k, y_min=y_min, y_max=y_max, device=self.device)

            self.middle_class_dist_params_ = middle_class_params_per_k

            # y_binned here is from the last loop iteration (k == max_k); reuse to avoid a second create_bins call.
            if self.constrain_middle_class and max_k % 2 == 1:
                mid_mask = y_binned == max_k // 2
                self._constant_head_init_value = float(y_flat[mid_mask].mean()) if mid_mask.any() else 0.0
            else:
                self._constant_head_init_value = None

            # Per-class centroids for monotonic head init (B1 fix) and anchored heads (C6).
            # Only computed when one of those two consumers is actually active — otherwise the
            # result is never read (_initialize_monotonic_head gates on use_monotonic_constraints,
            # AnchoredHead gates on use_anchored_heads).
            if self.regression_strategy == RegressionStrategy.SEPARATE_HEADS and (self.use_monotonic_constraints or self.use_anchored_heads):
                counts = np.bincount(y_binned, minlength=max_k)
                sums = np.bincount(y_binned, weights=y_flat, minlength=max_k)
                self._per_class_centroids = list(np.where(counts > 0, sums / counts, 0.0))

        forward_pass_kwargs = None
        if self.boundary_regularization_method == BoundaryRegularizationMethod.HARDSIGMOID:
            if not self.class_value_ranges_:
                raise ValueError("class_value_ranges must be pre-calculated for the HARDSIGMOID boundary method.")
            # This is a simplification. A more robust solution would handle multiple k values.
            k = next(iter(self.class_value_ranges_))
            forward_pass_kwargs = {"boundaries": self.class_value_ranges_[k].to(self.device)}

        return super()._fit_single(x_train, y_train, x_val, y_val, forced_iterations, forward_pass_kwargs=forward_pass_kwargs)

    def predict(self, x: np.ndarray | pd.DataFrame, filter_data: bool = True) -> np.ndarray:
        """Predicts on x, applying inverse target transform if configured.

        For symlog, predictions are computed as the MC mean of ``symexp(samples)`` so
        that the reported point estimate is consistent with the std produced by
        :meth:`predict_uncertainty` (both come from the same posterior sample).

        Args:
            x: Input features.
            filter_data: Whether to filter input columns to those seen at fit time.

        Under `capacity_selection=CapacitySelection.FIXED` (default) uses the trained selection
        strategy/full forward pass. Under `CapacitySelection.PER_INPUT`, routes with a
        `DistilledCapacityRouter` fitted via `fit_router()` -- no caller flag needed
        (capacity-programme Task FP-3; MASTER Decision 13). Under `CapacitySelection.GLOBAL_CHEAP`
        (M1), forces every input through the single dataset-wide k `fit()` selected via
        `fit_global_selector`. Under `CapacitySelection.GLOBAL_SWEEP` (M3), delegates to the
        winning per-k model `fit()` selected via `fit_sweep_selector`.
        """
        if self.capacity_selection == CapacitySelection.PER_INPUT:
            mean, log_var = self._forward_routed(x, filter_data=filter_data)
            if self.target_transform == "symlog":
                std_symlog = np.sqrt(np.exp(log_var))
                mean_orig, _ = self._symlog_mc_moments(mean, std_symlog)
                return mean_orig
            return mean

        if self.capacity_selection == CapacitySelection.GLOBAL_CHEAP:
            mean, log_var = self._forward_global_k(x, filter_data=filter_data)
            if self.target_transform == "symlog":
                std_symlog = np.sqrt(np.exp(log_var))
                mean_orig, _ = self._symlog_mc_moments(mean, std_symlog)
                return mean_orig
            return mean

        if self.capacity_selection == CapacitySelection.GLOBAL_SWEEP:
            return self._sweep_submodel().predict(x, filter_data=filter_data)

        return self._predict_unselected(x, filter_data=filter_data)

    def _predict_unselected(self, x: np.ndarray | pd.DataFrame, filter_data: bool = True) -> np.ndarray:
        """The trained net's OWN forward, bypassing the capacity-selection gate entirely.

        This is `CapacitySelection.FIXED`'s prediction path, factored out so that internal
        bookkeeping can reach it *before* a global k has been selected -- see
        `_fit_residual_std`/`_predict_for_scoring` below for why that matters.
        """
        predictions = super().predict(x, filter_data=filter_data)
        if self.target_transform == "symlog":
            std_symlog = super().predict_uncertainty(x, filter_data=filter_data)
            mean_orig, _ = self._symlog_mc_moments(predictions, std_symlog)
            return mean_orig
        return predictions

    def _fit_residual_std(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """Computes the CONSTANT-uncertainty residual std WITHOUT routing through the selection gate.

        Overrides `PyTorchModelBase._fit_residual_std` (capacity-programme FP-10) for the same
        reason `FlexibleWidthNN` does, but for a different gate. The base implementation calls the
        caller-facing `self.predict()`; under `GLOBAL_CHEAP` that raises, because this hook runs at
        the END of training and the global k is selected only AFTER training returns
        (`_fit_global_cheap`: `super().fit(...)` THEN `fit_global_selector(...)`). The model is
        therefore mid-fit with no `selected_k_` yet, and the caller-facing path is correct to refuse.
        Bookkeeping uses the un-selected forward instead.
        *(Integration defect caught by the root's post-enum smoke test, 2026-07-21: PA built the two
        global modes while `enums.py` still lacked their members, so this path could not be exercised
        end-to-end by PA; FP-10 landed the base hook without knowing the modes were coming. Neither
        task could have found it alone.)*
        """
        y_pred_train = self._predict_unselected(x_train, filter_data=False)
        self._train_residual_std = np.std(y_train - y_pred_train)

    def _predict_for_scoring(self, x: np.ndarray | pd.DataFrame, filter_data: bool = True) -> np.ndarray:
        """Internal scoring path for CV folds, the HPO objective and `evaluate()` (FP-10).

        Same rationale as `_fit_residual_std`: `BaseModel`'s generic machinery predicts
        polymorphically with no knowledge of capacity selection. Under `GLOBAL_SWEEP` the fitted
        sub-model IS the model, so scoring must go through it; otherwise the un-selected forward is
        the honest internal answer (under `GLOBAL_CHEAP` a global k may not exist yet).
        """
        if self.capacity_selection == CapacitySelection.GLOBAL_SWEEP and getattr(self, "_sweep_submodel_", None) is not None:
            return self._sweep_submodel().predict(x, filter_data=filter_data)
        return self._predict_unselected(x, filter_data=filter_data)

    def predict_distribution(self, x: np.ndarray | pd.DataFrame, filter_data: bool = True) -> MixtureOfGaussiansDistribution:
        """Returns the full per-input mixture-of-Gaussians predictive distribution.

        The classification bottleneck IS a mixture: each class k contributes a
        Gaussian (mu_k(x), sigma_k^2(x)) weighted by p(class=k | x). This method
        exposes that full distribution for proper evaluation (NLL, CRPS on the
        mixture, PIT histograms) rather than the collapsed Gaussian that
        ``predict`` + ``predict_uncertainty`` return via the law of total variance.

        Only supported for ``uncertainty_method=PROBABILISTIC`` with
        ``regression_strategy in {SEPARATE_HEADS, SINGLE_HEAD_N_OUTPUTS}`` and
        ``n_classes_selection_method=NONE``. Dynamic-k is not yet supported
        because the predictive mixture under dynamic-k mixes per-k predictions
        weighted by the k-selection distribution, not a single set of per-class
        heads; ``SINGLE_HEAD_FINAL_OUTPUT`` does not produce per-class parameters.

        Raises:
            NotImplementedError: for unsupported configurations.
            RuntimeError: if the model has not been fitted.
        """
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        if self.uncertainty_method != UncertaintyMethod.PROBABILISTIC:
            raise NotImplementedError(
                f"predict_distribution requires uncertainty_method=PROBABILISTIC, got {self.uncertainty_method}."
            )
        if self.regression_strategy == RegressionStrategy.SINGLE_HEAD_FINAL_OUTPUT:
            raise NotImplementedError(
                "predict_distribution is not available for SINGLE_HEAD_FINAL_OUTPUT; no per-class (mu, sigma) is produced."
            )
        if self.n_classes_selection_method != NClassesSelectionMethod.NONE:
            raise NotImplementedError(
                "predict_distribution with dynamic-k is not yet implemented (per_head_outputs not exposed by dynamic strategies)."
            )
        if self.target_transform == "symlog":
            raise NotImplementedError(
                "predict_distribution with target_transform='symlog' is not supported; "
                "the mixture is in symlog space and would need MC push-through symexp to be meaningful."
            )

        if filter_data:
            x = self._filter_predict_data(x)
        if isinstance(x, pd.DataFrame):
            x = x.values
        x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)

        self.model.eval()
        with torch.no_grad():
            classifier_raw_logits = self.model.classifier_layers(x_tensor)
            masked = torch.full_like(classifier_raw_logits, float("-inf"))
            masked[:, : self.n_classes] = classifier_raw_logits[:, : self.n_classes]
            probabilities = torch.softmax(masked, dim=1)[:, : self.n_classes]
            # SeparateHeads / SingleHeadNOutputs both honor return_head_outputs=True.
            _, per_head_outputs = self.model.regression_module(
                torch.softmax(masked, dim=1), return_head_outputs=True,
            )
            # per_head_outputs: (N, n_classes, 2) — col 0 mean, col 1 log_var.
            per_head_outputs = per_head_outputs[:, : self.n_classes, :]
            means = per_head_outputs[:, :, 0].cpu().numpy()
            log_var = per_head_outputs[:, :, 1].cpu().numpy()
        weights = probabilities.cpu().numpy()
        stds = np.sqrt(np.exp(log_var))
        return MixtureOfGaussiansDistribution(weights=weights, means=means, stds=stds)

    def predict_uncertainty(self, x: np.ndarray, filter_data: bool = True) -> np.ndarray:
        """Estimates uncertainty, converting from symlog space if configured.

        For symlog targets the linearized Jacobian (``exp(|μ|)``) is inaccurate
        near zero crossings and underestimates spread when ``σ_symlog`` is large.
        We instead push samples from ``N(μ_symlog, σ_symlog²)`` through ``symexp``
        and report the empirical std — this is exact in the limit of many samples
        and adds negligible cost (~one extra forward of arithmetic).

        Args:
            x: Input features.
            filter_data: Whether to filter input columns to those seen at fit time.

        Follows `self.capacity_selection`, same as :meth:`predict`.
        """
        if self.capacity_selection == CapacitySelection.PER_INPUT:
            mean, log_var = self._forward_routed(x, filter_data=filter_data)
            std = np.sqrt(np.exp(log_var))
            if self.target_transform == "symlog":
                _, std_orig = self._symlog_mc_moments(mean, std)
                return std_orig
            return std

        if self.capacity_selection == CapacitySelection.GLOBAL_CHEAP:
            mean, log_var = self._forward_global_k(x, filter_data=filter_data)
            std = np.sqrt(np.exp(log_var))
            if self.target_transform == "symlog":
                _, std_orig = self._symlog_mc_moments(mean, std)
                return std_orig
            return std

        if self.capacity_selection == CapacitySelection.GLOBAL_SWEEP:
            return self._sweep_submodel().predict_uncertainty(x, filter_data=filter_data)

        std = super().predict_uncertainty(x, filter_data=filter_data)
        if self.target_transform == "symlog":
            mean_symlog = super().predict(x, filter_data=filter_data)
            _, std_orig = self._symlog_mc_moments(mean_symlog, std)
            return std_orig
        return std

    def _forward_routed(self, x: np.ndarray | pd.DataFrame, filter_data: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """Per-sample (mean, log_var) from the `DistilledCapacityRouter`-routed rung.

        Each sample's rung -- `(1,)` the bypass, `(k,)` for `k>=2` the renormalized k-class
        mixture -- is picked by `self.capacity_router_` (fit via `fit_router()`), then forced
        through `ProbabilisticRegressionNet.forward_at_k`. Values are in whatever space the
        model was fit in (symlog-transformed if `target_transform="symlog"`); callers apply the
        same MC push-through `predict`/`predict_uncertainty` already use for the non-routed path.
        """
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        if getattr(self, "capacity_router_", None) is None:
            raise RuntimeError("No router fitted; call fit_router() before predict() under CapacitySelection.PER_INPUT.")
        if filter_data:
            x = self._filter_predict_data(x)
        x_array = x.values if hasattr(x, "values") else x
        x_tensor = torch.tensor(x_array, dtype=torch.float32).to(self.device)

        k_np = np.array([capacity[0] for capacity in self.capacity_router_.route(x_array)], dtype=np.int64)
        out = torch.zeros(x_tensor.size(0), self.model.regression_output_size, device=self.device)
        self.model.eval()
        with torch.no_grad():
            for k in np.unique(k_np):
                mask = k_np == k
                out[mask] = self.model.forward_at_k(x_tensor[mask], int(k))
        mean = out[:, 0].cpu().numpy()
        log_var = out[:, 1].cpu().numpy()
        return mean, log_var

    def _forward_global_k(self, x: np.ndarray | pd.DataFrame, filter_data: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """Forces EVERY input through the single global k `fit_global_selector` picked.

        Returns `(mean, log_var)` at `self.selected_k_` -- `CapacitySelection.GLOBAL_CHEAP`'s
        (M1) forward path. Values are in whatever space the model was fit in (symlog-transformed if
        `target_transform="symlog"`); callers apply the same MC push-through
        `predict`/`predict_uncertainty` already use for the non-routed path.
        """
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        if getattr(self, "selected_k_", None) is None:
            raise RuntimeError("No global k selected; call fit_global_selector() before predict() under CapacitySelection.GLOBAL_CHEAP.")
        if filter_data:
            x = self._filter_predict_data(x)
        x_array = x.values if hasattr(x, "values") else x
        x_tensor = torch.tensor(x_array, dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            out = self.model.forward_at_k(x_tensor, self.selected_k_)
        mean = out[:, 0].cpu().numpy()
        log_var = out[:, 1].cpu().numpy()
        return mean, log_var

    def _sweep_submodel(self) -> "ProbabilisticRegressionModel":
        """Returns the winning per-k model `fit_sweep_selector` trained.

        `CapacitySelection.GLOBAL_SWEEP`'s (M3) forward path -- `predict`/`predict_uncertainty`
        delegate to it directly.
        """
        submodel = getattr(self, "_sweep_submodel_", None)
        if submodel is None:
            raise RuntimeError("No sweep selector fitted; call fit_sweep_selector() before predict() under CapacitySelection.GLOBAL_SWEEP.")
        return submodel

    def fit_router(
        self,
        x_val: np.ndarray,
        y_val: np.ndarray,
        capacity_grid: list[tuple[int, ...]] | None = None,
        tolerance: float = DEFAULT_TOLERANCE,
        cost_fn: Callable[[tuple[int, ...]], float] | None = None,
    ) -> DistilledCapacityRouter:
        """Fits a `DistilledCapacityRouter` post-hoc for `predict()` under `CapacitySelection.PER_INPUT`.

        Decision 13 (distilled, never in-training): trains an MLP mapping raw `x_val` to a rung,
        entirely separate from the trained selection strategy. `eval_fn` forces every sample
        through a fixed rung via `ProbabilisticRegressionNet.forward_at_k` (bypassing the
        selection strategy for that call) and scores it with per-sample Gaussian NLL -- the
        natural error metric for a probabilistic model (unlike
        `FlexibleHiddenLayersNN`/`FlexibleWidthNN`'s squared-error `eval_fn`, which have no
        predictive variance to score against).

        Requires `n_classes_selection_method=NESTED`: routing assumes every rung's k-class
        mixture is individually well-calibrated (the K4 nesting property -- "the first c rungs
        are a genuine c-component mixture for every c"), which only the NESTED training scheme
        guarantees. Reading a per-rung ladder off one of the OTHER dynamic strategies would
        reproduce the R1 catastrophe those strategies were never audited against: prefix-slicing
        an unordered/non-nested mixture is invalid (`docs/plans/capacity_programme/...`,
        `automl_package/examples/capacity_ladder_results/R1_verdict.md`).

        Args:
            x_val: held-out inputs.
            y_val: held-out targets, in the SAME space as `fit()`'s `y_train`/`y_val` -- i.e.
                raw/original target units. When `target_transform="symlog"`, `y_val` is
                auto-transformed internally (via the same `symlog` helper `fit()` uses) before
                scoring, to match `forward_at_k`'s symlog-space outputs; do NOT pre-transform it
                yourself. This mirrors `fit()`'s contract exactly so callers can pass the same
                `y_val` array to both.
            capacity_grid: candidate rungs as 1-tuples, e.g. `[(1,), (2,), (3,)]` (`(1,)` is the
                bypass). Defaults to every rung `1..max_n_classes_for_probabilistic_path`.
            tolerance: cheapest-within-tolerance labeling tolerance (see `DistilledCapacityRouter`).
            cost_fn: `cost_fn(capacity) -> float`. Unlike `FlexibleHiddenLayersNN`/
                `FlexibleWidthNN`, no default is wired here -- no `executed_flops` accounting
                exists yet for this architecture (capacity-programme scope). Pass one explicitly
                to use `mean_deployed_cost`; `None` (default) leaves it disabled.

        Returns:
            The fitted `DistilledCapacityRouter` (also stored as `self.capacity_router_`).

        Raises:
            RuntimeError: the model has not been fitted, or was not trained with
                `n_classes_selection_method=NESTED`.
        """
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        if self.n_classes_selection_method != NClassesSelectionMethod.NESTED:
            raise RuntimeError(
                "fit_router requires n_classes_selection_method=NESTED (the certified prefix-nesting "
                f"property; got {self.n_classes_selection_method.value}). See fit_router docstring."
            )
        if capacity_grid is None:
            capacity_grid = [(k,) for k in range(1, self.max_n_classes_for_probabilistic_path + 1)]

        y_val_arr = np.asarray(y_val, dtype=np.float64)
        if self.target_transform == "symlog":
            # forward_at_k's outputs live in symlog space (fit() transforms y_train/y_val before
            # training, :526-529) -- eval_fn below must score against y_val in that same space, or
            # the per-capacity error table is silently wrong (F9-fix-b).
            y_val_arr = symlog(torch.tensor(y_val_arr, dtype=torch.float32)).numpy().astype(np.float64)

        def eval_fn(x: np.ndarray, capacity: tuple[int, ...]) -> np.ndarray:
            k = capacity[0]
            x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device)
            self.model.eval()
            with torch.no_grad():
                out = self.model.forward_at_k(x_tensor, k)
            mean = out[:, 0].cpu().numpy()
            log_var = out[:, 1].cpu().numpy()
            return 0.5 * (log_var + (y_val_arr - mean) ** 2 / np.exp(log_var))

        router = DistilledCapacityRouter(device=self.device)
        router.fit(eval_fn=eval_fn, x_val=x_val, y_val=y_val, capacity_grid=capacity_grid, tolerance=tolerance, cost_fn=cost_fn)
        self.capacity_router_ = router
        return router

    def fit_global_selector(
        self,
        x_val: np.ndarray,
        y_val: np.ndarray,
        capacity_grid: list[int] | None = None,
        n_bootstrap: int = 1000,
    ) -> int:
        """`CapacitySelection.GLOBAL_CHEAP` (M1): picks ONE k for the whole dataset.

        Reads a held-out `(x_val, y_val)` set (capacity-programme Task PA). Built on
        `NestedStrategy.all_rung_log_likelihood`
        (`automl_package/models/selection_strategies/n_classes_strategies.py:230`) -- the per-rung
        `(batch, n_classes)` held-out log-likelihood table -- NEVER on
        `held_out_arbiter_advantage`, a different, per-input readout that cannot answer a global
        "which k" question (`docs/plans/capacity_programme/probreg.md` §1 ⚠️, D5).

        Selection rule: the smallest (cheapest) k whose held-out score is not meaningfully worse
        than the best, "meaningfully" = exceeds twice a bootstrap-estimated standard error of the
        paired difference -- `automl_package.utils.capacity_selection.cheapest_within_tolerance`
        (capacity-programme FP-9.a; rule of record
        `docs/reports/probreg_kselection/probreg_kselection.md` §3.2). `fit_sweep_selector` (M3)
        applies the SAME rule and the SAME primitive to its own curve, so the two are selected
        consistently (though not on directly comparable units -- see that method's docstring).

        Args:
            x_val: held-out inputs, disjoint from whatever `fit()` trained on.
            y_val: held-out targets, in the SAME space as `fit()`'s `y_train`/`y_val` -- i.e.
                raw/original target units. Auto-transformed internally when
                `target_transform="symlog"` (D2's units-mismatch pattern, applied here exactly as
                `fit_router`/`_per_sample_log_likelihood_at_k` apply it -- `all_rung_log_likelihood`
                itself does NOT transform `y_target`, see its docstring) -- do NOT pre-transform it
                yourself.
            capacity_grid: candidate k values, cheapest first. Defaults to every rung
                `1..max_n_classes_for_probabilistic_path`.
            n_bootstrap: bootstrap resamples for the selection rule's standard-error estimate.

        Returns:
            The selected k (also stored as `self.selected_k_`, and the full curve as
            `self.global_selector_curve_`).

        Raises:
            RuntimeError: the model has not been fitted, or was not trained with
                `n_classes_selection_method=NESTED`.
        """
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        if self.n_classes_selection_method != NClassesSelectionMethod.NESTED:
            raise RuntimeError(
                "fit_global_selector requires n_classes_selection_method=NESTED (the certified "
                f"prefix-nesting property; got {self.n_classes_selection_method.value}). See docstring."
            )
        if capacity_grid is None:
            capacity_grid = list(range(1, self.max_n_classes_for_probabilistic_path + 1))

        y_arr = np.asarray(y_val, dtype=np.float64)
        if self.target_transform == "symlog":
            # all_rung_log_likelihood does not transform y_target itself (see its docstring,
            # D2/D6) -- the caller must, exactly like fit_router/_per_sample_log_likelihood_at_k.
            y_arr = symlog(torch.tensor(y_arr, dtype=torch.float32)).numpy().astype(np.float64)

        x_arr = x_val.values if hasattr(x_val, "values") else np.asarray(x_val)
        x_tensor = torch.tensor(x_arr, dtype=torch.float32, device=self.device)
        y_tensor = torch.tensor(y_arr, dtype=torch.float32, device=self.device)
        self.model.eval()
        with torch.no_grad():
            ll_table = self.model.n_classes_strategy.all_rung_log_likelihood(x_tensor, y_tensor).cpu().numpy()

        ll_selected = ll_table[:, [k - 1 for k in capacity_grid]]  # (N, len(capacity_grid))
        error_table = -ll_selected  # cheapest_within_tolerance wants error (lower is better)
        idx = cheapest_within_tolerance(error_table, n_boot=n_bootstrap, seed=self.random_seed or 0)
        selected_k = capacity_grid[idx]

        self.selected_k_ = selected_k
        self.global_selector_curve_ = {
            "capacity_grid": list(capacity_grid),
            "mean_log_likelihood": ll_selected.mean(axis=0).tolist(),
            "n_selection": len(y_arr),
            "selected_k": selected_k,
        }
        return selected_k

    def fit_sweep_selector(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        capacity_grid: list[int] | None = None,
        n_bootstrap: int = 1000,
    ) -> int:
        """`CapacitySelection.GLOBAL_SWEEP` (M3): trains one ORDINARY model per candidate k.

        Each is `NClassesSelectionMethod.NONE`, scored on held-out `(x_val, y_val)`; the winner
        is kept (capacity-programme Task PA). Generalises `report_a_benchmark.select_k_for_toy`
        (`automl_package/examples/report_a_benchmark.py:331`) into a package method: unlike that
        script's argmin-val-NLL rule, this applies the strand's own cheapest-within-tolerance rule
        (`docs/plans/capacity_programme/probreg.md` §1) via
        `automl_package.utils.capacity_selection.cheapest_within_tolerance` -- the SAME primitive
        `fit_global_selector` (M1) uses, so the two are selected consistently.

        Each sub-model is trained ORDINARILY at its own fixed k
        (`n_classes_selection_method=NONE`) -- never with k-dropout -- matching §1's ruling that a
        per-k model trained with dropout across the whole ladder "is not dedicated to anything...
        and cannot serve as an independent reference". Every sub-model is built from
        `self.get_params()` with `n_classes`/`n_classes_selection_method`/`capacity_selection`
        overridden, plus a short list of fields re-stated explicitly at the call site below.
        **Correction, root-verified 2026-07-21:** an earlier version of this docstring claimed
        `get_params()` "does not round-trip" six fields and that this was a pre-existing
        silent-wrong-answer bug. That was checked and is **FALSE for five of the six** --
        `target_transform`, `prob_reg_loss_type`, `use_anchored_heads`, `loss_type` and `beta` are
        all present in `get_params()` and survive `_clone()` intact. Only
        `calculate_feature_importance` is genuinely absent, and dropping it from a clone is
        arguably right (it is a diagnostic switch, not a model hyper-parameter). No CV/HPO bug
        follows, and none is filed.
        Each is trained on the FULL
        `(x_train, y_train)` this method receives -- expensive by design (`probreg.md` §1: M3 is
        the reference the cheap arms are measured against, exempt from the matched-cost budget).

        Scored via per-sample RAW-space Gaussian NLL, computed from each sub-model's own
        `predict`/`predict_uncertainty` (which already invert `target_transform`). UNLIKE M1's
        curve (which is always in whatever space the model was fit in), this one is therefore
        always in raw target units -- the two curves are not numerically comparable rung-for-rung,
        only each model's own selected k is reported.

        Args:
            x_train: data every per-k sub-model is trained on.
            y_train: targets every per-k sub-model is trained on.
            x_val: held-out inputs the per-k curve is scored on.
            y_val: held-out targets the per-k curve is scored on.
            capacity_grid: candidate k values, cheapest first. Defaults to every rung
                `1..max_n_classes_for_probabilistic_path` (the SAME default `fit_global_selector`
                uses, so M1 and M3 answer "which k" over the same grid -- §1's "(b) same choice"
                claim needs that).
            n_bootstrap: bootstrap resamples for the selection rule's standard-error estimate.

        Returns:
            The selected k (also stored as `self.selected_k_`; the winning sub-model is stored as
            `self._sweep_submodel_` and is what `predict`/`predict_uncertainty` delegate to under
            `CapacitySelection.GLOBAL_SWEEP`).
        """
        if capacity_grid is None:
            capacity_grid = list(range(1, self.max_n_classes_for_probabilistic_path + 1))

        base_params = self.get_params()
        # Explicit, not compensatory. VERIFIED AT THE ROOT 2026-07-21: `get_params()` DOES
        # round-trip `target_transform`, `prob_reg_loss_type`, `use_anchored_heads`, `loss_type`
        # and `beta` (checked by constructing with non-defaults and reading `_clone()` back --
        # e.g. symlog survives the clone). Only `calculate_feature_importance` is genuinely
        # absent, and that one matters here beyond correctness: left at `BaseModel`'s `True`
        # default, every per-k sub-model would run SHAP unasked, multiplying an
        # already-expensive-by-design sweep. Re-stating the other five is harmless and keeps this
        # sweep's contract explicit at the call site.
        for field in ("target_transform", "prob_reg_loss_type", "use_anchored_heads", "loss_type", "beta", "calculate_feature_importance"):
            base_params[field] = getattr(self, field)

        y_val_arr = np.asarray(y_val, dtype=np.float64).ravel()
        error_table = np.empty((len(y_val_arr), len(capacity_grid)), dtype=np.float64)
        submodels: list[ProbabilisticRegressionModel] = []
        for col, k in enumerate(capacity_grid):
            params = dict(base_params)
            params.update(n_classes=k, n_classes_selection_method=NClassesSelectionMethod.NONE, capacity_selection=CapacitySelection.FIXED)
            sub_model = ProbabilisticRegressionModel(**params)
            sub_model.fit(x_train, y_train)
            y_pred = np.asarray(sub_model.predict(x_val), dtype=np.float64).ravel()
            y_std = np.asarray(sub_model.predict_uncertainty(x_val), dtype=np.float64).ravel()
            variance = np.maximum(y_std**2, 1e-9)
            error_table[:, col] = 0.5 * (np.log(2.0 * np.pi * variance) + (y_val_arr - y_pred) ** 2 / variance)
            submodels.append(sub_model)

        idx = cheapest_within_tolerance(error_table, n_boot=n_bootstrap, seed=self.random_seed or 0)
        selected_k = capacity_grid[idx]

        self.selected_k_ = selected_k
        self._sweep_submodel_ = submodels[idx]
        self.global_selector_curve_ = {
            "capacity_grid": list(capacity_grid),
            "mean_nll": error_table.mean(axis=0).tolist(),
            "n_selection": len(y_val_arr),
            "selected_k": selected_k,
        }
        return selected_k

    def _per_sample_log_likelihood_at_k(self, x: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
        """Per-sample Gaussian log-likelihood of rung `k`'s forced readout (bypasses selection).

        Args:
            x: inputs, `(N, in_dim)`.
            y: targets in the SAME space as `fit()`'s `y_train`/`y_val` -- i.e. raw/original
                target units. When `target_transform="symlog"`, `y` is auto-transformed
                internally before scoring, to match `forward_at_k`'s symlog-space outputs; do
                NOT pre-transform it yourself. Mirrors `fit_router`'s contract exactly.
            k: rung to force.

        Returns:
            `(N,)` per-sample Gaussian log-likelihood.
        """
        y_arr = np.asarray(y, dtype=np.float64)
        if self.target_transform == "symlog":
            # forward_at_k's outputs live in symlog space (fit() transforms y_train/y_val before
            # training, :526-529) -- score against y in that same space, or the per-sample
            # likelihood is silently wrong (same units mismatch as fit_router's F9-fix-b, D1).
            y_arr = symlog(torch.tensor(y_arr, dtype=torch.float32)).numpy().astype(np.float64)

        x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device)
        self.model.eval()
        with torch.no_grad():
            out = self.model.forward_at_k(x_tensor, k)
        mean = out[:, 0].cpu().numpy()
        log_var = out[:, 1].cpu().numpy()
        return -0.5 * (np.log(2.0 * np.pi) + log_var + (y_arr - mean) ** 2 / np.exp(log_var))

    def held_out_arbiter_advantage(
        self,
        x: np.ndarray,
        y: np.ndarray,
        width: float = 0.075,
        top_k: int | None = None,
    ) -> np.ndarray:
        """Certified per-input capacity readout: the held-out ARBITER A(x) -- NOT the per-input knee.

        R2 (`automl_package/examples/capacity_ladder_results/R2_verdict.md`) certified that the
        per-input latent component count is recovered via this neighbour-averaged ARBITER --
        ``A(x) = mean_{x' near x}[ ll_top_k(x', y') - ll_bypass(x', y') ]``, the held-out
        advantage of the top rung (`top_k`, default `max_n_classes_for_probabilistic_path`)
        over the k=1 direct-regression bypass -- while the per-input hard KNEE (walking the
        ladder rung by rung, stopping at the first non-improving step,
        `capacity_ladder_k5.perinput_knee_curve`) was found NOT faithful: "noisy and wrong on
        D" (R2 §1). This is the readout F7/F10 point to instead of re-deriving it. COPIED
        (not imported) from `_capacity_ladder.perinput_curve`'s box-car neighbourhood average --
        package code under `models/` does not depend on `examples/` (see
        `models/common/distilled_router.py` module docstring for the same convention).

        Known boundary (state, do not hide): certified on toy D (fixed-mode staircase, monotone
        recovery on all 3 seeds); toy-E-like MOVING-mode drift cases are seed-fragile (2 of 3
        seeds negative, R2 §1) -- NOT certified. Requires `n_classes_selection_method=NESTED`
        for the same nesting reason `fit_router` does (see its docstring). The neighbourhood
        here is a Euclidean box-car, which reduces to the certified 1-D box-car when `x` is
        scalar; multi-dimensional `x` is an unaudited generalization.

        Args:
            x: held-out inputs, `(N,)` or `(N, in_dim)`.
            y: held-out targets, `(N,)`, in the SAME space as `fit()`'s `y_train`/`y_val` --
                i.e. raw/original target units. When `target_transform="symlog"`, `y` is
                auto-transformed internally (via `_per_sample_log_likelihood_at_k`) before
                scoring, to match `forward_at_k`'s symlog-space outputs; do NOT pre-transform it
                yourself. Mirrors `fit_router`'s contract exactly so callers can pass the same
                `y` array to both.
            width: neighbourhood box-car half-width, in units of `x` (Euclidean distance if
                `in_dim > 1`). Default 0.075 matches the certified toy-suite read
                (`capacity_ladder_k5.WIDTH`) -- re-tune for feature scales outside `[0, 1]`.
            top_k: top rung compared against the bypass. Defaults to
                `max_n_classes_for_probabilistic_path`.

        Returns:
            `(N,)` neighbour-averaged advantage A(x) at every row of `x` (positive = the top
            rung beats the bypass in that neighbourhood; NaN where no neighbour falls within
            `width`).

        Raises:
            RuntimeError: the model has not been fitted, or was not trained with
                `n_classes_selection_method=NESTED`.
        """
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        if self.n_classes_selection_method != NClassesSelectionMethod.NESTED:
            raise RuntimeError(
                "held_out_arbiter_advantage requires n_classes_selection_method=NESTED (the certified "
                f"prefix-nesting property; got {self.n_classes_selection_method.value}). See docstring."
            )
        top_k = top_k or self.max_n_classes_for_probabilistic_path

        x_arr = np.asarray(x, dtype=np.float64)
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(-1, 1)
        y_arr = np.asarray(y, dtype=np.float64).ravel()

        ll_bypass = self._per_sample_log_likelihood_at_k(x_arr, y_arr, 1)
        ll_top = self._per_sample_log_likelihood_at_k(x_arr, y_arr, top_k)
        delta = ll_top - ll_bypass  # (N,)

        dist = np.linalg.norm(x_arr[None, :, :] - x_arr[:, None, :], axis=-1)  # (N, N)
        mask = (dist <= width).astype(np.float64)
        counts = mask.sum(axis=1)
        summed = mask @ delta
        with np.errstate(invalid="ignore", divide="ignore"):
            out = summed / counts
        out[counts == 0] = np.nan
        return out

    @staticmethod
    def _symlog_mc_moments(mean_symlog: np.ndarray, std_symlog: np.ndarray, n_samples: int = 200, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """Monte Carlo mean/std in original space for a Gaussian in symlog space.

        Samples ``n_samples`` points from ``N(mean_symlog, std_symlog²)``, applies
        ``symexp``, and returns the empirical (mean, std) in original space.
        """
        rng = np.random.default_rng(seed)
        mean_t = torch.from_numpy(np.asarray(mean_symlog, dtype=np.float32))
        std_t = torch.from_numpy(np.asarray(std_symlog, dtype=np.float32))
        noise = torch.from_numpy(rng.standard_normal((n_samples, *mean_t.shape)).astype(np.float32))
        samples_symlog = mean_t.unsqueeze(0) + std_t.unsqueeze(0) * noise
        samples_orig = symexp(samples_symlog)
        mean_orig = samples_orig.mean(dim=0).numpy()
        std_orig = samples_orig.std(dim=0).numpy()
        return mean_orig, std_orig

    def _update_params(self, params: dict[str, Any]) -> None:
        """Updates the model's parameters from a given dictionary."""
        super()._update_params(params)
        # Handle nested params
        base_classifier_params = {}
        regression_head_params = {}
        direct_regression_head_params = {}

        for key, value in params.items():
            if key.startswith("base_classifier_params__"):
                param_name = key.split("__", 1)[1]
                base_classifier_params[param_name] = value
            elif key.startswith("regression_head_params__"):
                param_name = key.split("__", 1)[1]
                regression_head_params[param_name] = value
            elif key.startswith("direct_regression_head_params__"):
                param_name = key.split("__", 1)[1]
                direct_regression_head_params[param_name] = value
            else:
                if key == "regression_strategy" and isinstance(value, str):
                    setattr(self, key, RegressionStrategy[value.upper()])
                elif key == "n_classes_selection_method" and isinstance(value, str):
                    setattr(self, key, NClassesSelectionMethod[value.upper()])
                elif key == "optimization_strategy" and isinstance(value, str):
                    setattr(self, key, ProbabilisticRegressionOptimizationStrategy[value.upper()])
                else:
                    setattr(self, key, value)

        if base_classifier_params:
            self.base_classifier_params.update(base_classifier_params)
        if regression_head_params:
            self.regression_head_params.update(regression_head_params)
        if direct_regression_head_params:
            self.direct_regression_head_params.update(direct_regression_head_params)

    def get_shap_explainer_info(self) -> dict[str, Any]:
        """Gets the SHAP explainer type and the model to be explained.

        Dynamic n_classes strategies dispatch per-input through different heads
        (gather/scatter ops, batch repartitioning), which breaks ``shap.DeepExplainer``'s
        DeepLIFT gradient tracing. Fall back to ``KernelExplainer`` (model-agnostic,
        slower but correct) when dynamic-k selection is enabled.
        """
        if self.n_classes_selection_method != NClassesSelectionMethod.NONE:
            return {"explainer_type": ExplainerType.KERNEL, "model": self.predict}
        return {"explainer_type": ExplainerType.DEEP, "model": self.get_internal_model()}

    def _clone(self) -> "ProbabilisticRegressionModel":
        """Creates a new instance of the model with the same parameters."""
        return self.__class__(**self.get_params())

    def get_params(self) -> dict[str, Any]:
        """Gets parameters for this estimator."""
        params = super().get_params()
        params.update(
            {
                "n_classes": self.n_classes,
                "n_classes_inf": self.n_classes_inf,
                "max_n_classes_for_probabilistic_path": self.max_n_classes_for_probabilistic_path,
                "base_classifier_params": self.base_classifier_params,
                "regression_head_params": self.regression_head_params,
                "direct_regression_head_params": self.direct_regression_head_params,
                "regression_strategy": self.regression_strategy,
                "n_classes_selection_method": self.n_classes_selection_method,
                "gumbel_tau": self.gumbel_tau,
                "n_classes_predictor_learning_rate": self.n_classes_predictor_learning_rate,
                "optimization_strategy": self.optimization_strategy,
                "boundary_regularization_method": self.boundary_regularization_method,
                "boundary_loss_weight": self.boundary_loss_weight,
                "use_monotonic_constraints": self.use_monotonic_constraints,
                "constrain_middle_class": self.constrain_middle_class,
            }
        )
        return params

    def get_classifier_predictions(self, x: np.ndarray | pd.DataFrame, y_true_original: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns the internal classifier's predicted classes, probabilities, and.

        the corresponding (discretized) true labels for this composite model.

        Args:
            x (np.ndarray): Feature matrix.
            y_true_original (np.ndarray): Original true labels (will be discretized internally).

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                - Predicted classes from the internal classifier.
                - Predicted probabilities from the internal classifier.
                - Discretized true labels corresponding to the internal classifier's task.
        """
        if self.direct_regression:
            raise NotImplementedError("get_classifier_predictions is not available in direct regression mode.")
        if self.model is None or self.precomputed_class_boundaries is None:
            raise RuntimeError("Model has not been fitted yet or class boundaries were not computed.")

        self.model.eval()
        x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            # The model's forward pass now returns predictions, classifier_logits_out, and selected_k_values
            _, returned_classifier_logits, selected_k_values_tensor, _, _ = self.model(x_tensor)

            selected_k_values = selected_k_values_tensor.cpu().numpy()
            probabilistic_indices = np.where(selected_k_values != DIRECT_REGRESSION_K_SENTINEL)[0]

            if self.n_classes_selection_method != NClassesSelectionMethod.NONE and len(probabilistic_indices) == 0:
                raise NotImplementedError("No probabilistic predictions were made for the given X. All samples went to direct regression.")

            y_flat = y_true_original.flatten() if y_true_original.ndim > 1 else y_true_original
            y_true_discretized = np.full_like(y_flat, -1, dtype=int)  # Default to -1 (for direct regression)

            # Discretize y_true using the pre-computed boundaries from the training set by grouping samples by k
            probabilistic_k_values = selected_k_values[probabilistic_indices]
            unique_k_values = np.unique(probabilistic_k_values)

            for k_val in unique_k_values:
                k = int(k_val)
                # Create a mask for samples corresponding to the current k
                mask = probabilistic_k_values == k
                # Get the original indices for these samples
                original_indices = probabilistic_indices[mask]

                if len(original_indices) > 0:
                    assert k in self.precomputed_class_boundaries
                    boundaries = self.precomputed_class_boundaries[k]
                    # Discretize all samples for this k at once
                    _, discretized_labels = create_bins(data=y_flat[original_indices], unique_bin_edges=boundaries)
                    y_true_discretized[original_indices] = discretized_labels

            # Use the returned_classifier_logits directly
            classifier_logits_for_proba = returned_classifier_logits[probabilistic_indices]

            # Re-apply softmax to get probabilities for the probabilistic samples
            # Need to re-mask and softmax as the stored logits might be from the full max_n_classes_allowed
            k_values = torch.tensor(selected_k_values[probabilistic_indices], device=classifier_logits_for_proba.device).long()
            max_k = classifier_logits_for_proba.shape[1]
            col_indices = torch.arange(max_k, device=classifier_logits_for_proba.device)
            mask = col_indices < k_values.unsqueeze(1)
            masked_classifier_logits = torch.where(mask, classifier_logits_for_proba, float("-inf"))

            # Single masked-softmax path for every k >= 2 (no k==2 special case: the binary
            # softmax positive probability is sigmoid(logit1 - logit0), not sigmoid(logit0)).
            y_proba_internal = torch.softmax(masked_classifier_logits, dim=1).cpu().numpy()

            y_pred_internal = np.argmax(y_proba_internal, axis=1)

        return y_pred_internal, y_proba_internal, y_true_discretized

    def plot_probability_mappers(self, plot_path: str = "probability_mappers.png") -> None:
        """Plots the functions that map class probabilities to regression values."""
        if not self.model:
            logger.warning("No model found. Please fit the model first.")
        elif self.precomputed_class_boundaries is None:
            logger.warning("Class boundaries not computed. Please fit the model first.")
        else:
            # Use the pre-computed class boundaries for the fixed n_classes case
            self.class_boundaries = self.precomputed_class_boundaries[self.n_classes]

            plot_nn_probability_mappers(
                mapper_model=self.model.regression_module,
                regression_strategy=self.regression_strategy,
                n_classes=self.n_classes,
                class_boundaries=self.class_boundaries,
                device=self.device,
                plot_path=plot_path,
                model_name=self.name,
            )

    def get_num_parameters(self) -> int:
        """Returns the total number of trainable parameters in the model.

        Returns:
            int: The total number of parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
