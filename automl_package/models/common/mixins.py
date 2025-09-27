"""Common mixin classes for models."""

import numpy as np
import torch
import torch.nn as nn

from automl_package.enums import LearnedRegularizationType, Monotonicity
from automl_package.models.common.regression_heads import ConstantHead
from automl_package.utils.losses import boundary_regularization_loss
from automl_package.utils.numerics import calculate_binned_stats


class BoundaryLossMixin:
    """A mixin for calculating boundary regularization loss."""

    def calculate_middle_class_nll_penalty(
        self,
        heads: nn.ModuleList,
        per_head_outputs: torch.Tensor,
        y_binned_tensor: torch.Tensor,
        middle_class_dist_params: dict | None = None,
    ) -> torch.Tensor:
        """Calculates the NLL penalty for the middle class."""
        nll_penalty = torch.tensor(0.0)
        if not (middle_class_dist_params is None or not (len(heads) % 2 != 0)):
            middle_class_idx = (len(heads) - 1) / 2.0
            middle_class_mean = middle_class_dist_params["mean"]
            middle_class_std = middle_class_dist_params["std"]

            if middle_class_std > 0:
                # This loss should only apply to samples that are actually in the middle class
                middle_class_mask = y_binned_tensor == middle_class_idx
                if middle_class_mask.any():
                    head_outputs_for_middle_class_samples = per_head_outputs[middle_class_mask, int(middle_class_idx)]
                    dist = torch.distributions.Normal(middle_class_mean, middle_class_std)
                    nll_penalty = -dist.log_prob(head_outputs_for_middle_class_samples).mean()
        return nll_penalty

    def calculate_boundary_loss(
        self,
        heads: nn.ModuleList,
        per_head_outputs: torch.Tensor,
        y_true: torch.Tensor,
        y_binned_tensor: torch.Tensor,
        class_value_ranges: torch.Tensor,
        boundary_loss_weight: float,
    ) -> torch.Tensor:
        """Calculates the boundary regularization loss by applying the optimal strategy for each head type."""
        total_boundary_loss = 0.0

        for i, head in enumerate(heads):
            head_loss = 0.0
            # Get the allowed value range for the true class bin for the entire batch
            sample_boundaries = class_value_ranges[y_binned_tensor]

            if head.monotonic_constraint != Monotonicity.NONE:
                # Case 1: Monotonic Head - check only endpoints p=0 and p=1 against the most restrictive bounds in the batch
                p_endpoints = torch.tensor([[0.0], [1.0]], device=y_true.device)
                endpoint_outputs = head(p_endpoints)

                # For an increasing head, h(0) must be >= max(all lower bounds) and h(1) must be <= min(all upper bounds)
                # For a decreasing head, h(0) must be <= min(all upper bounds) and h(1) must be >= max(all lower bounds)
                most_restrictive_lower_bound = torch.max(sample_boundaries[:, 0])
                most_restrictive_upper_bound = torch.min(sample_boundaries[:, 1])

                # Create a single boundary tensor [max_lower, min_upper] to check against
                restrictive_bounds = torch.stack([most_restrictive_lower_bound, most_restrictive_upper_bound]).unsqueeze(0)

                head_loss += boundary_regularization_loss(endpoint_outputs, restrictive_bounds)

            elif isinstance(head, ConstantHead):
                # Case 2: Constant Head - check the single learned value
                constant_output = head(torch.zeros(1, 1, device=y_true.device))  # Input value doesn't matter
                head_loss = boundary_regularization_loss(constant_output, sample_boundaries)

            else:
                # Case 3: Standard Head - check all points in the batch
                head_outputs_for_true_bin = per_head_outputs[torch.arange(len(y_binned_tensor)), y_binned_tensor]
                head_loss = boundary_regularization_loss(head_outputs_for_true_bin, sample_boundaries)

            total_boundary_loss += head_loss

        # Average the loss across the heads
        avg_boundary_loss = total_boundary_loss / len(heads)

        return boundary_loss_weight * avg_boundary_loss


class BinnedUncertaintyMixin:
    """A mixin for handling binned residual standard deviation uncertainty."""

    def _init_binned_uncertainty(self) -> None:
        """Initializes the attributes for binned uncertainty."""
        self._binned_uncertainty_lookup: dict[int, float] = {}
        self._binned_uncertainty_edges: np.ndarray | None = None

    def calibrate_uncertainty(self, x: np.ndarray, y: np.ndarray, n_bins: int = 10) -> None:
        """Calibrates the uncertainty model by binning residuals against predictions."""
        predictions = self.predict(x)
        residuals = y.flatten() - predictions.flatten()

        bin_edges, stats = calculate_binned_stats(probas=predictions, values=residuals, n_bins=n_bins, aggregations={"std": np.std}, min_value=-np.inf, max_value=np.inf)

        self._binned_uncertainty_edges = bin_edges
        self._binned_uncertainty_lookup = stats["std"]

    def predict_binned_uncertainty(self, x: np.ndarray) -> np.ndarray:
        """Predicts uncertainty by looking up the calibrated binned standard deviations."""
        if self._binned_uncertainty_edges is None or not self._binned_uncertainty_lookup:
            raise RuntimeError("Uncertainty model has not been calibrated. Call `calibrate_uncertainty` first.")

        predictions = self.predict(x)
        # Find which bin each new prediction falls into
        bin_indices = np.searchsorted(self._binned_uncertainty_edges[1:], predictions.flatten(), side="left")
        bin_indices = np.clip(bin_indices, 0, len(self._binned_uncertainty_edges) - 2)

        # Look up the standard deviation
        stds = np.array([self._binned_uncertainty_lookup[i] for i in bin_indices])
        return np.nan_to_num(stds, nan=np.nanmean(list(self._binned_uncertainty_lookup.values())))


class RegularizationMixin:
    """A mixin for handling regularization parameter setup."""

    def _setup_regularization_parameters(self) -> None:
        """Initializes the log lambda nn.Parameters for learnable regularization."""
        self.l1_log_lambda, self.l2_log_lambda = None, None
        if self.learn_regularization_lambdas:
            if self.learned_regularization_type in [LearnedRegularizationType.L1_ONLY, LearnedRegularizationType.L1_L2]:
                self.l1_log_lambda = nn.Parameter(torch.tensor(np.log(1e-4), dtype=torch.float32))
            if self.learned_regularization_type in [LearnedRegularizationType.L2_ONLY, LearnedRegularizationType.L1_L2]:
                self.l2_log_lambda = nn.Parameter(torch.tensor(np.log(1e-4), dtype=torch.float32))

    def _setup_lambda_optimizer(self) -> None:
        """Initializes the optimizer for the learnable regularization lambdas."""
        self.lambda_optimizer = None
        if self.learn_regularization_lambdas:
            lambda_params = [p for p in [self.l1_log_lambda, self.l2_log_lambda] if p is not None]
            if lambda_params:
                self.lambda_optimizer = torch.optim.Adam(lambda_params, lr=self.lambda_learning_rate)
