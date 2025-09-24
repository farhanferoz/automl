import torch
import numpy as np

from automl_package.utils.losses import boundary_regularization_loss
from automl_package.utils.numerics import create_bins


class BoundaryLossMixin:
    """A mixin for calculating boundary regularization loss."""

    def calculate_boundary_loss(
        self, per_head_outputs: torch.Tensor, y_true: torch.Tensor, y_binned_tensor: torch.Tensor, class_value_ranges: torch.Tensor, boundary_loss_weight: float,
    ) -> torch.Tensor:
        """Calculates the boundary regularization loss on per-head outputs.

        Args:
            per_head_outputs: The output from each regression head. Shape: (batch, n_classes, output_size).
            y_true: The true regression values.
            y_binned_tensor: The binned (classified) true values.
            class_value_ranges: The allowed value ranges for each class.
            boundary_loss_weight: The weight to apply to the final boundary loss.

        Returns:
            The calculated boundary loss, multiplied by its weight.
        """
        # Select the output from the head corresponding to the true class bin
        head_outputs_for_true_bin = per_head_outputs[torch.arange(len(y_binned_tensor)), y_binned_tensor]

        # Get the allowed value range for the true class bin
        sample_boundaries = class_value_ranges[y_binned_tensor]

        boundary_loss = boundary_regularization_loss(head_outputs_for_true_bin, sample_boundaries)
        return boundary_loss_weight * boundary_loss


class BinnedUncertaintyMixin:
    """A mixin for handling binned residual standard deviation uncertainty."""

    def _init_binned_uncertainty(self):
        """Initializes the attributes for binned uncertainty."""
        self._binned_uncertainty_lookup: dict[int, float] = {}
        self._binned_uncertainty_edges: np.ndarray | None = None

    def calibrate_uncertainty(self, x: np.ndarray, y: np.ndarray, n_bins: int = 10):
        """Calibrates the uncertainty model by binning residuals against predictions."""
        predictions = self.predict(x)
        residuals = y.flatten() - predictions.flatten()

        bin_edges, stats = calculate_binned_stats(
            probas=predictions, values=residuals, n_bins=n_bins, aggregations={"std": np.std}, min_value=-np.inf, max_value=np.inf
        )

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

    def _setup_regularization_parameters(self):
        """Initializes the log lambda nn.Parameters for learnable regularization."""
        self.l1_log_lambda, self.l2_log_lambda = None, None
        if self.learn_regularization_lambdas:
            if self.learned_regularization_type in [LearnedRegularizationType.L1_ONLY, LearnedRegularizationType.L1_L2]:
                self.l1_log_lambda = nn.Parameter(torch.tensor(np.log(1e-4), dtype=torch.float32))
            if self.learned_regularization_type in [LearnedRegularizationType.L2_ONLY, LearnedRegularizationType.L1_L2]:
                self.l2_log_lambda = nn.Parameter(torch.tensor(np.log(1e-4), dtype=torch.float32))

    def _setup_lambda_optimizer(self):
        """Initializes the optimizer for the learnable regularization lambdas."""
        self.lambda_optimizer = None
        if self.learn_regularization_lambdas:
            lambda_params = [p for p in [self.l1_log_lambda, self.l2_log_lambda] if p is not None]
            if lambda_params:
                self.lambda_optimizer = torch.optim.Adam(lambda_params, lr=self.lambda_learning_rate)
