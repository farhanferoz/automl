"""Base class for selection strategies in neural networks."""

from abc import ABC, abstractmethod
from typing import Any

import torch

from automl_package.enums import ProbabilisticRegressionOptimizationStrategy, RegressionStrategy

# Sentinel value for direct regression mode (must exceed any valid n_classes_inf).
# Loss function checks `selected_k_values < n_classes_inf` to identify probabilistic samples.
DIRECT_REGRESSION_K_SENTINEL = 2**30


class BaseSelectionStrategy(ABC):
    """Abstract base class for all selection strategies (layer, n_classes, etc.)."""

    def __init__(self, model_instance: Any) -> None:
        """Initializes the BaseSelectionStrategy.

        Args:
            model_instance (Any): The model instance this strategy is associated with.
        """
        self.model = model_instance
        self.policy_optimizer: torch.optim.Optimizer | None = None
        self.mode_selection_probs: torch.Tensor | None = None  # To store probabilities for external use (e.g., classifier_logits_out)

    @abstractmethod
    def forward(
        self, x_input: torch.Tensor, logits: torch.Tensor | None
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor,
        torch.Tensor | None,
    ]:
        """Processes input and logits to determine selection and compute predictions.

        Returns: (final_predictions, n_actual_for_logging, n_probs_for_logging, n_logits_for_logging, log_prob_for_reinforce).
        """

    @abstractmethod
    def setup_optimizers(self, policy_params: Any) -> None:
        """Sets up optimizers for policy-based strategies (e.g., REINFORCE)."""
        self.policy_optimizer = torch.optim.Adam(policy_params, lr=self.model.n_predictor_learning_rate)

    @abstractmethod
    def on_epoch_end(self, **kwargs: Any) -> None:
        """Hook for epoch-end operations (e.g., REINFORCE policy updates)."""
        # Default implementation does nothing

    _DIRECT_REGRESSION_K_SENTINEL = DIRECT_REGRESSION_K_SENTINEL

    def _compute_per_head_outputs_full_k(
        self, classifier_raw_logits: torch.Tensor, boundaries: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Per-head outputs at k=n_classes, for ordering/MDN/middle-class penalties.

        Dynamic-k strategies call the regression heads inside a weighted-sum over
        modes with mode-dependent probability inputs; that per-mode view is not a
        coherent readout for head-structure penalties (ordering, MDN, middle-class,
        boundary loss). This helper mirrors NoneStrategy's fixed-k behaviour: mask
        to the first n_classes logits, softmax, feed through the regression module
        with ``return_head_outputs=True``. SINGLE_HEAD_FINAL_OUTPUT has no per-head
        structure, so None is returned and the loss-side guards become no-ops.
        """
        if self.model.regression_strategy == RegressionStrategy.SINGLE_HEAD_FINAL_OUTPUT:
            return None

        n_classes = self.model.n_classes
        masked = torch.full_like(classifier_raw_logits, float("-inf"))
        masked[:, :n_classes] = classifier_raw_logits[:, :n_classes]
        probabilities = torch.softmax(masked, dim=1)

        if self.model.optimization_strategy in (
            ProbabilisticRegressionOptimizationStrategy.CE_STOP_GRAD,
            ProbabilisticRegressionOptimizationStrategy.GRADIENT_STOP,
        ):
            probabilities = probabilities.detach()

        _, per_head_outputs = self.model.regression_module(
            probabilities, return_head_outputs=True, boundaries=boundaries,
        )
        return per_head_outputs

    # Helper for weighted average logic (common to GumbelSoftmax and SoftGating)
    def _weighted_average_logic(
        self, x_input: torch.Tensor, mode_selection_probs: torch.Tensor, boundaries: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        classifier_raw_logits = self.model.classifier_layers(x_input)
        final_predictions_contribution = torch.zeros(x_input.size(0), self.model.regression_output_size).to(x_input.device)

        inf_sentinel = self._DIRECT_REGRESSION_K_SENTINEL
        selected_k_indices = torch.argmax(mode_selection_probs, dim=1)
        selected_k_values = torch.where(
            selected_k_indices == mode_selection_probs.size(1) - 1,
            torch.tensor(inf_sentinel, dtype=torch.long).to(x_input.device),
            selected_k_indices + 2,
        )

        for i in range(mode_selection_probs.size(1)):
            prob_i = mode_selection_probs[:, i].unsqueeze(1)

            if i == mode_selection_probs.size(1) - 1:
                predictions_for_mode = self.model.direct_regression_head(x_input)
            else:
                k_val = i + 2
                predictions_for_mode = self.model._compute_predictions_for_k(classifier_raw_logits, k_val, boundaries=boundaries)

            final_predictions_contribution += prob_i * predictions_for_mode

        return final_predictions_contribution, selected_k_values, classifier_raw_logits

    # Helper for hard selection logic (common to STE and REINFORCE)
    def _hard_selection_logic(
        self, x_input: torch.Tensor, mode_selection_one_hot: torch.Tensor, boundaries: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Applies hard selection using weighted-sum pattern to preserve STE gradients.

        Uses prob * output for each mode so that gradients flow through mode_selection_one_hot
        back to the n_classes_predictor, matching the pattern in _weighted_average_logic and
        the layer_selection SteStrategy.
        """
        classifier_raw_logits = self.model.classifier_layers(x_input)
        final_predictions_contribution = torch.zeros(x_input.size(0), self.model.regression_output_size).to(x_input.device)

        inf_sentinel = self._DIRECT_REGRESSION_K_SENTINEL
        selected_k_indices = torch.argmax(mode_selection_one_hot, dim=1)
        selected_k_values = torch.where(
            selected_k_indices == mode_selection_one_hot.size(1) - 1,  # If it's the last mode (direct regression)
            torch.tensor(inf_sentinel, dtype=torch.long).to(x_input.device),
            selected_k_indices + 2,  # Otherwise, it's k_val = index + 2
        )

        # Use weighted-sum pattern (prob * output) to preserve gradient flow through
        # mode_selection_one_hot. Forward: only one mode contributes (one-hot).
        # Backward: STE gradients flow through all modes via the soft distribution.
        # Skip modes where no sample in the batch selected that mode (saves K-1
        # unnecessary forward passes per step, especially important for REINFORCE).
        for i in range(mode_selection_one_hot.size(1)):
            prob_i = mode_selection_one_hot[:, i].unsqueeze(1)
            if not torch.any(prob_i > 0):
                continue

            if i == mode_selection_one_hot.size(1) - 1:  # Last index is the direct regression bypass
                predictions_for_mode = self.model.direct_regression_head(x_input)
            else:
                predictions_for_mode = self.model._compute_predictions_for_k(classifier_raw_logits, i + 2, boundaries=boundaries)

            final_predictions_contribution += prob_i * predictions_for_mode

        return final_predictions_contribution, selected_k_values, classifier_raw_logits
