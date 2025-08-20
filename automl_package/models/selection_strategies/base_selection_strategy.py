"""Base class for selection strategies in neural networks."""

from abc import ABC, abstractmethod
from typing import Any

import torch


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
    def forward(self, x_input: torch.Tensor, logits: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor | None]:
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

    # Helper for weighted average logic (common to GumbelSoftmax and SoftGating)
    def _weighted_average_logic(self, x_input: torch.Tensor, mode_selection_probs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        final_predictions_contribution = torch.zeros(x_input.size(0), self.model.regression_output_size).to(x_input.device)

        # Determine the selected k values based on the argmax of probabilities
        selected_k_indices = torch.argmax(mode_selection_probs, dim=1)
        selected_k_values = torch.where(
            selected_k_indices == mode_selection_probs.size(1) - 1,  # If it's the last mode (direct regression)
            torch.tensor(self.model.n_classes_inf, dtype=torch.long).to(x_input.device),
            selected_k_indices + 2,  # Otherwise, it's k_val = index + 2
        )

        # Iterate over all possible k values (from 2 up to max_n_classes_for_probabilistic_path)
        # and the direct regression mode
        for i in range(mode_selection_probs.size(1)):
            prob_i = mode_selection_probs[:, i].unsqueeze(1)  # Probability for this mode

            if i == mode_selection_probs.size(1) - 1:  # This is the direct regression mode
                predictions_for_mode = self.model.direct_regression_head(x_input)
            else:  # This is a probabilistic path with k_val = i + 2
                k_val = i + 2
                predictions_for_mode = self.model._compute_predictions_for_k(x_input, k_val)

            final_predictions_contribution += prob_i * predictions_for_mode

        return final_predictions_contribution, selected_k_values

    # Helper for hard selection logic (common to STE and REINFORCE)
    def _hard_selection_logic(self, x_input: torch.Tensor, mode_selection_one_hot: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Applies hard selection based on one-hot encoded choices.

        Assumes self.model has methods/attributes similar to _weighted_average_logic.
        """
        classifier_raw_logits = self.model.classifier_layers(x_input)
        final_predictions_contribution = torch.zeros(x_input.size(0), self.model.regression_output_size).to(x_input.device)

        selected_k_indices = torch.argmax(mode_selection_one_hot, dim=1)
        selected_k_values = torch.where(
            selected_k_indices == mode_selection_one_hot.size(1) - 1,  # If it's the last mode (direct regression)
            torch.tensor(self.model.n_classes_inf, dtype=torch.long).to(x_input.device),
            selected_k_indices + 2,  # Otherwise, it's k_val = index + 2
        )

        for i in range(mode_selection_one_hot.size(1)):
            # Get indices where this mode is selected
            selected_indices = torch.where(selected_k_indices == i)[0]

            if len(selected_indices) > 0:
                if i == mode_selection_one_hot.size(1) - 1:  # Direct regression mode
                    predictions_for_mode = self.model.direct_regression_head(x_input[selected_indices])
                else:  # Probabilistic path with k_val = i + 2
                    k_val = i + 2
                    predictions_for_mode = self.model._compute_predictions_for_k(classifier_raw_logits[selected_indices], k_val)

                final_predictions_contribution[selected_indices] = predictions_for_mode

        return final_predictions_contribution, selected_k_values, classifier_raw_logits
