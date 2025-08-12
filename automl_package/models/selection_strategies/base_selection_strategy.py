"""Base class for selection strategies in neural networks."""

from abc import ABC, abstractmethod
from typing import Any

import torch


class BaseSelectionStrategy(ABC):
    """Abstract base class for all selection strategies (layer, n_classes, etc.)."""

    def __init__(self, model_instance: Any):
        """Initializes the BaseSelectionStrategy.

        Args:
            model_instance (Any): The model instance this strategy is associated with.
        """
        self.model = model_instance
        self.policy_optimizer = None
        self.mode_selection_probs = None  # To store probabilities for external use (e.g., classifier_logits_out)

    @abstractmethod
    def forward(self, x_input: torch.Tensor, logits: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Processes input and logits to determine selection and compute predictions.

        Returns: (final_predictions, n_actual_for_logging, n_probs_for_logging, n_logits_for_logging, log_prob_for_reinforce).
        """

    def setup_optimizers(self, policy_params: Any):
        """Sets up optimizers for policy-based strategies (e.g., REINFORCE)."""
        # Default implementation does nothing for non-RL strategies

    def on_epoch_end(self, **kwargs):
        """Hook for epoch-end operations (e.g., REINFORCE policy updates)."""
        # Default implementation does nothing

    # Helper for weighted average logic (common to GumbelSoftmax and SoftGating)
    def _weighted_average_logic(self, x_input: torch.Tensor, mode_selection_probs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Applies weighted average based on selection probabilities.

        Assumes self.model has methods/attributes like:
        - .regression_output_size
        - .direct_regression_head (optional)
        - .max_n_classes_for_probabilistic_path (for n_classes selection)
        - ._compute_predictions_for_k (for n_classes selection)
        - .max_hidden_layers (for layer selection)
        - .model.hidden_layers_blocks (for layer selection)
        - .model.output_layer (for layer selection).
        """
        final_predictions_contribution = torch.zeros(x_input.size(0), self.model.regression_output_size).to(x_input.device)

        # Handle direct regression contribution (specific to n_classes selection)
        if hasattr(self.model, "direct_regression_head") and self.model.direct_regression_head is not None:
            is_direct_regression_prob = mode_selection_probs[:, -1]
            direct_reg_predictions = self.model.direct_regression_head(x_input)
            final_predictions_contribution += is_direct_regression_prob.unsqueeze(1) * direct_reg_predictions

        # Handle probabilistic path contribution (specific to n_classes selection)
        if hasattr(self.model, "max_n_classes_for_probabilistic_path"):
            for k_val in range(2, self.model.max_n_classes_for_probabilistic_path + 1):
                prob_k_val = mode_selection_probs[:, k_val - 2]  # k_val=2 corresponds to index 0, k_val=3 to index 1, etc.
                predictions_for_k = self.model._compute_predictions_for_k(x_input, k_val)  # Assumes _compute_predictions_for_k is on the model
                final_predictions_contribution += prob_k_val.unsqueeze(1) * predictions_for_k

        # Handle layer selection contribution (specific to FlexibleHiddenLayersNN)
        elif hasattr(self.model, "max_hidden_layers"):
            # This part needs to be adapted from FlexibleHiddenLayersNN's _weighted_average_forward
            # It's more complex as it involves intermediate hidden representations.
            # For now, this is a placeholder, actual implementation will be in the concrete strategy.
            # This might be a case where the helper needs to be overridden or specialized.
            pass  # Actual implementation will be in LayerGumbelSoftmaxStrategy etc.

        selected_values_for_logging = torch.argmax(mode_selection_probs, dim=1) + 2  # Default for n_classes
        if hasattr(self.model, "n_classes_inf"):  # Adjust for direct regression mode in n_classes
            selected_values_for_logging[selected_values_for_logging == self.model.max_n_classes_for_probabilistic_path + 2] = int(self.model.n_classes_inf)
        elif hasattr(self.model, "max_hidden_layers"):  # Adjust for layer selection
            selected_values_for_logging = torch.argmax(mode_selection_probs, dim=1) + 1  # Layers start from 1

        return final_predictions_contribution, selected_values_for_logging

    # Helper for hard selection logic (common to STE and REINFORCE)
    def _hard_selection_logic(self, x_input: torch.Tensor, mode_selection_one_hot: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Applies hard selection based on one-hot encoded choices.

        Assumes self.model has methods/attributes similar to _weighted_average_logic.
        """
        final_predictions_contribution = torch.zeros(x_input.size(0), self.model.regression_output_size).to(x_input.device)

        is_direct_regression_mode = mode_selection_one_hot[:, -1].bool() if hasattr(self.model, "direct_regression_head") else torch.zeros_like(x_input[:, 0], dtype=torch.bool)

        if torch.any(is_direct_regression_mode):
            direct_reg_indices = torch.where(is_direct_regression_mode)[0]
            if hasattr(self.model, "direct_regression_head") and self.model.direct_regression_head is not None:
                final_predictions_contribution[direct_reg_indices] = self.model.direct_regression_head(x_input[direct_reg_indices])

        probabilistic_indices = torch.where(~is_direct_regression_mode)[0]
        if torch.any(probabilistic_indices):
            if hasattr(self.model, "max_n_classes_for_probabilistic_path"):
                k_selection_one_hot = mode_selection_one_hot[probabilistic_indices, :-1]
                k_values_for_probabilistic = torch.argmax(k_selection_one_hot, dim=1) + 2

                for i, sample_idx in enumerate(probabilistic_indices):
                    k_val = k_values_for_probabilistic[i].item()
                    predictions_for_k = self.model._compute_predictions_for_k(x_input[sample_idx].unsqueeze(0), k_val)
                    final_predictions_contribution[sample_idx] = predictions_for_k.squeeze(0)
            elif hasattr(self.model, "max_hidden_layers"):
                # This part needs to be adapted from FlexibleHiddenLayersNN's hard selection logic
                # It involves iterating through layers and applying masks.
                # For now, this is a placeholder, actual implementation will be in the concrete strategy.
                pass  # Actual implementation will be in LayerSteStrategy etc.

        selected_values_for_logging = torch.argmax(mode_selection_one_hot, dim=1) + 2  # Default for n_classes
        if hasattr(self.model, "n_classes_inf"):  # Adjust for direct regression mode in n_classes
            selected_values_for_logging[selected_values_for_logging == self.model.max_n_classes_for_probabilistic_path + 2] = int(self.model.n_classes_inf)
        elif hasattr(self.model, "max_hidden_layers"):  # Adjust for layer selection
            selected_values_for_logging = torch.argmax(mode_selection_one_hot, dim=1) + 1  # Layers start from 1

        return final_predictions_contribution, selected_values_for_logging
