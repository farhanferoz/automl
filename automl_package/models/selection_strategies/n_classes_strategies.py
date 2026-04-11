"""N-classes selection strategies for probabilistic regression models."""

from typing import Any

import torch
import torch.nn.functional as f

from automl_package.enums import RegressionStrategy
from automl_package.models.selection_strategies.base_selection_strategy import BaseSelectionStrategy


class NoneStrategy(BaseSelectionStrategy):
    """Uses a fixed n_classes, bypassing the n_classes_predictor."""

    def setup_optimizers(self, policy_params: Any) -> None:
        """Setup the optimizers for the policy parameters (no-op for this strategy)."""

    def on_epoch_end(self, **kwargs: Any) -> None:
        """This method is called at the end of each epoch (no-op for this strategy)."""

    def forward(
        self, x_input: torch.Tensor, _logits: torch.Tensor | None, boundaries: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor | None]:
        """Performs forward pass without n_classes selection.

        Args:
            x_input (torch.Tensor): Input tensor.
            _logits (torch.Tensor | None): Logits from the n_classes_predictor (ignored in this strategy).
            boundaries (torch.Tensor | None): Optional boundaries for the sigmoid transformation.

        Returns:
            tuple: A tuple containing final predictions, selected k values, log_prob_for_reinforce, and classifier_raw_logits.
        """
        # logits is ignored for this strategy

        classifier_raw_logits = self.model.classifier_layers(x_input)

        masked_classifier_logits = torch.full_like(classifier_raw_logits, float("-inf"))
        masked_classifier_logits[:, : self.model.n_classes] = classifier_raw_logits[:, : self.model.n_classes]

        probabilities = torch.softmax(masked_classifier_logits, dim=1)

        if self.model.regression_strategy == RegressionStrategy.SINGLE_HEAD_FINAL_OUTPUT:
            final_predictions_contribution = self.model.regression_module(probabilities, return_head_outputs=True, boundaries=boundaries)
            per_head_outputs = None  # This strategy doesn't have per-head outputs
        else:
            final_predictions_contribution, per_head_outputs = self.model.regression_module(probabilities, return_head_outputs=True, boundaries=boundaries)

        selected_k_values_for_logging = torch.full((x_input.size(0),), self.model.n_classes, dtype=torch.long).to(x_input.device)

        return final_predictions_contribution, selected_k_values_for_logging, None, classifier_raw_logits, per_head_outputs


class GumbelSoftmaxStrategy(BaseSelectionStrategy):
    """Uses Gumbel-Softmax for a weighted average of architectures."""

    def setup_optimizers(self, policy_params: Any) -> None:
        """Setup the optimizers for the policy parameters (no-op for this strategy)."""

    def on_epoch_end(self, **kwargs: Any) -> None:
        """This method is called at the end of each epoch (no-op for this strategy)."""

    def forward(
        self, x_input: torch.Tensor, logits: torch.Tensor, boundaries: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor | None]:
        """Performs forward pass using Gumbel-Softmax for n_classes selection."""
        if self.model.training:
            mode_selection_probs = f.gumbel_softmax(logits, tau=self.model.gumbel_tau, hard=False, dim=-1)
        else:
            # At eval time, use deterministic softmax — Gumbel noise makes predictions stochastic
            mode_selection_probs = f.softmax(logits / self.model.gumbel_tau, dim=-1)
        self.mode_selection_probs = mode_selection_probs
        final_predictions_contribution, selected_k_values_for_logging, classifier_raw_logits = self._weighted_average_logic(x_input, mode_selection_probs)
        return final_predictions_contribution, selected_k_values_for_logging, None, classifier_raw_logits, None


class SoftGatingStrategy(BaseSelectionStrategy):
    """Uses Softmax for a weighted average of architectures."""

    def setup_optimizers(self, policy_params: Any) -> None:
        """Setup the optimizers for the policy parameters (no-op for this strategy)."""

    def on_epoch_end(self, **kwargs: Any) -> None:
        """This method is called at the end of each epoch (no-op for this strategy)."""

    def forward(
        self, x_input: torch.Tensor, logits: torch.Tensor, boundaries: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor | None]:
        """Performs forward pass using Softmax for n_classes selection."""
        mode_selection_probs = f.softmax(logits, dim=-1)
        self.mode_selection_probs = mode_selection_probs
        final_predictions_contribution, selected_k_values_for_logging, classifier_raw_logits = self._weighted_average_logic(x_input, mode_selection_probs)
        return final_predictions_contribution, selected_k_values_for_logging, None, classifier_raw_logits, None


class SteStrategy(BaseSelectionStrategy):
    """Uses Straight-Through Estimator (hard Gumbel-Softmax)."""

    def setup_optimizers(self, policy_params: Any) -> None:
        """Setup the optimizers for the policy parameters (no-op for this strategy)."""

    def on_epoch_end(self, **kwargs: Any) -> None:
        """This method is called at the end of each epoch (no-op for this strategy)."""

    def forward(
        self, x_input: torch.Tensor, logits: torch.Tensor, boundaries: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor | None]:
        """Performs forward pass using STE for n_classes selection."""
        mode_selection_one_hot = f.gumbel_softmax(logits, tau=self.model.gumbel_tau, hard=True, dim=-1)
        self.mode_selection_probs = mode_selection_one_hot
        final_predictions_contribution, selected_k_values_for_logging, classifier_raw_logits = self._hard_selection_logic(x_input, mode_selection_one_hot)
        return final_predictions_contribution, selected_k_values_for_logging, None, classifier_raw_logits, None


class ReinforceStrategy(BaseSelectionStrategy):
    """Uses REINFORCE algorithm to select an architecture."""

    def setup_optimizers(self, policy_params: Any) -> None:
        """Sets up optimizers for the REINFORCE policy.

        Args:
            policy_params (Any): Parameters of the policy network.
        """
        self.policy_optimizer = torch.optim.Adam(policy_params, lr=self.model.n_classes_predictor_learning_rate)

    def forward(
        self, x_input: torch.Tensor, logits: torch.Tensor, boundaries: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor | None]:
        """Performs forward pass using REINFORCE for n_classes selection."""
        probs = f.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        mode_selection_one_hot = f.one_hot(action, num_classes=logits.size(-1)).float()
        self.mode_selection_probs = mode_selection_one_hot
        final_predictions_contribution, selected_k_values_for_logging, classifier_raw_logits = self._hard_selection_logic(x_input, mode_selection_one_hot)
        return final_predictions_contribution, selected_k_values_for_logging, log_prob, classifier_raw_logits, None

    def on_epoch_end(self, **kwargs: Any) -> None:
        """Performs operations at the end of each training epoch.

        Args:
            **kwargs: Keyword arguments including validation_loss and epoch_log_probs.
        """
        validation_loss = kwargs.get("validation_loss")
        epoch_log_probs = kwargs.get("epoch_log_probs")

        if not (validation_loss is None or not epoch_log_probs):
            reward = -validation_loss
            self.policy_optimizer.zero_grad()
            policy_loss = -torch.stack(epoch_log_probs).mean() * reward
            policy_loss.backward()
            self.policy_optimizer.step()
