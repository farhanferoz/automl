from .base_selection_strategy import BaseSelectionStrategy
import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Any

class NoneStrategy(BaseSelectionStrategy):
    """Uses a fixed n_classes, bypassing the n_classes_predictor."""
    def forward(self, x_input: torch.Tensor, logits: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        # logits is ignored for this strategy
        
        classifier_raw_logits = self.model.classifier_layers(x_input)
        
        masked_classifier_logits = torch.full_like(classifier_raw_logits, float("-inf"))
        masked_classifier_logits[:, :self.model.n_classes] = classifier_raw_logits[:, :self.model.n_classes]

        probabilities = torch.softmax(masked_classifier_logits, dim=1)
        final_predictions_contribution = self.model.regression_module(probabilities)
        
        selected_k_values_for_logging = torch.full((x_input.size(0),), self.model.n_classes, dtype=torch.long).to(x_input.device)
        
        return final_predictions_contribution, selected_k_values_for_logging, None


class GumbelSoftmaxStrategy(BaseSelectionStrategy):
    """Uses Gumbel-Softmax for a weighted average of architectures."""
    def forward(self, x_input: torch.Tensor, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        mode_selection_probs = F.gumbel_softmax(logits, tau=self.model.gumbel_tau, hard=False, dim=-1)
        self.mode_selection_probs = mode_selection_probs # Store for classifier_logits_out in _CombinedProbabilisticModel
        final_predictions_contribution, selected_k_values_for_logging = self._weighted_average_logic(x_input, mode_selection_probs)
        return final_predictions_contribution, selected_k_values_for_logging, None


class SoftGatingStrategy(BaseSelectionStrategy):
    """Uses Softmax for a weighted average of architectures."""
    def forward(self, x_input: torch.Tensor, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        mode_selection_probs = F.softmax(logits, dim=-1)
        self.mode_selection_probs = mode_selection_probs # Store for classifier_logits_out in _CombinedProbabilisticModel
        final_predictions_contribution, selected_k_values_for_logging = self._weighted_average_logic(x_input, mode_selection_probs)
        return final_predictions_contribution, selected_k_values_for_logging, None


class SteStrategy(BaseSelectionStrategy):
    """Uses Straight-Through Estimator (hard Gumbel-Softmax)."""
    def forward(self, x_input: torch.Tensor, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        mode_selection_one_hot = F.gumbel_softmax(logits, tau=self.model.gumbel_tau, hard=True, dim=-1)
        self.mode_selection_probs = mode_selection_one_hot # Store for classifier_logits_out in _CombinedProbabilisticModel
        final_predictions_contribution, selected_k_values_for_logging = self._hard_selection_logic(x_input, mode_selection_one_hot)
        return final_predictions_contribution, selected_k_values_for_logging, None


class ReinforceStrategy(BaseSelectionStrategy):
    """Uses REINFORCE algorithm to select an architecture."""
    def setup_optimizers(self, policy_params: Any):
        self.policy_optimizer = torch.optim.Adam(policy_params, lr=self.model.n_classes_predictor_learning_rate)

    def forward(self, x_input: torch.Tensor, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        mode_selection_one_hot = F.one_hot(action, num_classes=logits.size(-1)).float()
        self.mode_selection_probs = mode_selection_one_hot # Store for classifier_logits_out in _CombinedProbabilisticModel
        final_predictions_contribution, selected_k_values_for_logging = self._hard_selection_logic(x_input, mode_selection_one_hot)
        return final_predictions_contribution, selected_k_values_for_logging, log_prob

    def on_epoch_end(self, **kwargs):
        validation_loss = kwargs.get('validation_loss')
        epoch_log_probs = kwargs.get('epoch_log_probs')

        if validation_loss is None or not epoch_log_probs:
            return

        reward = -validation_loss
        self.policy_optimizer.zero_grad()
        policy_loss = -torch.stack(epoch_log_probs).mean() * reward
        policy_loss.backward()
        self.policy_optimizer.step()
