from .base_selection_strategy import BaseSelectionStrategy
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Any

class NoneStrategy(BaseSelectionStrategy):
    def forward(self, x_input: torch.Tensor, logits: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        current_output = x_input
        for i in range(self.model.max_hidden_layers):
            current_output = self.model.model.hidden_layers_blocks[i](current_output)
        final_output = self.model.model.output_layer(current_output)
        
        n_actual = torch.full((x_input.size(0),), self.model.max_hidden_layers, device=x_input.device, dtype=torch.long)
        n_probs = torch.zeros(x_input.size(0), self.model.max_hidden_layers, device=x_input.device)
        if self.model.max_hidden_layers > 0:
            n_probs[:, -1] = 1.0
        
        self.mode_selection_probs = n_probs # Store for external use
        return final_output, n_actual, None

class GumbelSoftmaxStrategy(BaseSelectionStrategy):
    def forward(self, x_input: torch.Tensor, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        n_probs = F.gumbel_softmax(logits, tau=self.model.gumbel_tau, hard=False, dim=1)
        self.mode_selection_probs = n_probs # Store for external use
        
        # Specific weighted average logic for layers
        final_output_neurons = self.model.model.output_layer.out_features
        aggregated_output = torch.zeros(x_input.size(0), final_output_neurons, device=x_input.device)

        max_depth_needed = self.model.max_hidden_layers # Assuming n_probs covers all possible layers
        hidden_representations = []
        current_output = x_input
        for i in range(max_depth_needed):
            current_output = self.model.model.hidden_layers_blocks[i](current_output)
            hidden_representations.append(current_output)

        for i in range(max_depth_needed):
            prob = n_probs[:, i]
            if not torch.any(prob > 1e-9):
                continue
            hidden_rep = hidden_representations[i]
            output_for_n = self.model.model.output_layer(hidden_rep)
            aggregated_output += prob.unsqueeze(1) * output_for_n
        
        n_actual = torch.argmax(n_probs, dim=1) + 1 # Layers start from 1
        
        return aggregated_output, n_actual, None

class SoftGatingStrategy(BaseSelectionStrategy):
    def forward(self, x_input: torch.Tensor, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        n_probs = F.softmax(logits, dim=1)
        self.mode_selection_probs = n_probs # Store for external use
        
        # Specific weighted average logic for layers (same as GumbelSoftmaxStrategy for layers)
        final_output_neurons = self.model.model.output_layer.out_features
        aggregated_output = torch.zeros(x_input.size(0), final_output_neurons, device=x_input.device)

        max_depth_needed = self.model.max_hidden_layers
        hidden_representations = []
        current_output = x_input
        for i in range(max_depth_needed):
            current_output = self.model.model.hidden_layers_blocks[i](current_output)
            hidden_representations.append(current_output)

        for i in range(max_depth_needed):
            prob = n_probs[:, i]
            if not torch.any(prob > 1e-9):
                continue
            hidden_rep = hidden_representations[i]
            output_for_n = self.model.model.output_layer(hidden_rep)
            aggregated_output += prob.unsqueeze(1) * output_for_n
        
        n_actual = torch.argmax(n_probs, dim=1) + 1
        
        return aggregated_output, n_actual, None

class SteStrategy(BaseSelectionStrategy):
    def forward(self, x_input: torch.Tensor, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        n_probs = F.gumbel_softmax(logits, tau=self.model.gumbel_tau, hard=True, dim=1)
        self.mode_selection_probs = n_probs # Store for external use
        
        chosen_indices = torch.argmax(n_probs, dim=1)
        n_actual = chosen_indices + 1

        current_output = x_input
        for i in range(self.model.max_hidden_layers):
            active_mask = (i < n_actual).unsqueeze(1)
            if active_mask.any():
                block_output = self.model.model.hidden_layers_blocks[i](current_output)
                current_output = torch.where(active_mask, block_output, current_output)

        final_output = self.model.model.output_layer(current_output)
        return final_output, n_actual, None

class ReinforceStrategy(BaseSelectionStrategy):
    def setup_optimizers(self, policy_params: Any):
        self.policy_optimizer = torch.optim.Adam(policy_params, lr=self.model.n_predictor_learning_rate)

    def forward(self, x_input: torch.Tensor, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        probs = F.softmax(logits, dim=1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        n_actual = action + 1
        self.mode_selection_probs = F.one_hot(action, num_classes=logits.size(-1)).float() # Store for external use

        current_output = x_input
        for i in range(self.model.max_hidden_layers):
            active_mask = (i < n_actual).unsqueeze(1)
            if active_mask.any():
                block_output = self.model.model.hidden_layers_blocks[i](current_output)
                current_output = torch.where(active_mask, block_output, current_output)

        final_output = self.model.model.output_layer(current_output)
        return final_output, n_actual, log_prob

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
