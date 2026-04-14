"""Layer selection strategies for flexible neural networks."""

from typing import Any

import torch
import torch.nn.functional as f

from automl_package.models.selection_strategies.base_selection_strategy import (
    BaseSelectionStrategy,
)


class NoneStrategy(BaseSelectionStrategy):
    """A strategy that does not perform any layer selection, using all layers."""

    def setup_optimizers(self, policy_params: Any) -> None:
        """No optimizer needed for the NoneStrategy."""

    def on_epoch_end(self, **kwargs: Any) -> None:
        """No special actions needed at the end of the epoch for NoneStrategy."""

    def forward(self, x_input: torch.Tensor, _logits: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
        """Performs forward pass without layer selection.

        Args:
            x_input (torch.Tensor): Input tensor.
            _logits (torch.Tensor | None): Logits from the n_predictor (ignored in this strategy).

        Returns:
            tuple: A tuple containing final predictions, actual n, None, n_probs, and log_prob_for_reinforce.
        """
        current_output = x_input
        for i in range(self.model.max_hidden_layers):
            current_output = self.model.model.hidden_layers_blocks[i](current_output)
        final_output = self.model.model.output_layer(current_output)

        n_actual = torch.full(
            (x_input.size(0),),
            self.model.max_hidden_layers,
            device=x_input.device,
            dtype=torch.long,
        )
        n_probs = torch.zeros(x_input.size(0), self.model.max_hidden_layers, device=x_input.device)
        if self.model.max_hidden_layers > 0:
            n_probs[:, -1] = 1.0

        self.mode_selection_probs = n_probs
        return final_output, n_actual, None, n_probs, torch.tensor(0.0)


class GumbelSoftmaxStrategy(BaseSelectionStrategy):
    """A strategy that uses Gumbel-Softmax for differentiable layer selection."""

    def setup_optimizers(self, policy_params: Any) -> None:
        """No separate optimizer needed for Gumbel-Softmax (gradients flow through main optimizer)."""

    def on_epoch_end(self, **kwargs: Any) -> None:
        """Performs operations at the end of each training epoch."""

    def forward(self, x_input: torch.Tensor, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
        """Performs forward pass using Gumbel-Softmax for layer selection.

        Args:
            x_input (torch.Tensor): Input tensor.
            logits (torch.Tensor): Logits from the n_predictor.

        Returns:
            tuple: A tuple containing final predictions, actual n, None, n_probs, and log_prob_for_reinforce.
        """
        n_probs = f.gumbel_softmax(logits, tau=self.model.gumbel_tau, hard=False, dim=1)
        self.mode_selection_probs = n_probs  # Store for external use

        # Specific weighted average logic for layers
        final_output_neurons = self.model.model.output_layer.out_features
        aggregated_output = torch.zeros(x_input.size(0), final_output_neurons, device=x_input.device)

        max_depth_needed = self.model.max_hidden_layers  # Assuming n_probs covers all possible layers
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

        n_actual = torch.argmax(n_probs, dim=1) + 1  # Layers start from 1

        return aggregated_output, n_actual, None, n_probs, torch.tensor(0.0)


class SoftGatingStrategy(BaseSelectionStrategy):
    """A strategy that uses Softmax for differentiable layer selection."""

    def setup_optimizers(self, policy_params: Any) -> None:
        """No separate optimizer needed for SoftGating."""

    def on_epoch_end(self, **kwargs: Any) -> None:
        """Performs operations at the end of each training epoch."""

    def forward(self, x_input: torch.Tensor, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
        """Performs forward pass using Softmax for layer selection.

        Args:
            x_input (torch.Tensor): Input tensor.
            logits (torch.Tensor): Logits from the n_predictor.

        Returns:
            tuple: A tuple containing final predictions, actual n, None, n_probs, and log_prob_for_reinforce.
        """
        n_probs = f.softmax(logits, dim=1)
        self.mode_selection_probs = n_probs  # Store for external use

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

        return aggregated_output, n_actual, None, n_probs, torch.tensor(0.0)


class SteStrategy(BaseSelectionStrategy):
    """A strategy that uses Straight-Through Estimator (STE) for hard layer selection."""

    def setup_optimizers(self, policy_params: Any) -> None:
        """No separate optimizer needed for STE (gradients flow via straight-through estimator)."""

    def on_epoch_end(self, **kwargs: Any) -> None:
        """Performs operations at the end of each training epoch."""

    def forward(self, x_input: torch.Tensor, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
        """Performs forward pass using STE for layer selection.

        Args:
            x_input (torch.Tensor): Input tensor.
            logits (torch.Tensor): Logits from the n_predictor.

        Returns:
            tuple: A tuple containing final predictions, actual n, None, n_probs, and log_prob_for_reinforce.
        """
        n_probs = f.gumbel_softmax(logits, tau=self.model.gumbel_tau, hard=True, dim=1)
        self.mode_selection_probs = n_probs  # Store for external use

        # Use weighted-sum pattern identical to GumbelSoftmax but with hard=True.
        # Forward: only one depth contributes (one-hot). Backward: STE gradients flow through all.
        final_output_neurons = self.model.model.output_layer.out_features
        aggregated_output = torch.zeros(x_input.size(0), final_output_neurons, device=x_input.device)

        hidden_representations = []
        current_output = x_input
        for i in range(self.model.max_hidden_layers):
            current_output = self.model.model.hidden_layers_blocks[i](current_output)
            hidden_representations.append(current_output)

        for i in range(self.model.max_hidden_layers):
            prob = n_probs[:, i]
            hidden_rep = hidden_representations[i]
            output_for_n = self.model.model.output_layer(hidden_rep)
            aggregated_output += prob.unsqueeze(1) * output_for_n

        n_actual = torch.argmax(n_probs, dim=1) + 1
        return aggregated_output, n_actual, None, n_probs, torch.tensor(0.0)


class ReinforceStrategy(BaseSelectionStrategy):
    """A strategy that uses REINFORCE for layer selection."""

    def setup_optimizers(self, policy_params: Any) -> None:
        """Setup separate optimizer for the policy (n_predictor)."""
        self.policy_optimizer = torch.optim.Adam(policy_params, lr=self.model.n_predictor_learning_rate)

    def forward(self, x_input: torch.Tensor, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Performs forward pass using REINFORCE for layer selection.

        Args:
            x_input (torch.Tensor): Input tensor.
            logits (torch.Tensor): Logits from the n_predictor.

        Returns:
            tuple: A tuple containing final predictions, actual n, log_prob, n_probs, and log_prob_for_reinforce.
        """
        probs = f.softmax(logits, dim=1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        n_actual = action + 1
        self.mode_selection_probs = f.one_hot(action, num_classes=logits.size(-1)).float()  # Store for external use

        current_output = x_input
        for i in range(self.model.max_hidden_layers):
            active_mask = (i < n_actual).unsqueeze(1)
            if active_mask.any():
                block_output = self.model.model.hidden_layers_blocks[i](current_output)
                current_output = torch.where(active_mask, block_output, current_output)

        final_output = self.model.model.output_layer(current_output)
        return (
            final_output,
            n_actual,
            self.mode_selection_probs,
            torch.zeros(1, device=x_input.device),
            log_prob,
        )

    def on_epoch_end(self, **kwargs: Any) -> None:
        """Performs operations at the end of each training epoch."""
