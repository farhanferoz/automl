"""Layer selection strategies for flexible neural networks."""

import math
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


class NestedStrategy(BaseSelectionStrategy):
    """Nested-depth training: per-sample depth draws as a SCHEDULE (capacity-ladder F2).

    Every sample independently draws a depth d ~ Uniform{1..max_hidden_layers} on each
    forward pass; the training loss (computed by the caller from `final_output`, same as
    every other strategy) is the readout at that sample's OWN drawn depth. This is a draw,
    not a learned selector: `logits` is ignored entirely (no depth input to the network,
    no penalty), so `n_predictor_layers` must be 0, same as `NoneStrategy`.

    All `max_hidden_layers` per-depth outputs are computed once via the same
    cached-representation loop the soft strategies use (`all_depth_outputs`), so the
    per-sample gather that follows is a free reshape of one forward pass — the prefix
    property this relies on is the one audited in `_flexnn_prefix_selftest.py`.
    `all_depth_log_likelihood` reuses that same cache to export the eval-time all-depth
    score table the capacity-ladder readers consume.
    """

    def setup_optimizers(self, policy_params: Any) -> None:
        """No policy optimizer needed: NESTED has no n_predictor to train."""

    def on_epoch_end(self, **kwargs: Any) -> None:
        """No epoch-end action: the draw distribution is fixed (uniform), nothing anneals."""

    def all_depth_outputs(self, x_input: torch.Tensor) -> torch.Tensor:
        """Computes the shared output layer's readout at every depth, in ONE forward pass.

        Args:
            x_input (torch.Tensor): Input tensor, shape (batch, in_features).

        Returns:
            torch.Tensor: Shape (batch, max_hidden_layers, out_features); index
                `[:, d - 1, :]` is the depth-d readout (blocks 1..d then the shared
                output layer) — the same value an independent truncated forward at
                depth d would give.
        """
        outputs = []
        current_output = x_input
        for i in range(self.model.max_hidden_layers):
            current_output = self.model.model.hidden_layers_blocks[i](current_output)
            outputs.append(self.model.model.output_layer(current_output))
        return torch.stack(outputs, dim=1)

    def all_depth_log_likelihood(self, x_input: torch.Tensor, y_target: torch.Tensor) -> torch.Tensor:
        """Per-sample Gaussian log-likelihood at every depth, from ONE forward pass.

        Requires `UncertaintyMethod.PROBABILISTIC` (native (mean, log_var) heads) — NLL
        scoring falls out of the same forward, no shim needed. This is the all-depth
        score table F2's ladder read and the `_capacity_ladder.py` readers consume.

        Args:
            x_input (torch.Tensor): Input tensor, shape (batch, in_features).
            y_target (torch.Tensor): Targets, shape (batch,) or (batch, 1).

        Returns:
            torch.Tensor: Shape (batch, max_hidden_layers); higher is better.
        """
        all_outputs = self.all_depth_outputs(x_input)  # (batch, depth, out_features)
        if all_outputs.size(-1) != 2:
            raise ValueError("all_depth_log_likelihood requires UncertaintyMethod.PROBABILISTIC (mean, log_var) heads.")
        mean = all_outputs[..., 0]
        log_var = all_outputs[..., 1]
        variance = torch.exp(log_var)
        y = y_target.reshape(-1, 1).to(all_outputs.device)
        return -0.5 * (math.log(2 * math.pi) + log_var + (y - mean) ** 2 / variance)

    def forward(self, x_input: torch.Tensor, _logits: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
        """Draws a per-sample depth ~ Uniform{1..max_hidden_layers}; returns that depth's readout.

        Args:
            x_input (torch.Tensor): Input tensor.
            _logits (torch.Tensor | None): Ignored — NESTED does not condition on the n_predictor.

        Returns:
            tuple: A tuple containing final predictions (each sample's own drawn-depth
                readout), actual n (the drawn depth), None, n_probs (one-hot at the
                drawn depth, for logging parity with the other strategies), and log_prob.
        """
        max_depth = self.model.max_hidden_layers
        all_outputs = self.all_depth_outputs(x_input)  # (batch, depth, out_features)

        depth_idx = torch.randint(0, max_depth, (x_input.size(0),), device=x_input.device)
        gather_index = depth_idx.view(-1, 1, 1).expand(-1, 1, all_outputs.size(-1))
        final_output = all_outputs.gather(1, gather_index).squeeze(1)

        n_actual = depth_idx + 1
        n_probs = f.one_hot(depth_idx, num_classes=max_depth).float()
        self.mode_selection_probs = n_probs

        return final_output, n_actual, None, n_probs, torch.tensor(0.0)


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
        # Tuple order matches the other layer strategies: (output, n_actual, _unused, n_probs, log_prob).
        # Putting n_probs at position 3 lets depth regularisation pick it up uniformly.
        return (
            final_output,
            n_actual,
            None,
            self.mode_selection_probs,
            log_prob,
        )

    def on_epoch_end(self, **kwargs: Any) -> None:
        """Performs operations at the end of each training epoch."""
