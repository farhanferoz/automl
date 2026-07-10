"""This module contains the strategies for the IndependentWeightsFlexibleNN model."""

from typing import Any

import torch
import torch.nn.functional as f
from torch.distributions import Categorical

from automl_package.models.selection_strategies.base_selection_strategy import (
    BaseSelectionStrategy,
)

SelectionOutput = tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]
ForwardOutput = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


class IndependentWeightsGumbelSoftmaxStrategy(BaseSelectionStrategy):
    """Gumbel-Softmax strategy for IndependentWeightsFlexibleNN.

    This strategy only performs the selection of 'n' (number of layers)
    and does NOT apply any neural network layers.
    """

    def __init__(self, outer_instance: Any) -> None:
        """Initialize the IndependentWeightsGumbelSoftmaxStrategy."""
        super().__init__(outer_instance)
        self.gumbel_tau: float = self.model.gumbel_tau
        self.gumbel_tau_anneal_rate: float = self.model.gumbel_tau_anneal_rate
        self.n_predictor_learning_rate: float = self.model.n_predictor_learning_rate

    def setup_optimizers(self, policy_params: Any) -> None:
        """Setup the optimizers for the policy parameters."""
        self.policy_optimizer = torch.optim.Adam(policy_params, lr=self.n_predictor_learning_rate)

    def select_n(self, x_input: torch.Tensor, n_logits: torch.Tensor) -> SelectionOutput:  # noqa: ARG002
        """Selects the number of layers 'n' using Gumbel-Softmax.

        :param x_input: The input tensor.
        :param n_logits: The logits for the number of layers.
        :return: A tuple containing the actual number of layers, the probabilities
            for the number of layers, the logits for the number of layers, and the
            log probability.
        """
        if n_logits is None:
            raise ValueError("n_logits cannot be None for Gumbel-Softmax strategy.")

        n_probs = f.gumbel_softmax(n_logits, tau=self.gumbel_tau, hard=True)
        n_actual = torch.argmax(n_probs, dim=-1) + 1
        log_prob = torch.log(n_probs.max(dim=-1).values)
        return n_actual, n_probs, n_logits, log_prob

    def forward(self, x_input: torch.Tensor, n_logits: torch.Tensor) -> ForwardOutput:
        """This forward method is called by the model's main forward pass.

        :param x_input: The input tensor.
        :param n_logits: The logits for the number of layers.
        :return: A tuple containing the input tensor, the actual number of layers,
            the probabilities for the number of layers, the logits for the number of
            layers, and the log probability.
        """
        n_actual, n_probs, n_logits_out, log_prob = self.select_n(x_input, n_logits)
        return x_input, n_actual, n_probs, n_logits_out, log_prob

    def on_epoch_end(self, validation_loss: float, epoch_log_probs: Any) -> None:  # noqa: ARG002
        """This method is called at the end of each epoch."""
        self.gumbel_tau = max(0.5, self.gumbel_tau * self.gumbel_tau_anneal_rate)


class IndependentWeightsNoneStrategy(BaseSelectionStrategy):
    """NONE strategy for IndependentWeightsFlexibleNN.

    Always selects max_hidden_layers.
    """

    def __init__(self, outer_instance: Any) -> None:
        """Initialize the IndependentWeightsNoneStrategy."""
        super().__init__(outer_instance)

    def setup_optimizers(self, policy_params: Any) -> None:  # noqa: ARG002
        """Setup the optimizers for the policy parameters."""
        # No policy optimizer needed for NONE strategy
        self.policy_optimizer = None

    def select_n(self, x_input: torch.Tensor, n_logits: torch.Tensor) -> SelectionOutput:
        """Selects the number of layers 'n' by always choosing the maximum.

        :param x_input: The input tensor.
        :param n_logits: The logits for the number of layers.
        :return: A tuple containing the actual number of layers, the probabilities
            for the number of layers, the logits for the number of layers, and the
            log probability.
        """
        n_actual = torch.full(
            (x_input.shape[0],),
            self.model.max_hidden_layers,
            dtype=torch.long,
            device=x_input.device,
        )
        n_probs = torch.zeros(x_input.shape[0], self.model.max_hidden_layers, device=x_input.device)
        n_probs.scatter_(1, n_actual.unsqueeze(1) - 1, 1)
        return n_actual, n_probs, n_logits, None

    def forward(self, x_input: torch.Tensor, n_logits: torch.Tensor) -> ForwardOutput:
        """This forward method is called by the model's main forward pass.

        :param x_input: The input tensor.
        :param n_logits: The logits for the number of layers.
        :return: A tuple containing the input tensor, the actual number of layers,
            the probabilities for the number of layers, the logits for the number of
            layers, and the log probability.
        """
        n_actual, n_probs, n_logits_out, log_prob = self.select_n(x_input, n_logits)
        return x_input, n_actual, n_probs, n_logits_out, log_prob

    def on_epoch_end(self, validation_loss: float, epoch_log_probs: Any) -> None:
        """This method is called at the end of each epoch."""
        # No annealing or policy update


class IndependentWeightsSoftGatingStrategy(BaseSelectionStrategy):
    """Soft Gating strategy for IndependentWeightsFlexibleNN.

    Uses softmax to get probabilities for layer selection.
    """

    def __init__(self, outer_instance: Any) -> None:
        """Initialize the IndependentWeightsSoftGatingStrategy."""
        super().__init__(outer_instance)
        self.n_predictor_learning_rate: float = self.model.n_predictor_learning_rate

    def setup_optimizers(self, policy_params: Any) -> None:
        """Setup the optimizers for the policy parameters."""
        self.policy_optimizer = torch.optim.Adam(policy_params, lr=self.n_predictor_learning_rate)

    def select_n(self, x_input: torch.Tensor, n_logits: torch.Tensor) -> SelectionOutput:  # noqa: ARG002
        """Selects the number of layers 'n' using softmax.

        :param x_input: The input tensor.
        :param n_logits: The logits for the number of layers.
        :return: A tuple containing the actual number of layers, the probabilities
            for the number of layers, the logits for the number of layers, and the
            log probability.
        """
        if n_logits is None:
            raise ValueError("n_logits cannot be None for SoftGating strategy.")

        n_probs = f.softmax(n_logits, dim=-1)
        n_actual = torch.argmax(n_probs, dim=-1) + 1
        log_prob = torch.log(n_probs.max(dim=-1).values)
        return n_actual, n_probs, n_logits, log_prob

    def forward(self, x_input: torch.Tensor, n_logits: torch.Tensor) -> ForwardOutput:
        """This forward method is called by the model's main forward pass.

        :param x_input: The input tensor.
        :param n_logits: The logits for the number of layers.
        :return: A tuple containing the input tensor, the actual number of layers,
            the probabilities for the number of layers, the logits for the number of
            layers, and the log probability.
        """
        n_actual, n_probs, n_logits_out, log_prob = self.select_n(x_input, n_logits)
        return x_input, n_actual, n_probs, n_logits_out, log_prob

    def on_epoch_end(self, validation_loss: float, epoch_log_probs: Any) -> None:
        """This method is called at the end of each epoch."""
        # No annealing or policy update specific to soft gating


class IndependentWeightsSteStrategy(BaseSelectionStrategy):
    """Straight-Through Estimator (STE) strategy for IndependentWeightsFlexibleNN."""

    def __init__(self, outer_instance: Any) -> None:
        """Initialize the IndependentWeightsSteStrategy."""
        super().__init__(outer_instance)
        self.n_predictor_learning_rate: float = self.model.n_predictor_learning_rate

    def setup_optimizers(self, policy_params: Any) -> None:
        """Setup the optimizers for the policy parameters."""
        self.policy_optimizer = torch.optim.Adam(policy_params, lr=self.n_predictor_learning_rate)

    def select_n(self, x_input: torch.Tensor, n_logits: torch.Tensor) -> SelectionOutput:  # noqa: ARG002
        """Selects the number of layers 'n' using STE.

        :param x_input: The input tensor.
        :param n_logits: The logits for the number of layers.
        :return: A tuple containing the actual number of layers, the probabilities
            for the number of layers, the logits for the number of layers, and the
            log probability.
        """
        if n_logits is None:
            raise ValueError("n_logits cannot be None for STE strategy.")

        n_probs = f.gumbel_softmax(n_logits, hard=True)
        n_actual = torch.argmax(n_probs, dim=-1) + 1
        log_prob = torch.tensor(0.0, device=n_logits.device)
        return n_actual, n_probs, n_logits, log_prob

    def forward(self, x_input: torch.Tensor, n_logits: torch.Tensor) -> ForwardOutput:
        """This forward method is called by the model's main forward pass.

        :param x_input: The input tensor.
        :param n_logits: The logits for the number of layers.
        :return: A tuple containing the input tensor, the actual number of layers,
            the probabilities for the number of layers, the logits for the number of
            layers, and the log probability.
        """
        n_actual, n_probs, n_logits_out, log_prob = self.select_n(x_input, n_logits)
        return x_input, n_actual, n_probs, n_logits_out, log_prob

    def on_epoch_end(self, validation_loss: float, epoch_log_probs: Any) -> None:
        """This method is called at the end of each epoch."""
        # No annealing or policy update specific to STE


class IndependentWeightsNestedStrategy(BaseSelectionStrategy):
    """NESTED strategy for IndependentWeightsFlexibleNN — the conditioned control.

    Every sample independently draws a depth d ~ Uniform{1..max_hidden_layers}, the same
    per-sample schedule as `NestedStrategy` on the shared-trunk model. `select_n` returns
    a ONE-HOT `n_probs` at the drawn depth, so the module's existing weighted-sum-over-
    independent-networks forward (`IndependentWeightsFlexibleNNModule.forward`) already
    selects exactly that sample's independent network — no change needed there. Ignores
    `n_logits` (no depth input to the network; `n_predictor_layers` must be 0, same as
    `IndependentWeightsNoneStrategy`).
    """

    def setup_optimizers(self, policy_params: Any) -> None:  # noqa: ARG002
        """No policy optimizer needed: NESTED has no n_predictor to train."""
        self.policy_optimizer = None

    def select_n(self, x_input: torch.Tensor, n_logits: torch.Tensor) -> SelectionOutput:
        """Draws a per-sample depth ~ Uniform{1..max_hidden_layers}.

        :param x_input: The input tensor.
        :param n_logits: Ignored — NESTED does not condition on the n_predictor.
        :return: A tuple containing the drawn depth, a one-hot selection over depths,
            the (unused) logits, and None (no log_prob — this is a draw, not a policy).
        """
        max_depth = self.model.max_hidden_layers
        depth_idx = torch.randint(0, max_depth, (x_input.shape[0],), device=x_input.device)
        n_actual = depth_idx + 1
        n_probs = f.one_hot(depth_idx, num_classes=max_depth).float()
        return n_actual, n_probs, n_logits, None

    def forward(self, x_input: torch.Tensor, n_logits: torch.Tensor) -> ForwardOutput:
        """This forward method is called by the model's main forward pass.

        :param x_input: The input tensor.
        :param n_logits: The logits for the number of layers (ignored).
        :return: A tuple containing the input tensor, the actual number of layers,
            the probabilities for the number of layers, the logits for the number of
            layers, and the log probability.
        """
        n_actual, n_probs, n_logits_out, log_prob = self.select_n(x_input, n_logits)
        return x_input, n_actual, n_probs, n_logits_out, log_prob

    def on_epoch_end(self, validation_loss: float, epoch_log_probs: Any) -> None:
        """No epoch-end action: the draw distribution is fixed (uniform), nothing anneals."""


class IndependentWeightsReinforceStrategy(BaseSelectionStrategy):
    """REINFORCE strategy for IndependentWeightsFlexibleNN."""

    def __init__(self, outer_instance: Any) -> None:
        """Initialize the IndependentWeightsReinforceStrategy."""
        super().__init__(outer_instance)
        self.n_predictor_learning_rate: float = self.model.n_predictor_learning_rate

    def setup_optimizers(self, policy_params: Any) -> None:
        """Setup the optimizers for the policy parameters."""
        self.policy_optimizer = torch.optim.Adam(policy_params, lr=self.n_predictor_learning_rate)

    def select_n(self, x_input: torch.Tensor, n_logits: torch.Tensor) -> SelectionOutput:  # noqa: ARG002
        """Selects the number of layers 'n' using REINFORCE.

        :param x_input: The input tensor.
        :param n_logits: The logits for the number of layers.
        :return: A tuple containing the actual number of layers, the probabilities
            for the number of layers, the logits for the number of layers, and the
            log probability.
        """
        if n_logits is None:
            raise ValueError("n_logits cannot be None for REINFORCE strategy.")

        n_probs = f.softmax(n_logits, dim=-1)
        m = Categorical(n_probs)
        n_action = m.sample()
        n_actual = n_action + 1
        log_prob = m.log_prob(n_action)
        return n_actual, n_probs, n_logits, log_prob

    def forward(self, x_input: torch.Tensor, n_logits: torch.Tensor) -> ForwardOutput:
        """This forward method is called by the model's main forward pass.

        :param x_input: The input tensor.
        :param n_logits: The logits for the number of layers.
        :return: A tuple containing the input tensor, the actual number of layers,
            the probabilities for the number of layers, the logits for the number of
            layers, and the log probability.
        """
        n_actual, n_probs, n_logits_out, log_prob = self.select_n(x_input, n_logits)
        return x_input, n_actual, n_probs, n_logits_out, log_prob

    def on_epoch_end(self, validation_loss: float, epoch_log_probs: Any) -> None:
        """This method is called at the end of each epoch."""
        # REINFORCE update happens in the main training loop (FlexibleHiddenLayersNN.fit)
        # The policy_optimizer.step() is called there.
        # This method can be used for any epoch-end logging or annealing specific to REINFORCE.
