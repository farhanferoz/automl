"""N-classes selection strategies for probabilistic regression models."""

import math
from typing import Any

import torch
import torch.nn.functional as f

from automl_package.enums import ProbabilisticRegressionOptimizationStrategy, RegressionStrategy
from automl_package.models.flexnn.strategies.base import BaseSelectionStrategy

# UncertaintyMethod.PROBABILISTIC heads emit (mean, log_var) -- this is that fixed width, not a tunable.
_PROBABILISTIC_HEAD_OUTPUT_SIZE = 2


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

        # CE_STOP_GRAD detaches so regression loss has no gradient path to the classifier —
        # classifier is trained by CE only (computed in _calculate_custom_loss).
        # GRADIENT_STOP only disables CE; the classifier is still supervised by the regression
        # loss through the non-detached probs path (pre-existing semantics, preserved).
        if self.model.optimization_strategy == ProbabilisticRegressionOptimizationStrategy.CE_STOP_GRAD:
            probabilities = probabilities.detach()

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
        # At eval time, use deterministic softmax — Gumbel noise makes predictions stochastic.
        mode_selection_probs = (
            f.gumbel_softmax(logits, tau=self.model.gumbel_tau, hard=False, dim=-1)
            if self.model.training
            else f.softmax(logits / self.model.gumbel_tau, dim=-1)
        )
        self.mode_selection_probs = mode_selection_probs
        final_predictions_contribution, selected_k_values_for_logging, classifier_raw_logits = self._weighted_average_logic(x_input, mode_selection_probs, boundaries=boundaries)
        per_head_outputs = self._compute_per_head_outputs_full_k(classifier_raw_logits, boundaries=boundaries)
        return final_predictions_contribution, selected_k_values_for_logging, None, classifier_raw_logits, per_head_outputs


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
        final_predictions_contribution, selected_k_values_for_logging, classifier_raw_logits = self._weighted_average_logic(x_input, mode_selection_probs, boundaries=boundaries)
        per_head_outputs = self._compute_per_head_outputs_full_k(classifier_raw_logits, boundaries=boundaries)
        return final_predictions_contribution, selected_k_values_for_logging, None, classifier_raw_logits, per_head_outputs


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
        final_predictions_contribution, selected_k_values_for_logging, classifier_raw_logits = self._hard_selection_logic(x_input, mode_selection_one_hot, boundaries=boundaries)
        per_head_outputs = self._compute_per_head_outputs_full_k(classifier_raw_logits, boundaries=boundaries)
        return final_predictions_contribution, selected_k_values_for_logging, None, classifier_raw_logits, per_head_outputs


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
        final_predictions_contribution, selected_k_values_for_logging, classifier_raw_logits = self._hard_selection_logic(x_input, mode_selection_one_hot, boundaries=boundaries)
        per_head_outputs = self._compute_per_head_outputs_full_k(classifier_raw_logits, boundaries=boundaries)
        return final_predictions_contribution, selected_k_values_for_logging, log_prob, classifier_raw_logits, per_head_outputs

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


class NestedStrategy(BaseSelectionStrategy):
    """Nested-k training: per-sample k draws as a SCHEDULE (capacity-ladder Task F9 port).

    Ports `_capacity_ladder_nested.NestedKSurrogate`'s per-sample ``k ~ Uniform{1..k_max}``
    "k-dropout" schedule (`automl_package/examples/_capacity_ladder_nested.py`) onto
    `ProbabilisticRegressionModel`'s existing k-class mixture ladder, the SAME way
    `layer_selection_strategies.NestedStrategy` ports it onto FlexNN depth: rung 1 is the
    direct-regression BYPASS (the k=1 single-Gaussian rung, `direct_regression_head`); rungs
    2..k_max are the renormalized k-class mixture `_compute_predictions_for_k` already produces
    (masked softmax over the first k classifier logits, weighted-combined through the SAME
    shared regression heads every rung reuses). That masked-softmax renormalization is exactly
    `NestedKSurrogate.masked_prefix_nll`'s prefix property -- "the first c rungs are a genuine
    c-component mixture for every c" -- so ProbReg's existing architecture gets K4's nesting
    guarantee for free, without a new component-head architecture. No `n_classes_predictor` is
    built for this strategy (`logits` is ignored; no k input to the network -- mirrors
    `layer_selection_strategies.NestedStrategy` / `NoneStrategy`).

    Known compatibility boundary (do not hide): this strategy always returns
    ``per_head_outputs=None`` (each sample trains a different rung, so there is no single
    per-head correspondence for a batch-level penalty). MDN loss (`ProbRegLossType.MDN`),
    the ordering-identifiability penalty, the middle-class NLL penalty, and boundary
    regularization all silently no-op under NESTED rather than erroring -- pair NESTED with
    `ProbRegLossType.GAUSSIAN_LTV` and leave those penalties off (the strictly-probabilistic,
    no-arbitrary-penalty premise `_capacity_ladder_nested.py` documents). Use one of the
    labeled comparison-arm strategies (GumbelSoftmax/SoftGating/STE/REINFORCE) if per-head
    penalties are required.
    """

    def setup_optimizers(self, policy_params: Any) -> None:
        """No policy optimizer needed: NESTED has no n_classes_predictor to train."""

    def on_epoch_end(self, **kwargs: Any) -> None:
        """No epoch-end action: the draw distribution is fixed (uniform), nothing anneals."""

    def all_rung_outputs(self, x_input: torch.Tensor, boundaries: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Every rung's (mean, log_var) readout, computed via the shared classifier/heads.

        Args:
            x_input: Input tensor, shape (batch, in_features).
            boundaries: Optional HARDSIGMOID boundary tensor, forwarded to
                `_compute_predictions_for_k` (ignored by the bypass rung, matching the other
                strategies' `direct_regression_head` calls).

        Returns:
            outputs: Shape (batch, n_classes, regression_output_size); column 0 is the bypass
                rung (k=1); column i (i >= 1) is the renormalized k=(i+1)-class mixture -- the
                prefix-of-first-(i+1)-rungs nesting property `_capacity_ladder_nested.py`
                requires (see class docstring).
            classifier_raw_logits: Shape (batch, n_classes) -- for downstream loss/logging
                bookkeeping (matches every other strategy's return contract).
        """
        classifier_raw_logits = self.model.classifier_layers(x_input)
        rungs = [self.model.direct_regression_head(x_input)]
        for k_val in range(2, self.model.n_classes + 1):
            rungs.append(self.model._compute_predictions_for_k(classifier_raw_logits, k_val, boundaries=boundaries))
        return torch.stack(rungs, dim=1), classifier_raw_logits

    def all_rung_log_likelihood(self, x_input: torch.Tensor, y_target: torch.Tensor) -> torch.Tensor:
        """Per-sample Gaussian log-likelihood at every rung, from ONE call to `all_rung_outputs`.

        Requires `UncertaintyMethod.PROBABILISTIC` (native (mean, log_var) heads). This is the
        all-rung score table `ProbabilisticRegressionModel.fit_global_selector`
        (capacity-programme Task PA, M1's selector) consumes -- mirrors
        `layer_selection_strategies.NestedStrategy.all_depth_log_likelihood`.
        (D6, corrected 2026-07-21: this docstring previously named
        `held_out_arbiter_advantage` as the consumer -- false, that function never calls this one;
        it calls `_per_sample_log_likelihood_at_k` twice instead. See `probreg.md` D5/D6.)

        Args:
            x_input: Input tensor, shape (batch, in_features).
            y_target: Targets, in whatever space `all_rung_outputs`' heads were trained in --
                i.e. already symlog-transformed if `target_transform="symlog"`. Unlike
                `ProbabilisticRegressionModel._per_sample_log_likelihood_at_k`, this method does
                NOT transform `y_target` itself; the caller is responsible for the transform
                (the same units-mismatch bug class as D2 -- see `fit_global_selector`'s contract
                for the pattern to follow). Shape (batch,) or (batch, 1).

        Returns:
            Shape (batch, n_classes); higher is better.
        """
        all_outputs, _ = self.all_rung_outputs(x_input)
        if all_outputs.size(-1) != _PROBABILISTIC_HEAD_OUTPUT_SIZE:
            raise ValueError("all_rung_log_likelihood requires UncertaintyMethod.PROBABILISTIC (mean, log_var) heads.")
        mean = all_outputs[..., 0]
        log_var = all_outputs[..., 1]
        variance = torch.exp(log_var)
        y = y_target.reshape(-1, 1).to(all_outputs.device)
        return -0.5 * (math.log(2 * math.pi) + log_var + (y - mean) ** 2 / variance)

    def forward(
        self, x_input: torch.Tensor, _logits: torch.Tensor | None, boundaries: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor | None]:
        """Draws a per-sample rung ~ Uniform{1..k_max}; returns that rung's own readout.

        Args:
            x_input: Input tensor.
            _logits: Ignored -- NESTED does not condition on an n_classes_predictor.
            boundaries: Optional boundaries for the HARDSIGMOID transformation.

        Returns:
            tuple: (final predictions at each sample's own drawn rung, selected k values --
            the direct-regression sentinel for the bypass rung else k=i+1, None log_prob, the
            classifier raw logits, per_head_outputs=None -- see class docstring).
        """
        all_outputs, classifier_raw_logits = self.all_rung_outputs(x_input, boundaries=boundaries)
        n_rungs = all_outputs.size(1)
        rung_idx = torch.randint(0, n_rungs, (x_input.size(0),), device=x_input.device)
        gather_index = rung_idx.view(-1, 1, 1).expand(-1, 1, all_outputs.size(-1))
        final_predictions = all_outputs.gather(1, gather_index).squeeze(1)

        inf_sentinel = self._DIRECT_REGRESSION_K_SENTINEL
        selected_k_values = torch.where(
            rung_idx == 0,
            torch.tensor(inf_sentinel, dtype=torch.long, device=x_input.device),
            (rung_idx + 1).to(torch.long),
        )
        n_probs = f.one_hot(rung_idx, num_classes=n_rungs).float()
        self.mode_selection_probs = n_probs

        return final_predictions, selected_k_values, None, classifier_raw_logits, None
