"""Internal network architecture for the ProbabilisticRegressionModel."""

from typing import Any

import torch
import torch.nn as nn

from automl_package.enums import NClassesSelectionMethod, ProbabilisticRegressionOptimizationStrategy, RegressionStrategy, TaskType, UncertaintyMethod
from automl_package.models.common.monotonicity_config_mixin import MonotonicityConfigMixin
from automl_package.models.common.regression_heads import SeparateHeadsRegressionModule, SingleHeadFinalOutputRegressionModule, SingleHeadNOutputsRegressionModule
from automl_package.models.neural_network import PyTorchNeuralNetwork
from automl_package.models.selection_strategies.n_classes_strategies import GumbelSoftmaxStrategy, NestedStrategy, NoneStrategy, ReinforceStrategy, SoftGatingStrategy, SteStrategy


class ProbabilisticRegressionNet(nn.Module, MonotonicityConfigMixin):
    """Internal PyTorch Module combining the classifier and regression heads."""

    def __init__(
        self,
        input_size: int,
        n_classes: int,
        n_classes_inf: float,
        max_n_classes_for_probabilistic_path: int,
        base_classifier_params: dict[str, Any],
        regression_head_params: dict[str, Any],
        direct_regression_head_params: dict[str, Any],
        regression_strategy: RegressionStrategy,
        uncertainty_method: UncertaintyMethod,
        n_classes_selection_method: NClassesSelectionMethod,
        optimization_strategy: ProbabilisticRegressionOptimizationStrategy,
        gumbel_tau: float,
        n_classes_predictor_learning_rate: float,
        device: torch.device,
        **kwargs: Any,
    ) -> None:
        """Initializes the ProbabilisticRegressionNet."""
        nn.Module.__init__(self)
        # Pop before passing to MonotonicityConfigMixin to avoid forwarding unknown kwargs to
        # object.__init__ via cooperative MRO.
        centroids = kwargs.pop("centroids", None)
        use_anchored_heads = kwargs.pop("use_anchored_heads", False)
        MonotonicityConfigMixin.__init__(self, **kwargs)
        self.input_size = input_size
        self.device = device
        self.n_classes_inf = n_classes_inf
        self.max_n_classes_for_probabilistic_path = max_n_classes_for_probabilistic_path
        self.regression_strategy = regression_strategy
        self.uncertainty_method = uncertainty_method
        self.regression_output_size = 2 if self.uncertainty_method == UncertaintyMethod.PROBABILISTIC else 1
        self.n_classes_selection_method = n_classes_selection_method
        self.optimization_strategy = optimization_strategy
        self.gumbel_tau = gumbel_tau
        self.n_classes_predictor_learning_rate = n_classes_predictor_learning_rate

        strategy_map = {
            NClassesSelectionMethod.NONE: NoneStrategy,
            NClassesSelectionMethod.GUMBEL_SOFTMAX: GumbelSoftmaxStrategy,
            NClassesSelectionMethod.SOFT_GATING: SoftGatingStrategy,
            NClassesSelectionMethod.STE: SteStrategy,
            NClassesSelectionMethod.REINFORCE: ReinforceStrategy,
            NClassesSelectionMethod.NESTED: NestedStrategy,
        }
        self.n_classes_strategy = strategy_map[n_classes_selection_method](self)

        if self.n_classes_selection_method != NClassesSelectionMethod.NONE:
            self.n_classes = self.max_n_classes_for_probabilistic_path

            if self.n_classes_selection_method == NClassesSelectionMethod.NESTED:
                # NESTED draws k as a per-sample SCHEDULE (n_classes_strategies.NestedStrategy) --
                # no k input to the network, so no n_classes_predictor is built at all.
                self.n_classes_predictor = None
            else:
                n_classes_predictor_output_size = (self.max_n_classes_for_probabilistic_path - 2 + 1) + 1
                n_classes_predictor_instance = PyTorchNeuralNetwork(
                    input_size=input_size,
                    output_size=n_classes_predictor_output_size,
                    task_type=TaskType.CLASSIFICATION,
                    device=self.device,
                    **base_classifier_params,
                )
                n_classes_predictor_instance.build_model()
                self.n_classes_predictor = n_classes_predictor_instance.get_internal_model()

            direct_regression_head_instance = PyTorchNeuralNetwork(
                input_size=input_size,
                output_size=self.regression_output_size,
                task_type=TaskType.REGRESSION,
                device=self.device,
                **direct_regression_head_params,
            )
            direct_regression_head_instance.build_model()
            self.direct_regression_head = direct_regression_head_instance.get_internal_model()
        else:
            self.n_classes = n_classes
            self.n_classes_predictor = None
            self.direct_regression_head = None

        classifier_output_size = self.n_classes
        temp_classifier_instance = PyTorchNeuralNetwork(
            input_size=input_size,
            output_size=classifier_output_size,
            task_type=TaskType.CLASSIFICATION,
            device=self.device,
            **base_classifier_params,
        )
        temp_classifier_instance.build_model()
        self.classifier_layers = temp_classifier_instance.get_internal_model()

        if self.regression_strategy == RegressionStrategy.SEPARATE_HEADS:
            self.regression_module = SeparateHeadsRegressionModule(
                n_classes=self.n_classes,
                regression_head_params=regression_head_params,
                uncertainty_method=self.uncertainty_method,
                regression_output_size=self.regression_output_size,
                use_monotonic_constraints=self.use_monotonic_constraints,
                constrain_middle_class=self.constrain_middle_class,
                centroids=centroids,
                use_anchored_heads=use_anchored_heads,
            )
        elif self.regression_strategy == RegressionStrategy.SINGLE_HEAD_N_OUTPUTS:
            self.regression_module = SingleHeadNOutputsRegressionModule(
                input_size=self.n_classes,
                n_classes=self.n_classes,
                regression_head_params=regression_head_params,
                uncertainty_method=self.uncertainty_method,
                regression_output_size=self.regression_output_size,
                constrain_middle_class=self.constrain_middle_class,
            )
        elif self.regression_strategy == RegressionStrategy.SINGLE_HEAD_FINAL_OUTPUT:
            self.regression_module = SingleHeadFinalOutputRegressionModule(
                input_size=self.n_classes,
                regression_head_params=regression_head_params,
                uncertainty_method=self.uncertainty_method,
                regression_output_size=self.regression_output_size,
            )
        else:
            raise ValueError(f"Unknown regression_strategy: {regression_strategy}")

    def _compute_predictions_for_k(self, classifier_raw_logits: torch.Tensor, k_val: int, boundaries: torch.Tensor | None = None) -> torch.Tensor:
        """Helper to compute regression predictions for a given k_val."""
        masked_classifier_logits = torch.full_like(classifier_raw_logits, float("-inf"))
        masked_classifier_logits[:, :k_val] = classifier_raw_logits[:, :k_val]

        probabilities = torch.softmax(masked_classifier_logits, dim=1)

        # CE_STOP_GRAD detaches probs so regression loss has no gradient path to the classifier.
        # GRADIENT_STOP preserves pre-existing semantics: detach in the dynamic-k path so the
        # k-selection head isn't pulled by regression gradient through probabilities.
        if self.optimization_strategy in (
            ProbabilisticRegressionOptimizationStrategy.GRADIENT_STOP,
            ProbabilisticRegressionOptimizationStrategy.CE_STOP_GRAD,
        ):
            probabilities = probabilities.detach()

        return self.regression_module(probabilities, boundaries=boundaries)

    def forward(self, x_input: torch.Tensor, boundaries: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Performs the forward pass through the network."""
        # NESTED builds no n_classes_predictor (no k input to the network -- see
        # NestedStrategy), so this gate is on the predictor's existence, not just the
        # selection method, and naturally yields None logits for it.
        n_classes_predictor_logits = self.n_classes_predictor(x_input) if self.n_classes_predictor is not None else None

        final_predictions, selected_k_values, log_prob_for_reinforce, classifier_logits_out, per_head_outputs = self.n_classes_strategy.forward(
            x_input, n_classes_predictor_logits, boundaries=boundaries
        )

        return final_predictions, classifier_logits_out, selected_k_values, log_prob_for_reinforce, per_head_outputs

    def forward_at_k(self, x_input: torch.Tensor, k: int, boundaries: torch.Tensor | None = None) -> torch.Tensor:
        """Forces every sample through rung `k`, bypassing the selection strategy entirely.

        `k=1` is the direct-regression bypass (`direct_regression_head`); `k>=2` is the
        renormalized k-class mixture (`_compute_predictions_for_k`). Used by
        `ProbabilisticRegressionModel.fit_router`'s per-capacity held-out error table and by
        `held_out_arbiter_advantage`'s certified readout (capacity-programme Task F9) --
        mirrors `FlexibleNNModule.forward_at_depth` / `FlexibleWidthNNModule.forward_width`.

        Raises:
            RuntimeError: `k=1` requested but no `direct_regression_head` was built (only
                a dynamic `n_classes_selection_method != NONE` model has a bypass rung).
            ValueError: `k` outside `[1, self.n_classes]`.
        """
        if not (1 <= k <= self.n_classes):
            raise ValueError(f"k={k} out of range [1, {self.n_classes}]")
        if k == 1:
            if self.direct_regression_head is None:
                raise RuntimeError("forward_at_k(k=1) requires a dynamic n_classes_selection_method (no direct_regression_head built).")
            return self.direct_regression_head(x_input)
        classifier_raw_logits = self.classifier_layers(x_input)
        return self._compute_predictions_for_k(classifier_raw_logits, k, boundaries=boundaries)

    def get_num_parameters(self) -> int:
        """Returns the total number of trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
