"""Internal network architecture for the ProbabilisticRegressionModel."""

from typing import Any

import torch
import torch.nn as nn

from automl_package.enums import NClassesSelectionMethod, ProbabilisticRegressionOptimizationStrategy, RegressionStrategy, TaskType, UncertaintyMethod
from automl_package.models.common.monotonicity_config_mixin import MonotonicityConfigMixin
from automl_package.models.common.regression_heads import SeparateHeadsRegressionModule, SingleHeadFinalOutputRegressionModule, SingleHeadNOutputsRegressionModule
from automl_package.models.neural_network import PyTorchNeuralNetwork
from automl_package.models.selection_strategies.n_classes_strategies import GumbelSoftmaxStrategy, NoneStrategy, ReinforceStrategy, SoftGatingStrategy, SteStrategy


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
        }
        self.n_classes_strategy = strategy_map[n_classes_selection_method](self)

        if self.n_classes_selection_method != NClassesSelectionMethod.NONE:
            self.n_classes = self.max_n_classes_for_probabilistic_path
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

    def _compute_predictions_for_k(self, classifier_raw_logits: torch.Tensor, k_val: int) -> torch.Tensor:
        """Helper to compute regression predictions for a given k_val."""
        masked_classifier_logits = torch.full_like(classifier_raw_logits, float("-inf"))
        masked_classifier_logits[:, :k_val] = classifier_raw_logits[:, :k_val]

        probabilities = torch.softmax(masked_classifier_logits, dim=1)

        if self.optimization_strategy == ProbabilisticRegressionOptimizationStrategy.GRADIENT_STOP:
            probabilities = probabilities.detach()

        if self.regression_strategy == RegressionStrategy.SINGLE_HEAD_FINAL_OUTPUT:
            class_means = self.regression_module(probabilities)
            preds = torch.sum(probabilities.unsqueeze(-1) * class_means, dim=1)
        else:
            preds = self.regression_module(probabilities)
        return preds

    def forward(self, x_input: torch.Tensor, boundaries: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Performs the forward pass through the network."""
        n_classes_predictor_logits = self.n_classes_predictor(x_input) if self.n_classes_selection_method != NClassesSelectionMethod.NONE else None

        final_predictions, selected_k_values, log_prob_for_reinforce, classifier_logits_out, per_head_outputs = self.n_classes_strategy.forward(
            x_input, n_classes_predictor_logits, boundaries=boundaries
        )

        return final_predictions, classifier_logits_out, selected_k_values, log_prob_for_reinforce, per_head_outputs

    def get_num_parameters(self) -> int:
        """Returns the total number of trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
