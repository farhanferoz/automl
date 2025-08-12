"""Probabilistic Regression model implemented in PyTorch."""

import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from ..enums import NClassesSelectionMethod, RegressionStrategy, TaskType, UncertaintyMethod
from ..logger import logger
from .base_pytorch import PyTorchModelBase
from .neural_network import PyTorchNeuralNetwork
from .selection_strategies.n_classes_strategies import (
    GumbelSoftmaxStrategy,
    NoneStrategy,
    ReinforceStrategy,
    SoftGatingStrategy,
    SteStrategy,
)


class ProbabilisticRegressionModel(PyTorchModelBase):
    """A PyTorch-based probabilistic regression model that directly learns both mean and variance.

    Can use different strategies for outputting mean and variance.
    """

    def __init__(
        self,
        input_size: int = None,
        n_classes: int = 3,  # Number of classes for the internal classifier (used if n_classes_selection_method is NONE)
        n_classes_inf: float = float("inf"),  # Threshold for direct regression
        max_n_classes_for_probabilistic_path: int = 10,  # Max n_classes if n_classes_selection_method is not NONE
        base_classifier_params: dict[str, Any] = None,  # Params for the internal PyTorch NN classifier
        regression_head_params: dict[str, Any] = None,  # Params for each internal PyTorch NN regression head
        direct_regression_head_params: dict[str, Any] = None,  # Params for the direct regression head if n_classes_selection_method is not NONE
        regression_strategy: RegressionStrategy = RegressionStrategy.SEPARATE_HEADS,  # Use enum
        n_classes_selection_method: NClassesSelectionMethod = NClassesSelectionMethod.NONE,
        gumbel_tau: float = 0.5,
        n_classes_predictor_learning_rate: float = 0.001,
        learning_rate: float = 0.001,
        n_epochs: int = 10,
        batch_size: int = 32,
        uncertainty_method: UncertaintyMethod = UncertaintyMethod.CONSTANT,  # Use enum
        n_mc_dropout_samples: int = 100,
        dropout_rate: float = 0.1,
        **kwargs,
    ):
        """Initializes the ProbabilisticRegressionModel.

        Args:
            input_size (int, optional): The number of input features.
            n_classes (int): Number of classes for the internal classifier.
            n_classes_inf (float): Threshold for direct regression.
            max_n_classes_for_probabilistic_path (int): Max n_classes if n_classes_selection_method is not NONE.
            base_classifier_params (dict[str, Any]): Parameters for the internal PyTorch NN classifier.
            regression_head_params (dict[str, Any]): Parameters for each internal PyTorch NN regression head.
            direct_regression_head_params (dict[str, Any]): Parameters for the direct regression head.
            regression_strategy (RegressionStrategy): Strategy for regression.
            n_classes_selection_method (NClassesSelectionMethod): Method for selecting n_classes.
            gumbel_tau (float): Gumbel-Softmax temperature.
            n_classes_predictor_learning_rate (float): Learning rate for n_classes predictor.
            learning_rate (float): Learning rate for the model.
            n_epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
            uncertainty_method (UncertaintyMethod): Method for uncertainty estimation.
            n_mc_dropout_samples (int): Number of MC dropout samples for uncertainty.
            dropout_rate (float): Dropout rate for MC dropout.
            **kwargs: Additional keyword arguments for PyTorchModelBase.
        """
        output_size = 2 if uncertainty_method == UncertaintyMethod.PROBABILISTIC else 1
        super().__init__(
            input_size=input_size,
            output_size=output_size,
            learning_rate=learning_rate,
            n_epochs=n_epochs,
            batch_size=batch_size,
            task_type=TaskType.REGRESSION,
            uncertainty_method=uncertainty_method,
            n_mc_dropout_samples=n_mc_dropout_samples,
            dropout_rate=dropout_rate,
            **kwargs,
        )
        self._returns_multiple_outputs = True  # This model returns multiple outputs
        self.n_classes = n_classes
        self.n_classes_inf = n_classes_inf
        self.max_n_classes_for_probabilistic_path = max_n_classes_for_probabilistic_path
        self.base_classifier_params = base_classifier_params if base_classifier_params is not None else {}
        self.regression_head_params = regression_head_params if regression_head_params is not None else {}
        self.direct_regression_head_params = direct_regression_head_params if direct_regression_head_params is not None else {}
        self.regression_strategy = regression_strategy
        self.n_classes_selection_method = n_classes_selection_method
        self.gumbel_tau = gumbel_tau
        self.n_classes_predictor_learning_rate = n_classes_predictor_learning_rate

        self.direct_regression = self.n_classes_selection_method == NClassesSelectionMethod.NONE and self.n_classes >= self.n_classes_inf
        if self.direct_regression:
            logger.info(f"Number of classes ({self.n_classes}) >= n_classes_inf ({self.n_classes_inf}). Using direct regression mode.")
        elif self.n_classes_selection_method == NClassesSelectionMethod.NONE:
            logger.info(f"Using probabilistic regression mode with fixed {self.n_classes} classes.")
        else:
            logger.info(f"Using probabilistic regression mode with dynamic n_classes selection via {self.n_classes_selection_method.value}.")

        # Validate regression strategy and uncertainty method
        if self.regression_strategy not in [RegressionStrategy.SEPARATE_HEADS, RegressionStrategy.SINGLE_HEAD_N_OUTPUTS, RegressionStrategy.SINGLE_HEAD_FINAL_OUTPUT]:
            raise ValueError(f"Unsupported regression_strategy: {self.regression_strategy.value}. Choose from enum values.")
        if self.uncertainty_method not in [UncertaintyMethod.CONSTANT, UncertaintyMethod.MC_DROPOUT, UncertaintyMethod.PROBABILISTIC]:
            raise ValueError(f"Unsupported uncertainty_method: {self.uncertainty_method.value}. Choose from enum values.")

    @property
    def name(self) -> str:
        """Returns the name of the model."""
        return "ProbabilisticRegression"

    def build_model(self):
        """Builds the internal PyTorch nn.Module for the ProbabilisticRegressionModel."""
        if self.n_classes_selection_method == NClassesSelectionMethod.NONE:
            regression_output_size = 2 if self.uncertainty_method == UncertaintyMethod.PROBABILISTIC else 1
            temp_model = PyTorchNeuralNetwork(input_size=self.input_size, output_size=regression_output_size, task_type=TaskType.REGRESSION, **self.base_classifier_params)
            temp_model.build_model()
            self.model = temp_model.get_internal_model()  # Get the internal nn.Module
        else:
            if self.max_n_classes_for_probabilistic_path >= self.n_classes_inf:
                raise ValueError("max_n_classes_for_probabilistic_path must be less than n_classes_inf when n_classes_selection_method is not NONE.")
            self.model = _CombinedProbabilisticModel(
                input_size=self.input_size,
                n_classes=self.n_classes,  # This n_classes is the user-specified default, not the estimated one
                n_classes_inf=self.n_classes_inf,
                max_n_classes_for_probabilistic_path=self.max_n_classes_for_probabilistic_path,
                base_classifier_params=self.base_classifier_params,
                regression_head_params=self.regression_head_params,
                direct_regression_head_params=self.direct_regression_head_params,
                regression_strategy=self.regression_strategy,
                uncertainty_method=self.uncertainty_method,
                n_classes_selection_method=self.n_classes_selection_method,
                gumbel_tau=self.gumbel_tau,
                n_classes_predictor_learning_rate=self.n_classes_predictor_learning_rate,
            )
        self.model.to(self.device)

        # Set up criterion based on task type and uncertainty method
        if self.uncertainty_method == UncertaintyMethod.PROBABILISTIC:
            # Custom Negative Log-Likelihood Loss for Gaussian output
            def nll_loss(outputs, targets):
                mean = outputs[:, 0]
                variance = outputs[:, 1]
                # Ensure targets has the same shape as mean for element-wise operations
                targets = targets.squeeze(-1) if targets.ndim > 1 else targets
                # Add a small epsilon to variance to prevent log(0) or division by zero
                variance = torch.clamp(variance, min=1e-6)
                # Calculate per-sample NLL
                per_sample_nll = 0.5 * (torch.log(2 * math.pi * variance) + (targets - mean) ** 2 / variance)
                # Average over the batch
                loss = torch.mean(per_sample_nll)
                return loss

            self.criterion = nll_loss
        else:
            self.criterion = nn.MSELoss()

    def get_hyperparameter_search_space(self) -> dict[str, Any]:
        """Defines the hyperparameter search space for ProbabilisticRegressionModel.

        Returns:
            Dict[str, Any]: A dictionary defining the hyperparameter search space.
        """
        space = super().get_hyperparameter_search_space()
        space.update(
            {
                "regression_strategy": {"type": "categorical", "choices": [s.value for s in RegressionStrategy]},
                "n_classes_selection_method": {"type": "categorical", "choices": [s.value for s in NClassesSelectionMethod]},
                "gumbel_tau": {"type": "float", "low": 1e-8, "high": 1.0, "log": True},
                "n_classes_predictor_learning_rate": {"type": "float", "low": 1e-8, "high": 1e-2, "log": True},
                "base_classifier_params__hidden_layers": {"type": "int", "low": 1, "high": 2},
                "base_classifier_params__hidden_size": {"type": "int", "low": 32, "high": 64, "step": 32},
                "base_classifier_params__use_batch_norm": {"type": "categorical", "choices": [True, False]},
                "base_classifier_params__dropout_rate": {"type": "float", "low": 0.0, "high": 0.5, "step": 0.1},
                "regression_head_params__hidden_layers": {"type": "int", "low": 0, "high": 1},
                "regression_head_params__hidden_size": {"type": "int", "low": 16, "high": 32, "step": 16},
                "regression_head_params__use_batch_norm": {"type": "categorical", "choices": [True, False]},
                "regression_head_params__dropout_rate": {"type": "float", "low": 0.0, "high": 0.5, "step": 0.1},
            }
        )

        if self.n_classes_selection_method == NClassesSelectionMethod.NONE:
            space["n_classes"] = {"type": "int", "low": 2, "high": int(self.n_classes_inf) - 1 if self.n_classes_inf != float("inf") else 5}
        else:
            space["max_n_classes_for_probabilistic_path"] = {"type": "int", "low": 2, "high": int(self.n_classes_inf) - 1 if self.n_classes_inf != float("inf") else 10}
            space["direct_regression_head_params__hidden_layers"] = {"type": "int", "low": 1, "high": 2}
            space["direct_regression_head_params__hidden_size"] = {"type": "int", "low": 32, "high": 64, "step": 32}
            space["direct_regression_head_params__use_batch_norm"] = {"type": "categorical", "choices": [True, False]}
            space["direct_regression_head_params__dropout_rate"] = {"type": "float", "low": 0.0, "high": 0.5, "step": 0.1}

        return space

    def _setup_optimizers(self, model):
        super()._setup_optimizers(model)
        if self.n_classes_selection_method != NClassesSelectionMethod.NONE:
            if hasattr(self.model, "n_classes_predictor") and self.model.n_classes_predictor is not None:
                n_classes_predictor_params = self.model.n_classes_predictor.parameters()
                self.model.n_classes_strategy.setup_optimizers(n_classes_predictor_params)

    def get_classifier_predictions(self, X: np.ndarray, y_true_original: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns the internal classifier's predicted classes, probabilities, and.

        the corresponding (discretized) true labels for this composite model.

        Args:
            X (np.ndarray): Feature matrix.
            y_true_original (np.ndarray): Original true labels (will be discretized internally).

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                - Predicted classes from the internal classifier.
                - Predicted probabilities from the internal classifier.
                - Discretized true labels corresponding to the internal classifier's task.
        """
        if self.direct_regression:
            raise NotImplementedError("get_classifier_predictions is not available in direct regression mode.")
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")

        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            # The model's forward pass now returns predictions, classifier_logits_out, and selected_k_values
            _, returned_classifier_logits, selected_k_values_tensor, _ = self.model(X_tensor)

            # Initialize selected_k_values and n_classes_for_classifier_output
            selected_k_values = selected_k_values_tensor.cpu().numpy()
            n_classes_for_classifier_output = self.n_classes
            probabilistic_indices = np.where(selected_k_values < self.n_classes_inf)[0]

            if self.n_classes_selection_method != NClassesSelectionMethod.NONE:
                if len(probabilistic_indices) == 0:
                    raise NotImplementedError("No probabilistic predictions were made for the given X. All samples went to direct regression.")
                n_classes_for_classifier_output = self.max_n_classes_for_probabilistic_path

            y_flat = y_true_original.flatten() if y_true_original.ndim > 1 else y_true_original
            y_true_discretized = np.zeros_like(y_flat, dtype=int)

            # Precompute class boundaries for all possible k values
            precomputed_class_boundaries = {}
            # Iterate up to max_n_classes_for_probabilistic_path if dynamic, else self.n_classes
            max_k_val_for_boundaries = self.max_n_classes_for_probabilistic_path if self.n_classes_selection_method != NClassesSelectionMethod.NONE else self.n_classes
            for k_val in range(2, int(self.n_classes_inf) if self.n_classes_inf != float("inf") else max_k_val_for_boundaries + 1):
                percentiles = np.linspace(0, 100, k_val + 1)[1:-1]
                precomputed_class_boundaries[k_val] = np.percentile(y_flat, percentiles)

            for i in range(y_true_original.shape[0]):
                k = selected_k_values[i]
                if k < self.n_classes_inf:  # Only discretize if it's a probabilistic mode
                    boundaries = precomputed_class_boundaries[k]
                    y_true_discretized[i] = np.digitize(y_flat[i], boundaries)
                else:
                    y_true_discretized[i] = -1  # Indicate no discretization for direct regression samples

            # Use the returned_classifier_logits directly
            classifier_logits_for_proba = returned_classifier_logits[probabilistic_indices]

            # Re-apply softmax to get probabilities for the probabilistic samples
            # Need to re-mask and softmax as the stored logits might be from the full max_n_classes_allowed
            masked_classifier_logits = torch.full_like(classifier_logits_for_proba, float("-inf"))
            for i, k in enumerate(selected_k_values[probabilistic_indices]):
                masked_classifier_logits[i, :k] = classifier_logits_for_proba[i, :k]

            if n_classes_for_classifier_output == 2:
                proba_positive = torch.sigmoid(masked_classifier_logits[:, 0])  # Assuming binary classification for classifier
                y_proba_internal = torch.cat((1 - proba_positive.unsqueeze(1), proba_positive.unsqueeze(1)), dim=1).cpu().numpy()
            else:
                y_proba_internal = torch.softmax(masked_classifier_logits, dim=1).cpu().numpy()

            y_pred_internal = np.argmax(y_proba_internal, axis=1)

        return y_pred_internal, y_proba_internal, y_true_discretized


class _BaseRegressionHead(nn.Module):
    """A single regression head for ProbabilisticRegressionModel.

    Handles its own layers and probabilistic output processing.
    """

    def __init__(self, input_size: int, output_size: int, hidden_layers: int, hidden_size: int, use_batch_norm: bool, dropout_rate: float, uncertainty_method: UncertaintyMethod):
        super().__init__()
        self.uncertainty_method = uncertainty_method
        self.output_size = output_size

        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_size))
        layers.append(nn.ReLU())

        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            if dropout_rate > 0 and self.uncertainty_method == UncertaintyMethod.MC_DROPOUT:
                layers.append(nn.Dropout(dropout_rate))

        layers.append(nn.Linear(hidden_size, output_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.layers(x)
        return self.process_output(output)

    def process_output(self, raw_output: torch.Tensor) -> torch.Tensor:
        if self.uncertainty_method == UncertaintyMethod.PROBABILISTIC and self.output_size == 2:
            mean = raw_output[:, 0].unsqueeze(1)
            log_var = raw_output[:, 1].unsqueeze(1)
            var = torch.exp(log_var)
            # Ensure var is not zero or negative for log
            var = torch.clamp(var, min=1e-6)
            return torch.cat((mean, var), dim=1)
        return raw_output


class _SeparateHeadsRegressionModule(nn.Module):
    """Manages multiple _BaseRegressionHead instances for the SEPARATE_HEADS strategy."""

    def __init__(self, n_classes: int, regression_head_params: dict[str, Any], uncertainty_method: UncertaintyMethod, regression_output_size: int):
        super().__init__()
        self.heads = nn.ModuleList()
        for _ in range(n_classes):
            self.heads.append(
                _BaseRegressionHead(
                    input_size=1, output_size=regression_output_size, uncertainty_method=uncertainty_method, **regression_head_params  # Each head takes a single probability
                )
            )

    def forward(self, probabilities: torch.Tensor) -> torch.Tensor:
        final_predictions = torch.zeros(probabilities.size(0), self.heads[0].output_size).to(probabilities.device)
        for i in range(len(self.heads)):
            p_i = probabilities[:, i].unsqueeze(1)
            y_i_processed = self.heads[i](p_i)
            final_predictions += p_i * y_i_processed
        return final_predictions


class _SingleHeadNOutputsRegressionModule(nn.Module):
    """Manages a single _BaseRegressionHead instance for the SINGLE_HEAD_N_OUTPUTS strategy."""

    def __init__(self, input_size: int, n_classes: int, regression_head_params: dict[str, Any], uncertainty_method: UncertaintyMethod, regression_output_size: int):
        super().__init__()
        self.n_classes = n_classes
        self.regression_output_size = regression_output_size
        self.head = _BaseRegressionHead(input_size=input_size, output_size=n_classes * regression_output_size, uncertainty_method=uncertainty_method, **regression_head_params)

    def forward(self, head_input_probas: torch.Tensor) -> torch.Tensor:
        y_output_all_classes = self.head(head_input_probas)
        y_output_all_classes = y_output_all_classes.reshape(head_input_probas.shape[0], self.n_classes, self.regression_output_size)
        return y_output_all_classes


class _SingleHeadFinalOutputRegressionModule(nn.Module):
    """Manages a single _BaseRegressionHead instance for the SINGLE_HEAD_FINAL_OUTPUT strategy."""

    def __init__(self, input_size: int, regression_head_params: dict[str, Any], uncertainty_method: UncertaintyMethod, regression_output_size: int):
        super().__init__()
        self.head = _BaseRegressionHead(input_size=input_size, output_size=regression_output_size, uncertainty_method=uncertainty_method, **regression_head_params)

    def forward(self, head_input_probas: torch.Tensor) -> torch.Tensor:
        return self.head(head_input_probas)


class _CombinedProbabilisticModel(nn.Module):
    """Internal PyTorch Module combining the classifier and regression heads.

    This is the actual neural network structure for ProbabilisticRegressionModel.
    """

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
        gumbel_tau: float,
        n_classes_predictor_learning_rate: float,
    ):
        super().__init__()
        self.n_classes_inf = n_classes_inf
        self.max_n_classes_for_probabilistic_path = max_n_classes_for_probabilistic_path
        self.regression_strategy = regression_strategy
        self.uncertainty_method = uncertainty_method
        self.regression_output_size = 2 if self.uncertainty_method == UncertaintyMethod.PROBABILISTIC else 1
        self.n_classes_selection_method = n_classes_selection_method
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
            # n_classes_predictor: outputs logits for (max_n_classes_for_probabilistic_path - 2 + 1) modes
            # (2 to max_n_classes_for_probabilistic_path) + 1 for direct regression
            n_classes_predictor_output_size = (self.max_n_classes_for_probabilistic_path - 2 + 1) + 1
            self.n_classes_predictor = PyTorchNeuralNetwork(
                input_size=input_size, output_size=n_classes_predictor_output_size, task_type=TaskType.CLASSIFICATION, **base_classifier_params
            ).get_internal_model()

            # Direct regression head
            self.direct_regression_head = PyTorchNeuralNetwork(
                input_size=input_size, output_size=self.regression_output_size, task_type=TaskType.REGRESSION, **direct_regression_head_params
            ).get_internal_model()
        else:
            self.n_classes = n_classes
            self.n_classes_predictor = None
            self.direct_regression_head = None

        # Classifier part: Use PyTorchNeuralNetwork's internal architecture logic
        classifier_output_size = 1 if self.n_classes == 2 else self.n_classes
        temp_classifier_instance = PyTorchNeuralNetwork(input_size=input_size, output_size=classifier_output_size, task_type=TaskType.CLASSIFICATION, **base_classifier_params)
        self.classifier_layers = temp_classifier_instance.model

        if self.regression_strategy == RegressionStrategy.SEPARATE_HEADS:
            self.regression_module = _SeparateHeadsRegressionModule(
                n_classes=self.n_classes,
                regression_head_params=regression_head_params,
                uncertainty_method=self.uncertainty_method,
                regression_output_size=self.regression_output_size,
            )
        elif self.regression_strategy == RegressionStrategy.SINGLE_HEAD_N_OUTPUTS:
            self.regression_module = _SingleHeadNOutputsRegressionModule(
                input_size=self.n_classes if self.n_classes == 1 else self.n_classes - 1,
                n_classes=self.n_classes,
                regression_head_params=regression_head_params,
                uncertainty_method=self.uncertainty_method,
                regression_output_size=self.regression_output_size,
            )
        elif self.regression_strategy == RegressionStrategy.SINGLE_HEAD_FINAL_OUTPUT:
            self.regression_module = _SingleHeadFinalOutputRegressionModule(
                input_size=self.n_classes if self.n_classes == 1 else self.n_classes - 1,
                regression_head_params=regression_head_params,
                uncertainty_method=self.uncertainty_method,
                regression_output_size=self.regression_output_size,
            )
        else:
            raise ValueError(f"Unknown regression_strategy: {regression_strategy}")

    def _compute_predictions_for_k(self, x_input: torch.Tensor, k_val: int) -> torch.Tensor:
        """Helper to compute regression predictions for a given k_val."""
        classifier_raw_logits = self.classifier_layers(x_input)

        masked_classifier_logits = torch.full_like(classifier_raw_logits, float("-inf"))
        masked_classifier_logits[:, :k_val] = classifier_raw_logits[:, :k_val]

        probabilities = torch.softmax(masked_classifier_logits, dim=1)
        predictions = self.regression_module(probabilities)
        return predictions

    def forward(self, x_input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if self.n_classes_selection_method != NClassesSelectionMethod.NONE:
            n_classes_predictor_logits = self.n_classes_predictor(x_input)
        else:
            n_classes_predictor_logits = None  # Not used for NONE strategy

        final_predictions, selected_k_values, log_prob_for_reinforce = self.n_classes_strategy.forward(x_input, n_classes_predictor_logits)

        # For classifier_logits_out, we need to decide what to return.
        # For soft methods, it's not a single set of logits.
        # For simplicity, let's return the logits corresponding to the argmax selected k.
        # This is primarily for the get_classifier_predictions method.
        # This part might need further refinement depending on how get_classifier_predictions is used.
        classifier_logits_out = torch.zeros(x_input.size(0), self.n_classes).to(x_input.device)
        if self.n_classes_selection_method != NClassesSelectionMethod.NONE:
            # Exclude direct reg mode from argmax for classifier_logits_out
            # This assumes that the strategy has a 'mode_selection_probs' attribute after forward pass
            # which is true for GumbelSoftmaxStrategy and SoftGatingStrategy.
            # For STE and REINFORCE, it's a hard one-hot, so argmax is still valid.
            argmax_k_indices = torch.argmax(self.n_classes_strategy.mode_selection_probs[:, :-1], dim=1)
            for i in range(x_input.size(0)):
                k_for_classifier_out = argmax_k_indices[i] + 2  # Convert index to k value
                # Ensure that classifier_layers is called with the correct input shape
                classifier_logits_out[i, :k_for_classifier_out] = self.classifier_layers(x_input[i].unsqueeze(0))[:, :k_for_classifier_out]
        else:
            classifier_logits_out = self.classifier_layers(x_input)

        return final_predictions, classifier_logits_out, selected_k_values, log_prob_for_reinforce
