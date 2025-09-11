# ruff: noqa: ERA001
"""Probabilistic Regression model implemented in PyTorch."""

import math
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from automl_package.enums import ExplainerType, NClassesSelectionMethod, RegressionStrategy, TaskType, UncertaintyMethod
from automl_package.logger import logger
from automl_package.models.base_pytorch import PyTorchModelBase
from automl_package.models.common.regression_heads import SeparateHeadsRegressionModule, SingleHeadFinalOutputRegressionModule, SingleHeadNOutputsRegressionModule
from automl_package.models.neural_network import PyTorchNeuralNetwork
from automl_package.models.selection_strategies.n_classes_strategies import GumbelSoftmaxStrategy, NoneStrategy, ReinforceStrategy, SoftGatingStrategy, SteStrategy
from automl_package.utils.losses import nll_loss
from automl_package.utils.numerics import create_bins
from automl_package.utils.plotting import plot_nn_probability_mappers


class ProbabilisticRegressionModel(PyTorchModelBase):
    """A PyTorch-based probabilistic regression model that directly learns both mean and variance.

    Can use different strategies for outputting mean and variance.
    """

    _defaults: ClassVar[dict[str, Any]] = {
        "input_size": None,
        "n_classes": 3,
        "n_classes_inf": float("inf"),
        "max_n_classes_for_probabilistic_path": 10,
        "base_classifier_params": None,
        "regression_head_params": None,
        "direct_regression_head_params": None,
        "regression_strategy": RegressionStrategy.SEPARATE_HEADS,
        "n_classes_selection_method": NClassesSelectionMethod.NONE,
        "gumbel_tau": 0.5,
        "n_classes_predictor_learning_rate": 0.001,
        "add_classification_loss": False,
    }

    def __init__(self, **kwargs: Any) -> None:
        """Initializes the ProbabilisticRegressionModel."""
        for key, value in self._defaults.items():
            kwargs.setdefault(key, value)

        if kwargs.get("add_classification_loss") and kwargs.get("uncertainty_method") != UncertaintyMethod.PROBABILISTIC:
            logger.warning(
                f"add_classification_loss=True requires a probabilistic uncertainty method. "
                f"Overriding uncertainty_method from {kwargs.get('uncertainty_method').value} to {UncertaintyMethod.PROBABILISTIC.value}."
            )
            kwargs["uncertainty_method"] = UncertaintyMethod.PROBABILISTIC

        output_size = 2 if kwargs.get("uncertainty_method") == UncertaintyMethod.PROBABILISTIC else 1
        kwargs["output_size"] = output_size

        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)

        if isinstance(self.regression_strategy, str):
            self.regression_strategy = RegressionStrategy[self.regression_strategy.upper()]
        if isinstance(self.n_classes_selection_method, str):
            self.n_classes_selection_method = NClassesSelectionMethod[self.n_classes_selection_method.upper()]

        self.base_classifier_params = self.base_classifier_params if self.base_classifier_params is not None else {}
        self.regression_head_params = self.regression_head_params if self.regression_head_params is not None else {}
        self.direct_regression_head_params = self.direct_regression_head_params if self.direct_regression_head_params is not None else {}

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
        return f"ProbabilisticRegression_{self.regression_strategy.value}"

    def _calculate_custom_loss(self, model_outputs: tuple, y_true: torch.Tensor) -> torch.Tensor:
        """Calculates the loss for the ProbabilisticRegressionModel.

        This can be a composite loss including regression and classification components.
        """
        final_predictions, classifier_logits_out, selected_k_values, _ = model_outputs
        y_true_squeezed = y_true.squeeze(-1) if y_true.ndim > 1 else y_true

        # 1. Calculate Regression Loss
        if self.uncertainty_method == UncertaintyMethod.PROBABILISTIC:
            mean = final_predictions[:, 0]
            log_var = torch.log(torch.clamp(final_predictions[:, 1], min=1e-6))
            regression_loss = nll_loss(torch.stack((mean, log_var), dim=1), y_true_squeezed)
        else:
            assert not self.add_classification_loss
            regression_loss = nn.MSELoss()(final_predictions.squeeze(), y_true_squeezed)

        total_loss = regression_loss

        # 2. Optionally, add Classification Loss
        if self.add_classification_loss:
            # We only add classification loss for samples that went through the probabilistic path
            probabilistic_indices = torch.where(selected_k_values < self.n_classes_inf)[0]

            if probabilistic_indices.numel() > 0:
                y_true_prob = y_true_squeezed[probabilistic_indices]
                logits_prob = classifier_logits_out[probabilistic_indices]
                k_values_prob = selected_k_values[probabilistic_indices]

                # Discretize y_true based on the k value chosen for each sample
                y_binned_prob = torch.zeros_like(y_true_prob, dtype=torch.long)
                unique_k = torch.unique(k_values_prob)

                for k in unique_k:
                    k_int = int(k.item())
                    mask = k_values_prob == k
                    boundaries = self.precomputed_class_boundaries[k_int]
                    _, y_binned_k = create_bins(data=y_true_prob[mask].cpu().numpy(), unique_bin_edges=boundaries)
                    y_binned_prob[mask] = torch.tensor(y_binned_k, dtype=torch.long, device=self.device)

                # Mask logits for valid classes based on k
                max_k_in_batch = logits_prob.shape[1]
                col_indices = torch.arange(max_k_in_batch, device=self.device)
                mask = col_indices < k_values_prob.unsqueeze(1)
                masked_logits = torch.where(mask, logits_prob, float("-inf"))

                classification_loss = nn.CrossEntropyLoss()(masked_logits, y_binned_prob)
                total_loss += classification_loss

        return total_loss

    def build_model(self) -> None:
        """Builds the internal PyTorch nn.Module for the ProbabilisticRegressionModel."""
        if self.n_classes_selection_method != NClassesSelectionMethod.NONE and self.max_n_classes_for_probabilistic_path >= self.n_classes_inf:
            raise ValueError("max_n_classes_for_probabilistic_path must be less than n_classes_inf when n_classes_selection_method is not NONE.")

        self.model = _CombinedProbabilisticModel(
            input_size=self.input_size,
            n_classes=self.n_classes,
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
            device=self.device,
        )
        self.model.to(self.device)

        # Set up criterion to our custom loss function
        self.criterion = self._calculate_custom_loss

    def get_hyperparameter_search_space(self) -> dict[str, Any]:
        """Defines the hyperparameter search space for ProbabilisticRegressionModel.

        Returns:
            Dict[str, Any]: A dictionary defining the hyperparameter search space.
        """
        space = super().get_hyperparameter_search_space()
        space.update(
            {
                "regression_strategy": {"type": "categorical", "choices": [s.value for s in RegressionStrategy]},
                # "n_classes_selection_method": {"type": "categorical", "choices": [s.value for s in NClassesSelectionMethod]},
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
            space["n_classes"] = {"type": "int", "low": 2, "high": (int(self.n_classes_inf) - 1 if self.n_classes_inf != float("inf") else 5)}
        else:
            space["max_n_classes_for_probabilistic_path"] = {"type": "int", "low": 2, "high": (int(self.n_classes_inf) - 1 if self.n_classes_inf != float("inf") else 10)}
            space["direct_regression_head_params__hidden_layers"] = {"type": "int", "low": 1, "high": 2}
            space["direct_regression_head_params__hidden_size"] = {"type": "int", "low": 32, "high": 64, "step": 32}
            space["direct_regression_head_params__use_batch_norm"] = {"type": "categorical", "choices": [True, False]}
            space["direct_regression_head_params__dropout_rate"] = {"type": "float", "low": 0.0, "high": 0.5, "step": 0.1}

        if self.search_space_override:
            space.update(self.search_space_override)

        return space

    def get_internal_model(self) -> Any:
        """Returns a wrapper around the internal model that is compatible with SHAP."""

        class _ShapModelWrapper(nn.Module):
            def __init__(self, model: nn.Module) -> None:
                super().__init__()
                self.model = model

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.model(x)[0]

        return _ShapModelWrapper(self.model)

    def _setup_optimizers(self, model: nn.Module) -> None:
        super()._setup_optimizers(model)
        if self.n_classes_selection_method != NClassesSelectionMethod.NONE and hasattr(self.model, "n_classes_predictor") and self.model.n_classes_predictor is not None:
            n_classes_predictor_params = self.model.n_classes_predictor.parameters()
            self.model.n_classes_strategy.setup_optimizers(n_classes_predictor_params)

    def _fit_single(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        forced_iterations: int | None = None,
    ) -> tuple[int, list[float]]:
        """Fits a single model instance.

        Args:
            x_train (np.ndarray): The training features.
            y_train (np.ndarray): The training targets.
            x_val (np.ndarray | None): The validation features.
            y_val (np.ndarray | None): The validation targets.
            forced_iterations (int | None): If provided, train for this many iterations, ignoring early stopping.

        Returns:
            tuple[int, list[float]]: A tuple containing:
                - The number of iterations the model was trained for.
                - A list of the validation loss values for each epoch.
        """
        if not self.direct_regression:
            self.precomputed_class_boundaries = {}
            y_flat = y_train.flatten() if y_train.ndim > 1 else y_train

            max_k = self.max_n_classes_for_probabilistic_path if self.n_classes_selection_method != NClassesSelectionMethod.NONE else self.n_classes

            # Pre-calculate and store boundaries for all possible k values
            for k in [max_k] if self.n_classes_selection_method == NClassesSelectionMethod.NONE else range(2, max_k + 1):
                boundaries, _ = create_bins(data=y_flat, n_bins=k, min_value=-np.inf, max_value=np.inf)
                self.precomputed_class_boundaries[k] = boundaries

        return super()._fit_single(x_train, y_train, x_val, y_val, forced_iterations)

    def _update_params(self, params: dict[str, Any]) -> None:
        """Updates the model's parameters from a given dictionary."""
        super()._update_params(params)
        # Handle nested params
        base_classifier_params = {}
        regression_head_params = {}
        direct_regression_head_params = {}

        for key, value in params.items():
            if key.startswith("base_classifier_params__"):
                param_name = key.split("__", 1)[1]
                base_classifier_params[param_name] = value
            elif key.startswith("regression_head_params__"):
                param_name = key.split("__", 1)[1]
                regression_head_params[param_name] = value
            elif key.startswith("direct_regression_head_params__"):
                param_name = key.split("__", 1)[1]
                direct_regression_head_params[param_name] = value
            else:
                # Handle enums and other direct attributes
                if key == "regression_strategy" and isinstance(value, str):
                    setattr(self, key, RegressionStrategy[value.upper()])
                elif key == "n_classes_selection_method" and isinstance(value, str):
                    setattr(self, key, NClassesSelectionMethod[value.upper()])
                else:
                    setattr(self, key, value)

        if base_classifier_params:
            self.base_classifier_params.update(base_classifier_params)
        if regression_head_params:
            self.regression_head_params.update(regression_head_params)
        if direct_regression_head_params:
            self.direct_regression_head_params.update(direct_regression_head_params)

    def get_shap_explainer_info(self) -> dict[str, Any]:
        """Gets the SHAP explainer type and the model to be explained."""
        return {"explainer_type": ExplainerType.DEEP, "model": self.get_internal_model()}

    def _clone(self) -> "ProbabilisticRegressionModel":
        """Creates a new instance of the model with the same parameters."""
        return self.__class__(**self.get_params())

    def get_params(self) -> dict[str, Any]:
        """Gets parameters for this estimator."""
        params = super().get_params()
        params.update(
            {
                "n_classes": self.n_classes,
                "n_classes_inf": self.n_classes_inf,
                "max_n_classes_for_probabilistic_path": self.max_n_classes_for_probabilistic_path,
                "base_classifier_params": self.base_classifier_params,
                "regression_head_params": self.regression_head_params,
                "direct_regression_head_params": self.direct_regression_head_params,
                "regression_strategy": self.regression_strategy,
                "n_classes_selection_method": self.n_classes_selection_method,
                "gumbel_tau": self.gumbel_tau,
                "n_classes_predictor_learning_rate": self.n_classes_predictor_learning_rate,
                "add_classification_loss": self.add_classification_loss,
            }
        )
        return params

    def get_classifier_predictions(self, x: np.ndarray | pd.DataFrame, y_true_original: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns the internal classifier's predicted classes, probabilities, and.

        the corresponding (discretized) true labels for this composite model.

        Args:
            x (np.ndarray): Feature matrix.
            y_true_original (np.ndarray): Original true labels (will be discretized internally).

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                - Predicted classes from the internal classifier.
                - Predicted probabilities from the internal classifier.
                - Discretized true labels corresponding to the internal classifier's task.
        """
        if self.direct_regression:
            raise NotImplementedError("get_classifier_predictions is not available in direct regression mode.")
        if self.model is None or self.precomputed_class_boundaries is None:
            raise RuntimeError("Model has not been fitted yet or class boundaries were not computed.")

        self.model.eval()
        x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            # The model's forward pass now returns predictions, classifier_logits_out, and selected_k_values
            _, returned_classifier_logits, selected_k_values_tensor, _ = self.model(x_tensor)

            selected_k_values = selected_k_values_tensor.cpu().numpy()
            probabilistic_indices = np.where(selected_k_values < self.n_classes_inf)[0]

            if self.n_classes_selection_method != NClassesSelectionMethod.NONE:
                if len(probabilistic_indices) == 0:
                    raise NotImplementedError("No probabilistic predictions were made for the given X. All samples went to direct regression.")
                n_classes_for_classifier_output = self.max_n_classes_for_probabilistic_path
            else:
                n_classes_for_classifier_output = self.n_classes

            y_flat = y_true_original.flatten() if y_true_original.ndim > 1 else y_true_original
            y_true_discretized = np.full_like(y_flat, -1, dtype=int)  # Default to -1 (for direct regression)

            # Discretize y_true using the pre-computed boundaries from the training set by grouping samples by k
            probabilistic_k_values = selected_k_values[probabilistic_indices]
            unique_k_values = np.unique(probabilistic_k_values)

            for k_val in unique_k_values:
                k = int(k_val)
                # Create a mask for samples corresponding to the current k
                mask = probabilistic_k_values == k
                # Get the original indices for these samples
                original_indices = probabilistic_indices[mask]

                if len(original_indices) > 0:
                    assert k in self.precomputed_class_boundaries
                    boundaries = self.precomputed_class_boundaries[k]
                    # Discretize all samples for this k at once
                    _, discretized_labels = create_bins(data=y_flat[original_indices], unique_bin_edges=boundaries)
                    y_true_discretized[original_indices] = discretized_labels

            # Use the returned_classifier_logits directly
            classifier_logits_for_proba = returned_classifier_logits[probabilistic_indices]

            # Re-apply softmax to get probabilities for the probabilistic samples
            # Need to re-mask and softmax as the stored logits might be from the full max_n_classes_allowed
            k_values = torch.tensor(selected_k_values[probabilistic_indices], device=classifier_logits_for_proba.device).long()
            max_k = classifier_logits_for_proba.shape[1]
            col_indices = torch.arange(max_k, device=classifier_logits_for_proba.device)
            mask = col_indices < k_values.unsqueeze(1)
            masked_classifier_logits = torch.where(mask, classifier_logits_for_proba, float("-inf"))

            if n_classes_for_classifier_output == 2:
                proba_positive = torch.sigmoid(masked_classifier_logits[:, 0])  # Assuming binary classification for classifier
                y_proba_internal = torch.cat((1 - proba_positive.unsqueeze(1), proba_positive.unsqueeze(1)), dim=1).cpu().numpy()
            else:
                y_proba_internal = torch.softmax(masked_classifier_logits, dim=1).cpu().numpy()

            y_pred_internal = np.argmax(y_proba_internal, axis=1)

        return y_pred_internal, y_proba_internal, y_true_discretized

    def plot_probability_mappers(self, plot_path: str = "probability_mappers.png") -> None:
        """Plots the functions that map class probabilities to regression values."""
        if not self.model:
            logger.warning("No model found. Please fit the model first.")
            return
        if self.precomputed_class_boundaries is None:
            logger.warning("Class boundaries not computed. Please fit the model first.")
            return

        # Use the pre-computed class boundaries for the fixed n_classes case
        self.class_boundaries = self.precomputed_class_boundaries[self.n_classes]

        plot_nn_probability_mappers(
            mapper_model=self.model.regression_module,
            regression_strategy=self.regression_strategy,
            n_classes=self.n_classes,
            class_boundaries=self.class_boundaries,
            device=self.device,
            plot_path=plot_path,
            model_name=self.name,
        )


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
        device: torch.device,  # Add device parameter
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.device = device  # Assign device to self
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
            # n_classes_predictor: outputs logits for (max_n_classes_for_probabilistic_path - 2 + 1) modes e.g. with max_n_classes_for_probabilistic_path = 5, possible choices
            # for k (the number of bins) in the probabilistic path are: 2, 3, 4, and 5
            # (2 to max_n_classes_for_probabilistic_path) + 1 for direct regression
            n_classes_predictor_output_size = (self.max_n_classes_for_probabilistic_path - 2 + 1) + 1
            n_classes_predictor_instance = PyTorchNeuralNetwork(
                input_size=input_size,
                output_size=n_classes_predictor_output_size,
                task_type=TaskType.CLASSIFICATION,
                device=self.device,
                **base_classifier_params,
            )
            n_classes_predictor_instance.build_model()  # Explicitly build the model
            self.n_classes_predictor = n_classes_predictor_instance.get_internal_model()

            # Direct regression head
            direct_regression_head_instance = PyTorchNeuralNetwork(
                input_size=input_size,
                output_size=self.regression_output_size,
                task_type=TaskType.REGRESSION,
                device=self.device,
                **direct_regression_head_params,
            )
            direct_regression_head_instance.build_model()  # Explicitly build the model
            self.direct_regression_head = direct_regression_head_instance.get_internal_model()
        else:
            self.n_classes = n_classes
            self.n_classes_predictor = None
            self.direct_regression_head = None

        # Classifier part: Use PyTorchNeuralNetwork's internal architecture logic
        classifier_output_size = self.n_classes
        temp_classifier_instance = PyTorchNeuralNetwork(
            input_size=input_size,
            output_size=classifier_output_size,
            task_type=TaskType.CLASSIFICATION,
            device=self.device,
            **base_classifier_params,  # Pass device here as well
        )
        temp_classifier_instance.build_model()  # Explicitly build the model
        self.classifier_layers = temp_classifier_instance.get_internal_model()

        if self.regression_strategy == RegressionStrategy.SEPARATE_HEADS:
            self.regression_module = SeparateHeadsRegressionModule(
                n_classes=self.n_classes,
                regression_head_params=regression_head_params,
                uncertainty_method=self.uncertainty_method,
                regression_output_size=self.regression_output_size,
            )
        elif self.regression_strategy == RegressionStrategy.SINGLE_HEAD_N_OUTPUTS:
            self.regression_module = SingleHeadNOutputsRegressionModule(
                input_size=self.n_classes,
                n_classes=self.n_classes,
                regression_head_params=regression_head_params,
                uncertainty_method=self.uncertainty_method,
                regression_output_size=self.regression_output_size,
            )
        elif self.regression_strategy == RegressionStrategy.SINGLE_HEAD_FINAL_OUTPUT:
            self.regression_module = SingleHeadFinalOutputRegressionModule(
                input_size=self.n_classes,  # Changed from self.n_classes - 1
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

        if self.regression_strategy == RegressionStrategy.SINGLE_HEAD_FINAL_OUTPUT:
            # The regression_module's head directly outputs the mean for each class.
            # The final prediction is the weighted average of these means.
            class_means = self.regression_module(probabilities)  # Shape: (batch, n_classes, 1)
            return torch.sum(probabilities.unsqueeze(-1) * class_means, dim=1)
        return self.regression_module(probabilities)

    def forward(self, x_input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        n_classes_predictor_logits = self.n_classes_predictor(x_input) if self.n_classes_selection_method != NClassesSelectionMethod.NONE else None

        final_predictions, selected_k_values, log_prob_for_reinforce, classifier_logits_out = self.n_classes_strategy.forward(x_input, n_classes_predictor_logits)

        return final_predictions, classifier_logits_out, selected_k_values, log_prob_for_reinforce

    def get_num_parameters(self) -> int:
        """Returns the total number of trainable parameters in the model.

        Returns:
            int: The total number of parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
