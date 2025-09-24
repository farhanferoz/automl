"""Probabilistic Regression model implemented in PyTorch."""

from typing import Any, ClassVar

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from automl_package.enums import ExplainerType, NClassesSelectionMethod, ProbabilisticRegressionOptimizationStrategy, RegressionStrategy, UncertaintyMethod
from automl_package.logger import logger
from automl_package.models.architectures.probabilistic_regression_net import ProbabilisticRegressionNet
from automl_package.models.base_pytorch import PyTorchModelBase
from automl_package.models.common.losses import calculate_combined_loss
from automl_package.models.common.mixins import BoundaryLossMixin
from automl_package.utils.losses import boundary_regularization_loss, masked_cross_entropy_loss
from automl_package.utils.numerics import calculate_class_value_ranges, create_bins
from automl_package.utils.plotting import plot_nn_probability_mappers


class ProbabilisticRegressionModel(PyTorchModelBase, BoundaryLossMixin):
    """A PyTorch-based probabilistic regression model that directly learns both mean and variance."""

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
        "optimization_strategy": ProbabilisticRegressionOptimizationStrategy.REGRESSION_ONLY,
        "use_boundary_regularization": False,
        "boundary_loss_weight": 1.0,
        "use_monotonic_constraints": False,
        "constrain_middle_class": True,
    }

    def __init__(self, **kwargs: Any) -> None:
        """Initializes the ProbabilisticRegressionModel."""
        for key, value in self._defaults.items():
            kwargs.setdefault(key, value)

        if (
            kwargs.get("optimization_strategy") != ProbabilisticRegressionOptimizationStrategy.REGRESSION_ONLY
            and kwargs.get("uncertainty_method") != UncertaintyMethod.PROBABILISTIC
        ):
            logger.warning(
                f"Selected optimization_strategy requires a probabilistic uncertainty method. "
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
        if isinstance(self.optimization_strategy, str):
            self.optimization_strategy = ProbabilisticRegressionOptimizationStrategy[self.optimization_strategy.upper()]

        if self.use_monotonic_constraints and self.uncertainty_method == UncertaintyMethod.PROBABILISTIC:
            logger.warning(
                "PROBABILISTIC uncertainty is not recommended with monotonic constraints. "
                "Overriding uncertainty_method to BINNED_RESIDUAL_STD for more robust uncertainty estimates."
            )
            self.uncertainty_method = UncertaintyMethod.BINNED_RESIDUAL_STD

        if self.use_monotonic_constraints and self.regression_strategy != RegressionStrategy.SEPARATE_HEADS:
            raise ValueError("Monotonic constraints are only supported for the 'SEPARATE_HEADS' regression strategy.")

        self.base_classifier_params = self.base_classifier_params if self.base_classifier_params is not None else {}
        self.regression_head_params = self.regression_head_params if self.regression_head_params is not None else {}
        self.direct_regression_head_params = self.direct_regression_head_params if self.direct_regression_head_params is not None else {}

        self.direct_regression = self.n_classes_selection_method == NClassesSelectionMethod.NONE and self.n_classes >= self.n_classes_inf
        self.is_composite_regression_model = not self.direct_regression

        if self.direct_regression:
            logger.info(f"Number of classes ({self.n_classes}) >= n_classes_inf ({self.n_classes_inf}). Using direct regression mode.")
        elif self.n_classes_selection_method == NClassesSelectionMethod.NONE:
            logger.info(f"Using probabilistic regression mode with fixed {self.n_classes} classes.")
        else:
            logger.info(f"Using probabilistic regression mode with dynamic n_classes selection via {self.n_classes_selection_method.value}.")

        self.precomputed_class_boundaries = {}
        self.class_value_ranges_ = {}

    @property
    def name(self) -> str:
        """Returns the name of the model."""
        return f"ProbabilisticRegression_{self.regression_strategy.value}"

    def _calculate_custom_loss(self, model_outputs: tuple, y_true: torch.Tensor, include_boundary_loss: bool = True) -> torch.Tensor:
        """Calculates the loss for the ProbabilisticRegressionModel."""
        final_predictions, classifier_logits_out, selected_k_values, log_prob_for_reinforce, per_head_outputs = model_outputs
        y_true_squeezed = y_true.squeeze(-1) if y_true.ndim > 1 else y_true

        # 1. Calculate main regression loss
        regression_loss = calculate_combined_loss(
            predictions=final_predictions,
            y_true=y_true,
            uncertainty_method=self.uncertainty_method,
            use_boundary_regularization=False,  # Boundary loss is handled separately
            class_boundaries=None,
            class_value_ranges=None,
            boundary_loss_weight=self.boundary_loss_weight,
            device=self.device,
            include_boundary_loss=False,
        )

        total_loss = regression_loss
        unique_k = torch.tensor([])
        probabilistic_indices = torch.tensor([])

        # 2. Calculate classification loss if applicable
        if self.optimization_strategy != ProbabilisticRegressionOptimizationStrategy.REGRESSION_ONLY:
            probabilistic_indices = torch.where(selected_k_values < self.n_classes_inf)[0]

            if probabilistic_indices.numel() > 0:
                y_true_prob = y_true_squeezed[probabilistic_indices]
                logits_prob = classifier_logits_out[probabilistic_indices]
                k_values_prob = selected_k_values[probabilistic_indices]

                y_binned_prob = torch.zeros_like(y_true_prob, dtype=torch.long)
                unique_k = torch.unique(k_values_prob)

                for k in unique_k:
                    k_int = int(k.item())
                    mask = k_values_prob == k
                    boundaries = self.precomputed_class_boundaries[k_int]
                    _, y_binned_k = create_bins(data=y_true_prob[mask].cpu().numpy(), unique_bin_edges=boundaries)
                    y_binned_prob[mask] = torch.tensor(y_binned_k, dtype=torch.long, device=self.device)

                classification_loss = masked_cross_entropy_loss(logits_prob, y_binned_prob, k_values_prob)
                total_loss += classification_loss

        # 3. Apply Boundary Regularization Loss using the mixin
        if self.use_boundary_regularization and include_boundary_loss and per_head_outputs is not None:
            # Ensure we only consider samples that went through the probabilistic path
            if probabilistic_indices.numel() == 0:
                probabilistic_indices = torch.where(selected_k_values < self.n_classes_inf)[0]

            if probabilistic_indices.numel() > 0:
                y_true_prob = y_true_squeezed[probabilistic_indices]
                per_head_outputs_prob = per_head_outputs[probabilistic_indices]
                k_values_prob = selected_k_values[probabilistic_indices]

                y_binned_prob = torch.zeros_like(y_true_prob, dtype=torch.long)
                if not torch.is_tensor(unique_k) or unique_k.numel() == 0:
                    unique_k = torch.unique(k_values_prob)

                boundary_loss = 0
                num_k_groups = 0

                for k in unique_k:
                    k_int = int(k.item())
                    if k_int not in self.class_value_ranges_:
                        continue

                    k_mask = k_values_prob == k
                    if not torch.any(k_mask):
                        continue

                    # Bin the true values for this specific k
                    _, y_binned_k = create_bins(data=y_true_prob[k_mask].cpu().numpy(), unique_bin_edges=self.precomputed_class_boundaries[k_int])
                    y_binned_k_tensor = torch.tensor(y_binned_k, dtype=torch.long, device=self.device)

                    # Use the mixin to calculate loss for this group
                    boundary_loss += self.calculate_boundary_loss(
                        per_head_outputs=per_head_outputs_prob[k_mask],
                        y_true=y_true_prob[k_mask],
                        y_binned_tensor=y_binned_k_tensor,
                        class_value_ranges=self.class_value_ranges_[k_int].to(self.device),
                        boundary_loss_weight=1.0,  # Weight is applied at the end
                    )
                    num_k_groups += 1

                if num_k_groups > 0:
                    total_loss += self.boundary_loss_weight * (boundary_loss / num_k_groups)

        return total_loss

    def _calculate_performance_score(self, y_true: np.ndarray, y_pred: np.ndarray, x_val: np.ndarray | None = None, y_pred_std: np.ndarray | None = None) -> float:
        """Calculates the performance score, including the classification penalty if applicable."""
        if self.optimization_strategy == ProbabilisticRegressionOptimizationStrategy.REGRESSION_ONLY:
            score = super()._calculate_performance_score(y_true, y_pred, x_val, y_pred_std)
        else:
            self.model.eval()
            with torch.no_grad():
                x_val_tensor = torch.tensor(x_val, dtype=torch.float32).to(self.device)
                y_val_tensor = torch.tensor(y_true, dtype=torch.float32).to(self.device).unsqueeze(1)

                model_outputs = self.model(x_val_tensor)
                total_loss = self._calculate_custom_loss(model_outputs, y_val_tensor, include_boundary_loss=False)

            score = total_loss.item()
        return score

    def build_model(self) -> None:
        """Builds the internal PyTorch nn.Module for the ProbabilisticRegressionModel."""
        if self.n_classes_selection_method != NClassesSelectionMethod.NONE and self.max_n_classes_for_probabilistic_path >= self.n_classes_inf:
            raise ValueError("max_n_classes_for_probabilistic_path must be less than n_classes_inf when n_classes_selection_method is not NONE.")

        self.model = ProbabilisticRegressionNet(
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
            optimization_strategy=self.optimization_strategy,
            gumbel_tau=self.gumbel_tau,
            n_classes_predictor_learning_rate=self.n_classes_predictor_learning_rate,
            device=self.device,
            use_monotonic_constraints=self.use_monotonic_constraints,
            constrain_middle_class=self.constrain_middle_class,
        )
        self.model.to(self.device)

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
                "optimization_strategy": {"type": "categorical", "choices": [s.value for s in ProbabilisticRegressionOptimizationStrategy]},
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
                "use_boundary_regularization": {"type": "categorical", "choices": [True, False]},
                "boundary_loss_weight": {"type": "float", "low": 0.1, "high": 10.0, "log": True},
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
        self, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray | None = None, y_val: np.ndarray | None = None, forced_iterations: int | None = None
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
            self.class_value_ranges_ = {}
            y_flat = y_train.flatten() if y_train.ndim > 1 else y_train
            y_min, y_max = np.min(y_flat), np.max(y_flat)

            max_k = self.max_n_classes_for_probabilistic_path if self.n_classes_selection_method != NClassesSelectionMethod.NONE else self.n_classes

            # Pre-calculate and store boundaries for all possible k values
            for k in [max_k] if self.n_classes_selection_method == NClassesSelectionMethod.NONE else range(2, max_k + 1):
                boundaries, y_binned = create_bins(data=y_flat, n_bins=k, min_value=-np.inf, max_value=np.inf)
                self.precomputed_class_boundaries[k] = boundaries

                if self.use_boundary_regularization:
                    self.class_value_ranges_[k] = calculate_class_value_ranges(y_flat=y_flat, y_binned=y_binned, k=k, y_min=y_min, y_max=y_max, device=self.device)

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
                if key == "regression_strategy" and isinstance(value, str):
                    setattr(self, key, RegressionStrategy[value.upper()])
                elif key == "n_classes_selection_method" and isinstance(value, str):
                    setattr(self, key, NClassesSelectionMethod[value.upper()])
                elif key == "optimization_strategy" and isinstance(value, str):
                    setattr(self, key, ProbabilisticRegressionOptimizationStrategy[value.upper()])
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
                "optimization_strategy": self.optimization_strategy,
                "use_boundary_regularization": self.use_boundary_regularization,
                "boundary_loss_weight": self.boundary_loss_weight,
                "use_monotonic_constraints": self.use_monotonic_constraints,
                "constrain_middle_class": self.constrain_middle_class,
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
        elif self.precomputed_class_boundaries is None:
            logger.warning("Class boundaries not computed. Please fit the model first.")
        else:
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

    def _compute_predictions_for_k(self, classifier_raw_logits: torch.Tensor, k_val: int) -> torch.Tensor:
        """Helper to compute regression predictions for a given k_val."""
        masked_classifier_logits = torch.full_like(classifier_raw_logits, float("-inf"))
        masked_classifier_logits[:, :k_val] = classifier_raw_logits[:, :k_val]

        probabilities = torch.softmax(masked_classifier_logits, dim=1)

        if self.optimization_strategy == ProbabilisticRegressionOptimizationStrategy.GRADIENT_STOP:
            probabilities = probabilities.detach()

        if self.regression_strategy == RegressionStrategy.SINGLE_HEAD_FINAL_OUTPUT:
            # The regression_module's head directly outputs the mean for each class.
            # The final prediction is the weighted average of these means.
            class_means = self.regression_module(probabilities)  # Shape: (batch, n_classes, 1)
            preds = torch.sum(probabilities.unsqueeze(-1) * class_means, dim=1)
        else:
            preds = self.regression_module(probabilities)
        return preds

    def forward(self, x_input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Performs the forward pass through the model."""
        n_classes_predictor_logits = self.n_classes_predictor(x_input) if self.n_classes_selection_method != NClassesSelectionMethod.NONE else None

        final_predictions, selected_k_values, log_prob_for_reinforce, classifier_logits_out = self.n_classes_strategy.forward(x_input, n_classes_predictor_logits)

        return final_predictions, classifier_logits_out, selected_k_values, log_prob_for_reinforce

    def get_num_parameters(self) -> int:
        """Returns the total number of trainable parameters in the model.

        Returns:
            int: The total number of parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
