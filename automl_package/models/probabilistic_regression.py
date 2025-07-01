from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .base import BaseModel
from .neural_network import PyTorchNeuralNetwork  # Only PyTorch NN is supported as base classifier for now
from ..enums import UncertaintyMethod, RegressionStrategy, TaskType  # Import enums


class ProbabilisticRegressionModel(BaseModel):
    """
    A PyTorch-based probabilistic regression model that directly learns both mean and variance.
    Can use different strategies for outputting mean and variance.
    """

    def __init__(
        self,
        input_size: int = None,
        n_classes: int = 5,  # Number of classes for the internal classifier
        base_classifier_params: Dict[str, Any] = None,  # Params for the internal PyTorch NN classifier
        regression_head_params: Dict[str, Any] = None,  # Params for each internal PyTorch NN regression head
        regression_strategy: RegressionStrategy = RegressionStrategy.SEPARATE_HEADS,  # Use enum
        learning_rate: float = 0.001,
        n_epochs: int = 10,
        batch_size: int = 32,
        uncertainty_method: UncertaintyMethod = UncertaintyMethod.CONSTANT,  # Use enum
        n_mc_dropout_samples: int = 100,
        dropout_rate: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_size = input_size
        self.n_classes = n_classes
        self.base_classifier_params = base_classifier_params if base_classifier_params is not None else {}
        self.regression_head_params = regression_head_params if regression_head_params is not None else {}
        self.regression_strategy = regression_strategy
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.uncertainty_method = uncertainty_method
        self.n_mc_dropout_samples = n_mc_dropout_samples
        self.dropout_rate = dropout_rate

        # Validate regression strategy and uncertainty method
        if self.regression_strategy not in [RegressionStrategy.SEPARATE_HEADS, RegressionStrategy.SINGLE_HEAD_N_OUTPUTS, RegressionStrategy.SINGLE_HEAD_FINAL_OUTPUT]:
            raise ValueError(f"Unsupported regression_strategy: {self.regression_strategy.value}. Choose from enum values.")
        if self.uncertainty_method not in [UncertaintyMethod.CONSTANT, UncertaintyMethod.MC_DROPOUT, UncertaintyMethod.PROBABILISTIC]:
            raise ValueError(f"Unsupported uncertainty_method: {self.uncertainty_method.value}. Choose from enum values.")

        self.classifier_model: Optional[PyTorchNeuralNetwork] = None
        self.regression_heads: Optional[nn.ModuleList] = None  # List of PyTorch regression heads (for separate_heads)
        self.regression_head: Optional[nn.Module] = None  # Single PyTorch regression head (for single_head_n_outputs, single_head_final_output)
        self.combined_model: Optional[nn.Module] = None  # The main PyTorch nn.Module that combines everything

        self._is_regression_model = True  # This model is always regression
        self.is_composite_regression_model = True # Flag for AutoML to identify composite models
        self._train_residual_std = 0.0  # For 'constant' uncertainty method

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def name(self) -> str:
        return "ProbabilisticRegression"  # ModelName.PROBABILISTIC_REGRESSION.value (using string for consistency)

    class _CombinedProbabilisticModel(nn.Module):
        """
        Internal PyTorch Module combining the classifier and regression heads.
        This is the actual neural network structure for ProbabilisticRegressionModel.
        """

        def __init__(
            self,
            input_size: int,
            n_classes: int,
            base_classifier_params: Dict[str, Any],
            regression_head_params: Dict[str, Any],
            regression_strategy: RegressionStrategy,
            uncertainty_method: UncertaintyMethod,
        ):
            super().__init__()
            self.n_classes = n_classes
            self.regression_strategy = regression_strategy
            self.uncertainty_method = uncertainty_method

            # Classifier part: Use PyTorchNeuralNetwork's internal architecture logic
            # This will be a sequential module. Output size is `n_classes`.
            # Make sure to handle the case where n_classes=2, as PyTorch NN defaults to 1 output for binary.
            classifier_output_size = 1 if n_classes == 2 else n_classes
            temp_classifier_instance = PyTorchNeuralNetwork(
                input_size=input_size, output_size=classifier_output_size, task_type=TaskType.CLASSIFICATION, **base_classifier_params  # Always classification internally
            )
            # Call _build_model to initialize its internal sequential model
            temp_classifier_instance.build_model()
            self.classifier_layers = temp_classifier_instance.model

            # Common regression head parameters
            head_hidden_layers = regression_head_params.get("hidden_layers", 1)
            head_hidden_size = regression_head_params.get("hidden_size", 32)
            use_batch_norm_heads = regression_head_params.get("use_batch_norm", False)
            head_dropout_rate = regression_head_params.get("dropout_rate", 0.0)

            # Determine regression head output size (1 for mean, 2 for mean+log_var)
            self.regression_output_size = 1
            if self.uncertainty_method == UncertaintyMethod.PROBABILISTIC:
                self.regression_output_size = 2

            if self.regression_strategy == RegressionStrategy.SEPARATE_HEADS:
                self.regression_heads = nn.ModuleList()
                for _ in range(n_classes):
                    head_layers = []
                    # Each head takes a single probability as input
                    head_input_size = 1

                    head_layers.append(nn.Linear(head_input_size, head_hidden_size))
                    if use_batch_norm_heads:
                        head_layers.append(nn.BatchNorm1d(head_hidden_size))
                    head_layers.append(nn.ReLU())

                    for _ in range(head_hidden_layers - 1):  # -1 for the first layer
                        head_layers.append(nn.Linear(head_hidden_size, head_hidden_size))
                        if use_batch_norm_heads:
                            head_layers.append(nn.BatchNorm1d(head_hidden_size))
                        head_layers.append(nn.ReLU())
                        if head_dropout_rate > 0 and self.uncertainty_method == UncertaintyMethod.MC_DROPOUT:
                            head_layers.append(nn.Dropout(head_dropout_rate))  # Dropout for MC Dropout heads

                    head_layers.append(nn.Linear(head_hidden_size, self.regression_output_size))  # Output mean or mean+log_var
                    self.regression_heads.append(nn.Sequential(*head_layers))

            elif self.regression_strategy == RegressionStrategy.SINGLE_HEAD_N_OUTPUTS:
                head_layers = []
                # Input to this head is all but the last probability (if n_classes > 1)
                head_input_size = n_classes if n_classes == 1 else n_classes - 1
                # Output N*outputs (e.g., N*1 for mean only, N*2 for mean+log_var)
                head_output_neurons = n_classes * self.regression_output_size

                head_layers.append(nn.Linear(head_input_size, head_hidden_size))
                if use_batch_norm_heads:
                    head_layers.append(nn.BatchNorm1d(head_hidden_size))
                head_layers.append(nn.ReLU())

                for _ in range(head_hidden_layers - 1):
                    head_layers.append(nn.Linear(head_hidden_size, head_hidden_size))
                    if use_batch_norm_heads:
                        head_layers.append(nn.BatchNorm1d(head_hidden_size))
                    head_layers.append(nn.ReLU())
                    if head_dropout_rate > 0 and self.uncertainty_method == UncertaintyMethod.MC_DROPOUT:
                        head_layers.append(nn.Dropout(head_dropout_rate))

                head_layers.append(nn.Linear(head_hidden_size, head_output_neurons))
                self.regression_head = nn.Sequential(*head_layers)

            elif self.regression_strategy == RegressionStrategy.SINGLE_HEAD_FINAL_OUTPUT:
                head_layers = []
                # Input is all but the last probability (if n_classes > 1)
                head_input_size = n_classes if n_classes == 1 else n_classes - 1
                head_output_neurons = self.regression_output_size  # Just 1 or 2 outputs

                head_layers.append(nn.Linear(head_input_size, head_hidden_size))
                if use_batch_norm_heads:
                    head_layers.append(nn.BatchNorm1d(head_hidden_size))
                head_layers.append(nn.ReLU())

                for _ in range(head_hidden_layers - 1):
                    head_layers.append(nn.Linear(head_hidden_size, head_hidden_size))
                    if use_batch_norm_heads:
                        head_layers.append(nn.BatchNorm1d(head_hidden_size))
                    head_layers.append(nn.ReLU())
                    if head_dropout_rate > 0 and self.uncertainty_method == UncertaintyMethod.MC_DROPOUT:
                        head_layers.append(nn.Dropout(head_dropout_rate))

                head_layers.append(nn.Linear(head_hidden_size, head_output_neurons))
                self.regression_head = nn.Sequential(*head_layers)
            else:
                raise ValueError(f"Unknown regression_strategy: {regression_strategy}")

        def forward(self, x_input: torch.Tensor) -> torch.Tensor:
            # 1. Pass through classifier to get logits and then probabilities
            classifier_logits = self.classifier_layers(x_input)
            if self.n_classes == 2:
                # For binary, outputs will be 1 logit, expand to 2 probabilities
                proba_positive = torch.sigmoid(classifier_logits)
                probabilities = torch.cat((1 - proba_positive, proba_positive), dim=1)
            else:
                probabilities = torch.softmax(classifier_logits, dim=1)

            # 2. Pass probabilities through regression heads based on strategy
            if self.regression_strategy == RegressionStrategy.SEPARATE_HEADS:
                # Sum (P_i * Y_i_expected) across all classes
                final_predictions = torch.zeros(x_input.size(0), self.regression_output_size).to(x_input.device)
                for i in range(self.n_classes):
                    p_i = probabilities[:, i].unsqueeze(1)  # Probability for class i, shape (batch_size, 1)
                    y_i_expected_or_dist_params = self.regression_heads[i](p_i)
                    final_predictions += p_i * y_i_expected_or_dist_params

                return final_predictions.squeeze(-1)  # Squeeze if output_size is 1 (mean only)

            elif self.regression_strategy == RegressionStrategy.SINGLE_HEAD_N_OUTPUTS:
                # Input to head is all but the last probability (or all if n_classes=1, though unlikely for this model)
                head_input_probas = probabilities if self.n_classes == 1 else probabilities[:, :-1]

                y_expected_all_classes = self.regression_head(head_input_probas)
                # Reshape to (batch_size, n_classes, regression_output_size)
                y_expected_all_classes = y_expected_all_classes.reshape(x_input.shape[0], self.n_classes, self.regression_output_size)

                # Expand probabilities for broadcasting: (batch_size, n_classes, 1)
                probabilities_expanded = probabilities.unsqueeze(-1)

                # Final output is sum of (P_i * Y_i_expected)
                final_predictions_summed = (probabilities_expanded * y_expected_all_classes).sum(dim=1)

                return final_predictions_summed.squeeze(-1)

            elif self.regression_strategy == RegressionStrategy.SINGLE_HEAD_FINAL_OUTPUT:
                # Input to head is all but the last probability (if n_classes > 1)
                head_input_probas = probabilities if self.n_classes == 1 else probabilities[:, :-1]

                final_predictions = self.regression_head(head_input_probas)
                return final_predictions.squeeze(-1)  # Squeeze if output_size is 1 (mean only)
            else:
                raise ValueError("Invalid regression_strategy")

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Trains the probabilistic regression model.
        Args:
            X (np.ndarray): Training features.
            y (np.ndarray): Training targets.
        """
        # Ensure input_size is set if not provided at init
        if self.input_size is None:
            self.input_size = X.shape[1]

        # Build the combined PyTorch model
        self.combined_model = self._CombinedProbabilisticModel(
            input_size=self.input_size,
            n_classes=self.n_classes,
            base_classifier_params=self.base_classifier_params,
            regression_head_params=self.regression_head_params,
            regression_strategy=self.regression_strategy,
            uncertainty_method=self.uncertainty_method,  # Pass uncertainty method to internal model
        ).to(self.device)

        # Define criterion (MSE for mean, NLL for mean+log_var)
        if self.uncertainty_method == UncertaintyMethod.PROBABILISTIC:

            def nll_loss(nll_outputs, targets):
                # outputs from forward will be (mean, log_var)
                mean = nll_outputs[:, 0]
                log_var = nll_outputs[:, 1]
                loss_value = 0.5 * torch.mean(torch.exp(-log_var) * (targets - mean) ** 2 + log_var)
                return loss_value

            criterion = nll_loss
        else:
            criterion = nn.MSELoss()

        optimizer = optim.Adam(self.combined_model.parameters(), lr=self.learning_rate)

        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1).to(self.device)  # Ensure target is 2D for loss

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.combined_model.train()  # Set model to training mode
        for epoch in range(self.n_epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.combined_model(batch_X)  # This now directly returns prediction(s)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            # logger.info(f"Probabilistic Model Epoch {epoch+1}, Loss: {total_loss / len(dataloader):.4f}")

        # Calculate residual standard deviation for 'constant' uncertainty (if applicable)
        if self.uncertainty_method == UncertaintyMethod.CONSTANT:
            y_pred_train = self.predict(X)
            self._train_residual_std = np.std(y - y_pred_train)
            if np.isnan(self._train_residual_std):
                self._train_residual_std = 0.0

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Makes predictions (mean only) from the probabilistic model.
        Args:
            X (np.ndarray): Features for prediction.
        Returns:
            np.ndarray: Predicted mean values.
        """
        if self.combined_model is None:
            raise RuntimeError("Model has not been fitted yet.")
        self.combined_model.eval()  # Set model to evaluation mode
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = self.combined_model(X_tensor)
            if self.uncertainty_method == UncertaintyMethod.PROBABILISTIC:
                # If probabilistic, outputs will be (mean, log_var), return only mean
                predictions = outputs[:, 0]
            else:
                predictions = outputs
        return predictions.cpu().numpy().flatten()

    def predict_uncertainty(self, X: np.ndarray) -> np.ndarray:
        """
        Estimates uncertainty (standard deviation) from the probabilistic model's learned variance
        or constant residual std. MC Dropout is handled separately if implemented.
        Args:
            X (np.ndarray): Features for uncertainty estimation.
        Returns:
            np.ndarray: Predicted standard deviation values.
        """
        if not self._is_regression_model:
            raise ValueError("predict_uncertainty is only available for regression models.")
        if self.combined_model is None:
            raise RuntimeError("Model has not been fitted yet.")

        self.combined_model.eval()  # Set model to evaluation mode
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)

        if self.uncertainty_method == UncertaintyMethod.CONSTANT:
            return np.full(X.shape[0], self._train_residual_std)
        elif self.uncertainty_method == UncertaintyMethod.PROBABILISTIC:
            with torch.no_grad():
                outputs = self.combined_model(X_tensor)  # Outputs (mean, log_var)
                log_var = outputs[:, 1]
            return torch.exp(0.5 * log_var).cpu().numpy().flatten()  # Standard deviation
        elif self.uncertainty_method == UncertaintyMethod.MC_DROPOUT:
            self.combined_model.train()  # Enable dropout for MC sampling
            mc_predictions = []
            with torch.no_grad():
                for _ in range(self.n_mc_dropout_samples):
                    outputs = self.combined_model(X_tensor)
                    mc_predictions.append(outputs.cpu().numpy().flatten())
            self.combined_model.eval()  # Set back to eval mode
            return np.std(mc_predictions, axis=0)  # Return std dev of MC samples
        else:
            raise ValueError(f"Unknown uncertainty_method: {self.uncertainty_method.value}")

    def get_hyperparameter_search_space(self) -> Dict[str, Any]:
        """
        Defines the hyperparameter search space for ProbabilisticRegressionModel.
        Returns:
            Dict[str, Any]: Search space configuration for Optuna.
        """
        space = {
            "n_classes": {"type": "int", "low": 2, "high": 5},
            "learning_rate": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True},
            "n_epochs": {"type": "int", "low": 10, "high": 50, "step": 10},
            "batch_size": {"type": "categorical", "choices": [16, 32, 64]},
            "regression_strategy": {"type": "categorical", "choices": [s.value for s in RegressionStrategy]},
            "uncertainty_method": {"type": "categorical", "choices": [e.value for e in UncertaintyMethod]},
            # Nested parameters for the internal classifier (PyTorchNeuralNetwork-like structure)
            "base_classifier_params__hidden_layers": {"type": "int", "low": 1, "high": 2},
            "base_classifier_params__hidden_size": {"type": "int", "low": 32, "high": 64, "step": 32},
            "base_classifier_params__use_batch_norm": {"type": "categorical", "choices": [True, False]},
            # Add dropout for base classifier if MC_DROPOUT is selected
            "base_classifier_params__dropout_rate": {"type": "float", "low": 0.0, "high": 0.5, "step": 0.1},
            # Nested parameters for the internal regression heads (PyTorchNN-like structure)
            "regression_head_params__hidden_layers": {"type": "int", "low": 0, "high": 1},
            "regression_head_params__hidden_size": {"type": "int", "low": 16, "high": 32, "step": 16},
            "regression_head_params__use_batch_norm": {"type": "categorical", "choices": [True, False]},
            # Add dropout for regression heads if MC_DROPOUT is selected
            "regression_head_params__dropout_rate": {"type": "float", "low": 0.0, "high": 0.5, "step": 0.1},
        }

        # Conditional hyperparameters for MC_DROPOUT (sampled only if uncertainty_method is MC_DROPOUT)
        space["n_mc_dropout_samples"] = {"type": "int", "low": 50, "high": 200, "step": 50}
        space["dropout_rate"] = {"type": "float", "low": 0.1, "high": 0.5, "step": 0.1}

        return space

    def get_classifier_predictions(self, X: np.ndarray, y_true_original: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns the internal classifier's predicted classes, probabilities, and
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
        if self.combined_model is None:
            raise RuntimeError("Model has not been fitted yet.")

        # 1. Discretize the original true labels using the same boundaries as during training
        y_flat = y_true_original.flatten() if y_true_original.ndim > 1 else y_true_original
        y_true_discretized = np.zeros_like(y_flat, dtype=int)
        # Reconstruct class boundaries based on percentiles of the original target
        # This is crucial to ensure consistency with how y_clf was created during training
        percentiles = np.linspace(0, 100, self.n_classes + 1)[1:-1]
        class_boundaries = np.percentile(y_flat, percentiles)

        for i, boundary in enumerate(class_boundaries):
            y_true_discretized[y_flat > boundary] = i + 1

        # 2. Get predictions and probabilities from the internal classifier
        # Need to ensure the internal classifier part is in eval mode
        self.combined_model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            classifier_logits = self.combined_model.classifier_layers(X_tensor)
            if self.n_classes == 2:
                proba_positive = torch.sigmoid(classifier_logits)
                y_proba_internal = torch.cat((1 - proba_positive, proba_positive), dim=1).cpu().numpy()
            else:
                y_proba_internal = torch.softmax(classifier_logits, dim=1).cpu().numpy()

            y_pred_internal = np.argmax(y_proba_internal, axis=1)

        return y_pred_internal, y_proba_internal, y_true_discretized

    def get_internal_model(self):
        """
        Returns the raw underlying PyTorch combined model (nn.Module).
        """
        return self.combined_model

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError("ProbabilisticRegressionModel is a regression model and does not support predict_proba.")
