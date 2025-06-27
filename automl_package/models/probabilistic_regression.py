import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any, Type, List, Tuple

from .base import BaseModel
from .neural_network import PyTorchNeuralNetwork  # Only PyTorch NN is supported as base classifier for now
from ..enums import UncertaintyMethod, RegressionStrategy, TaskType  # Import enums
from ..logger import logger  # Import logger


class ProbabilisticRegressionModel(BaseModel):
    """
    A regression model that first performs classification, then feeds
    probabilities into separate regression heads, and combines their outputs
    into a final regression prediction. The entire model is trained end-to-end.
    """

    def __init__(
        self,
        input_size: int = None,
        n_classes: int = 5,
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

        if self.regression_strategy not in [RegressionStrategy.SEPARATE_HEADS, RegressionStrategy.SINGLE_HEAD_N_OUTPUTS, RegressionStrategy.SINGLE_HEAD_FINAL_OUTPUT]:
            raise ValueError(f"Unsupported regression_strategy: {self.regression_strategy.value}. Choose from enum values.")
        if self.uncertainty_method not in [UncertaintyMethod.CONSTANT, UncertaintyMethod.MC_DROPOUT, UncertaintyMethod.PROBABILISTIC]:
            raise ValueError(f"Unsupported uncertainty_method: {self.uncertainty_method.value}. Choose from enum values.")

        self.classifier_model: PyTorchNeuralNetwork = None
        self.regression_heads: nn.ModuleList = None  # List of PyTorch regression heads (for separate_heads)
        self.regression_head: nn.Module = None  # Single PyTorch regression head (for single_head_n_outputs, single_head_final_output)
        self.combined_model: nn.Module = None

        self._is_regression_model = True  # This model is always regression
        self._train_residual_std = 0.0  # For 'constant' uncertainty method

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def name(self) -> str:
        return "ProbabilisticRegression"

    class _CombinedProbabilisticModel(nn.Module):
        """
        Internal PyTorch Module combining the classifier and regression heads.
        """

        def __init__(
            self, input_size: int, n_classes: int, base_classifier_params: Dict[str, Any], regression_head_params: Dict[str, Any], regression_strategy: RegressionStrategy
        ):  # Use enum
            super().__init__()
            self.n_classes = n_classes
            self.regression_strategy = regression_strategy

            # Classifier part - must be PyTorchNeuralNetwork
            temp_classifier = PyTorchNeuralNetwork(
                input_size=input_size, output_size=n_classes, task_type=TaskType.CLASSIFICATION, **base_classifier_params  # Always classification
            )
            temp_classifier._build_model()
            self.classifier_layers = temp_classifier.model

            # Common regression head parameters
            head_hidden_layers = regression_head_params.get("hidden_layers", 1)
            head_hidden_size = regression_head_params.get("hidden_size", 32)
            use_batch_norm_heads = regression_head_params.get("use_batch_norm", False)

            if self.regression_strategy == RegressionStrategy.SEPARATE_HEADS:
                self.regression_heads = nn.ModuleList()
                for _ in range(n_classes):
                    head_layers = []
                    head_input_size = 1  # Only the probability is fed into the regression head

                    head_layers.append(nn.Linear(head_input_size, head_hidden_size))
                    if use_batch_norm_heads:
                        head_layers.append(nn.BatchNorm1d(head_hidden_size))
                    head_layers.append(nn.ReLU())

                    for _ in range(head_hidden_layers - 1):
                        head_layers.append(nn.Linear(head_hidden_size, head_hidden_size))
                        if use_batch_norm_heads:
                            head_layers.append(nn.BatchNorm1d(head_hidden_size))
                        head_layers.append(nn.ReLU())

                    head_layers.append(nn.Linear(head_hidden_size, 1))  # Output is a single regression value
                    self.regression_heads.append(nn.Sequential(*head_layers))

            elif self.regression_strategy == RegressionStrategy.SINGLE_HEAD_N_OUTPUTS:
                head_layers = []
                head_input_size = n_classes - 1
                head_output_size = n_classes

                head_layers.append(nn.Linear(head_input_size, head_hidden_size))
                if use_batch_norm_heads:
                    head_layers.append(nn.BatchNorm1d(head_hidden_size))
                head_layers.append(nn.ReLU())

                for _ in range(head_hidden_layers - 1):
                    head_layers.append(nn.Linear(head_hidden_size, head_hidden_size))
                    if use_batch_norm_heads:
                        head_layers.append(nn.BatchNorm1d(head_hidden_size))
                    head_layers.append(nn.ReLU())

                head_layers.append(nn.Linear(head_hidden_size, head_output_size))
                self.regression_head = nn.Sequential(*head_layers)

            elif self.regression_strategy == RegressionStrategy.SINGLE_HEAD_FINAL_OUTPUT:
                head_layers = []
                head_input_size = n_classes - 1
                head_output_size = 1

                head_layers.append(nn.Linear(head_input_size, head_hidden_size))
                if use_batch_norm_heads:
                    head_layers.append(nn.BatchNorm1d(head_hidden_size))
                head_layers.append(nn.ReLU())

                for _ in range(head_hidden_layers - 1):
                    head_layers.append(nn.Linear(head_hidden_size, head_hidden_size))
                    if use_batch_norm_heads:
                        head_layers.append(nn.BatchNorm1d(head_hidden_size))
                    head_layers.append(nn.ReLU())

                head_layers.append(nn.Linear(head_hidden_size, head_output_size))
                self.regression_head = nn.Sequential(*head_layers)

        def forward(self, x_input: torch.Tensor) -> torch.Tensor:
            classifier_logits = self.classifier_layers(x_input)
            if self.n_classes == 2:
                probabilities = torch.cat((1 - torch.sigmoid(classifier_logits), torch.sigmoid(classifier_logits)), dim=1)
            else:
                probabilities = torch.softmax(classifier_logits, dim=1)

            if self.regression_strategy == RegressionStrategy.SEPARATE_HEADS:
                final_predictions = torch.zeros(x_input.size(0), 1).to(x_input.device)
                for i in range(self.n_classes):
                    p_i = probabilities[:, i].unsqueeze(1)
                    y_i_expected = self.regression_heads[i](p_i)
                    final_predictions += p_i * y_i_expected
                return final_predictions.squeeze(1)

            elif self.regression_strategy == RegressionStrategy.SINGLE_HEAD_N_OUTPUTS:
                if self.n_classes > 1:
                    head_input_probas = probabilities[:, :-1]
                else:
                    head_input_probas = torch.empty(probabilities.size(0), 0).to(x_input.device)  # Handle n_classes=1
                y_expected_all_classes = self.regression_head(head_input_probas)

                final_predictions = (probabilities * y_expected_all_classes).sum(dim=1)
                return final_predictions

            elif self.regression_strategy == RegressionStrategy.SINGLE_HEAD_FINAL_OUTPUT:
                if self.n_classes > 1:
                    head_input_probas = probabilities[:, :-1]
                else:
                    head_input_probas = torch.empty(probabilities.size(0), 0).to(x_input.device)  # Handle n_classes=1

                final_predictions = self.regression_head(head_input_probas)
                return final_predictions.squeeze(1)

    def fit(self, X: np.ndarray, y: np.ndarray):
        if self.base_classifier_class != PyTorchNeuralNetwork:
            raise ValueError(
                f"ProbabilisticRegressionModel currently only supports PyTorchNeuralNetwork as base_classifier_class for end-to-end training. Got {self.base_classifier_class.__name__}."
            )

        if self.input_size is None:
            self.input_size = X.shape[1]

        self.combined_model = self._CombinedProbabilisticModel(
            input_size=self.input_size,
            n_classes=self.n_classes,
            base_classifier_params=self.base_classifier_params,
            regression_head_params=self.regression_head_params,
            regression_strategy=self.regression_strategy,
        ).to(self.device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.combined_model.parameters(), lr=self.learning_rate)

        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(self.device)

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.combined_model.train()
        for epoch in range(self.n_epochs):
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.combined_model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        # Calculate residual standard deviation for uncertainty estimation
        y_pred_train = self.predict(X)
        self._train_residual_std = np.std(y - y_pred_train)
        if np.isnan(self._train_residual_std):
            self._train_residual_std = 0.0

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.combined_model is None:
            raise RuntimeError("Model has not been fitted yet.")
        self.combined_model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            predictions = self.combined_model(X_tensor)
        return predictions.cpu().numpy()

    def predict_uncertainty(self, X: np.ndarray) -> np.ndarray:
        if not self._is_regression_model:
            raise ValueError("predict_uncertainty is only available for regression models.")
        if self.combined_model is None:
            raise RuntimeError("Model has not been fitted yet.")
        # For simplicity, return a constant uncertainty based on training residuals
        return np.full(X.shape[0], self._train_residual_std)

    def get_hyperparameter_search_space(self) -> Dict[str, Any]:
        return {
            "n_classes": {"type": "int", "low": 2, "high": 5},
            "learning_rate": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True},
            "n_epochs": {"type": "int", "low": 10, "high": 50, "step": 10},
            "batch_size": {"type": "categorical", "choices": [16, 32, 64]},
            "regression_strategy": {"type": "categorical", "choices": [e.value for e in RegressionStrategy]},  # Use enum values
            # Nested parameters for the internal classifier (PyTorchNeuralNetwork)
            "base_classifier_params__hidden_layers": {"type": "int", "low": 1, "high": 2},
            "base_classifier_params__hidden_size": {"type": "int", "low": 32, "high": 64, "step": 32},
            "base_classifier_params__use_batch_norm": {"type": "categorical", "choices": [True, False]},
            # Nested parameters for the internal regression heads (will be sampled conditionally)
            "regression_head_params__hidden_layers": {"type": "int", "low": 0, "high": 1},
            "regression_head_params__hidden_size": {"type": "int", "low": 16, "high": 32, "step": 16},
            "regression_head_params__use_batch_norm": {"type": "categorical", "choices": [True, False]},
        }

    def get_internal_model(self):
        """
        Returns the raw underlying PyTorch combined model (nn.Module).
        """
        return self.combined_model
