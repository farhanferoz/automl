import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from .base import BaseModel
from typing import Dict, Any, Tuple
from ..enums import UncertaintyMethod, TaskType  # Import enums
from ..logger import logger  # Import logger


class PyTorchNeuralNetwork(BaseModel):
    """
    A simple Feed-Forward Neural Network implemented using PyTorch.
    Can be configured with variable hidden layers and neurons.
    Supports optional Batch Normalization.
    Supports constant, MC-Dropout, and probabilistic layer uncertainty estimation for regression.
    """

    def __init__(
        self,
        input_size: int = None,
        hidden_layers: int = 1,
        hidden_size: int = 64,
        output_size: int = 1,
        learning_rate: float = 0.001,
        n_epochs: int = 10,
        batch_size: int = 32,
        task_type: TaskType = TaskType.REGRESSION,  # Use enum
        use_batch_norm: bool = False,
        uncertainty_method: UncertaintyMethod = UncertaintyMethod.CONSTANT,  # Use enum
        n_mc_dropout_samples: int = 100,
        dropout_rate: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.task_type = task_type
        self.use_batch_norm = use_batch_norm
        self.uncertainty_method = uncertainty_method
        self.n_mc_dropout_samples = n_mc_dropout_samples
        self.dropout_rate = dropout_rate

        self.model = None
        self.criterion = None
        self.optimizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._is_regression_model = task_type == TaskType.REGRESSION
        self._train_residual_std = 0.0  # For 'constant' uncertainty method

        if self._is_regression_model:
            if self.uncertainty_method not in [UncertaintyMethod.CONSTANT, UncertaintyMethod.MC_DROPOUT, UncertaintyMethod.PROBABILISTIC]:
                raise ValueError(f"Unsupported uncertainty_method for regression: {self.uncertainty_method.value}. Choose from 'constant', 'mc_dropout', 'probabilistic'.")
            if self.uncertainty_method == UncertaintyMethod.PROBABILISTIC and self.output_size != 1:
                raise ValueError("For 'probabilistic' uncertainty, base output_size must be 1 (it will be expanded to 2 internally).")
        elif self.uncertainty_method != UncertaintyMethod.CONSTANT:
            logger.warning(f"uncertainty_method '{self.uncertainty_method.value}' is not applicable for classification task. Using 'constant'.")
            self.uncertainty_method = UncertaintyMethod.CONSTANT

    @property
    def name(self) -> str:
        return "PyTorchNeuralNetwork"

    def _build_model(self):
        """Dynamically builds the neural network architecture."""
        layers = []
        current_output_size = self.output_size

        if self._is_regression_model and self.uncertainty_method == UncertaintyMethod.PROBABILISTIC:
            # For probabilistic regression, output 2 values: mean and log-variance
            current_output_size = 2

        # Input layer
        layers.append(nn.Linear(self.input_size, self.hidden_size))
        if self.use_batch_norm:
            layers.append(nn.BatchNorm1d(self.hidden_size))
        layers.append(nn.ReLU())

        # Hidden layers
        for _ in range(self.hidden_layers - 1):
            layers.append(nn.Linear(self.hidden_size, self.hidden_size))
            if self.use_batch_norm:
                layers.append(nn.BatchNorm1d(self.hidden_size))
            layers.append(nn.ReLU())
            if self._is_regression_model and self.uncertainty_method == UncertaintyMethod.MC_DROPOUT and self.dropout_rate > 0:
                layers.append(nn.Dropout(self.dropout_rate))  # Add dropout for MC-Dropout

        # Output layer
        layers.append(nn.Linear(self.hidden_size, current_output_size))

        self.model = nn.Sequential(*layers).to(self.device)

        if self.task_type == TaskType.REGRESSION:
            if self.uncertainty_method == UncertaintyMethod.PROBABILISTIC:
                # Custom Negative Log-Likelihood Loss for Gaussian output
                def nll_loss(outputs, targets):
                    mean = outputs[:, 0]
                    log_var = outputs[:, 1]
                    loss = 0.5 * torch.mean(torch.exp(-log_var) * (targets - mean) ** 2 + log_var)
                    return loss

                self.criterion = nll_loss
            else:  # Standard MSE loss for other regression methods
                self.criterion = nn.MSELoss()
        elif self.task_type == TaskType.CLASSIFICATION:
            if self.output_size == 1:
                self.criterion = nn.BCEWithLogitsLoss()
            else:
                self.criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError("task_type must be 'regression' or 'classification'")

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def fit(self, X: np.ndarray, y: np.ndarray):
        if self.input_size is None:
            self.input_size = X.shape[1]

        self._build_model()  # Build model with correct output size and dropout if needed

        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(self.device)
        if self.task_type == TaskType.CLASSIFICATION and self.output_size == 1:
            y_tensor = y_tensor.unsqueeze(1)

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()  # Set model to training mode
        for epoch in range(self.n_epochs):
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()

        if self._is_regression_model and self.uncertainty_method == UncertaintyMethod.CONSTANT:
            y_pred_train = self.predict(X)  # Use standard predict which returns mean
            _train_residual_std = np.std(y - y_pred_train)  # Local variable, won't overwrite self._train_residual_std from MC-Dropout/Probabilistic
            if np.isnan(_train_residual_std):
                self._train_residual_std = 0.0
            else:
                self._train_residual_std = _train_residual_std

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)

        if self._is_regression_model and self.uncertainty_method == UncertaintyMethod.MC_DROPOUT:
            self.model.train()  # Enable dropout for MC sampling
            mc_predictions = []
            with torch.no_grad():
                for _ in range(self.n_mc_dropout_samples):
                    outputs = self.model(X_tensor)
                    mc_predictions.append(outputs.cpu().numpy().flatten())
            self.model.eval()  # Set back to eval mode
            return np.mean(mc_predictions, axis=0)  # Return mean of MC samples

        self.model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            outputs = self.model(X_tensor)
            if self.task_type == TaskType.CLASSIFICATION:
                if self.output_size == 1:
                    predictions = (torch.sigmoid(outputs) > 0.5).cpu().numpy().astype(int)
                else:
                    predictions = torch.argmax(outputs, dim=1).cpu().numpy()
            else:  # Regression
                if self.uncertainty_method == UncertaintyMethod.PROBABILISTIC:
                    # For probabilistic, predict returns only the mean
                    predictions = outputs[:, 0].cpu().numpy()
                else:
                    predictions = outputs.cpu().numpy()
        return predictions.flatten()  # Ensure flat array output

    def predict_uncertainty(self, X: np.ndarray) -> np.ndarray:
        if not self._is_regression_model:
            raise ValueError("predict_uncertainty is only available for regression models.")
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)

        if self.uncertainty_method == UncertaintyMethod.CONSTANT:
            return np.full(X.shape[0], self._train_residual_std)
        elif self.uncertainty_method == UncertaintyMethod.PROBABILISTIC:
            self.model.eval()  # Use eval mode for prediction
            with torch.no_grad():
                outputs = self.model(X_tensor)
                log_var = outputs[:, 1]
                # Standard deviation is exp(0.5 * log_var)
                uncertainty = torch.exp(0.5 * log_var).cpu().numpy()
            return uncertainty.flatten()
        elif self.uncertainty_method == UncertaintyMethod.MC_DROPOUT:
            self.model.train()  # Enable dropout for MC sampling
            mc_predictions = []
            with torch.no_grad():
                for _ in range(self.n_mc_dropout_samples):
                    outputs = self.model(X_tensor)
                    mc_predictions.append(outputs.cpu().numpy().flatten())
            self.model.eval()  # Set back to eval mode
            return np.std(mc_predictions, axis=0)  # Return std dev of MC samples
        else:
            raise ValueError(f"Unknown uncertainty_method: {self.uncertainty_method.value}")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_tensor)
            if self.task_type == TaskType.CLASSIFICATION:
                if self.output_size == 1:
                    proba = torch.sigmoid(outputs).cpu().numpy().flatten()
                    return np.vstack((1 - proba, proba)).T
                else:
                    return torch.softmax(outputs, dim=1).cpu().numpy()
            else:
                raise ValueError("predict_proba is only available for classification tasks.")

    def get_hyperparameter_search_space(self) -> Dict[str, Any]:
        space = {
            "hidden_layers": {"type": "int", "low": 1, "high": 3},
            "hidden_size": {"type": "int", "low": 32, "high": 128, "step": 32},
            "learning_rate": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True},
            "n_epochs": {"type": "int", "low": 10, "high": 50, "step": 10},
            "batch_size": {"type": "categorical", "choices": [16, 32, 64]},
            "use_batch_norm": {"type": "categorical", "choices": [True, False]},
            "uncertainty_method": {"type": "categorical", "choices": [e.value for e in UncertaintyMethod]},  # Use enum values
        }
        # Conditional hyperparameters for uncertainty methods
        space["n_mc_dropout_samples"] = {"type": "int", "low": 50, "high": 200, "step": 50}
        space["dropout_rate"] = {"type": "float", "low": 0.1, "high": 0.5, "step": 0.1}
        return space

    def get_internal_model(self):
        """
        Returns the raw underlying PyTorch model (nn.Sequential).
        """
        return self.model
