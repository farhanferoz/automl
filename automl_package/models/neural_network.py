from typing import Dict, Any  # Updated imports
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # New import for gumbel_softmax
import torch.optim as optim

from .base import BaseModel
from ..enums import UncertaintyMethod, TaskType  # Import enums
from ..logger import logger  # Import logger


class PyTorchNeuralNetwork(BaseModel):
    """
    A simple Feed-Forward Neural Network implemented using PyTorch.
    Can be configured with variable hidden layers and neurons.
    Supports optional Batch Normalization.
    Supports constant, MC-Dropout, and probabilistic layer uncertainty estimation for regression.
    Includes L1 and L2 regularization.
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
        l1_lambda: float = 0.0,  # New parameter for L1 regularization
        l2_lambda: float = 0.0,  # New parameter for L2 regularization (weight_decay)
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
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda  # Corresponds to weight_decay in Adam

        self.model = None
        self.criterion = None
        self.optimizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._is_regression_model = task_type == TaskType.REGRESSION
        self._train_residual_std = 0.0  # For 'constant' uncertainty method

        # Validate uncertainty method based on task type
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

    def build_model(self):
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
        for _ in range(self.hidden_layers - 1):  # -1 because input layer is already counted
            layers.append(nn.Linear(self.hidden_size, self.hidden_size))
            if self.use_batch_norm:
                layers.append(nn.BatchNorm1d(self.hidden_size))
            layers.append(nn.ReLU())
            # Add dropout for MC-Dropout if regression and uncertainty method is MC_DROPOUT and dropout_rate > 0
            if self._is_regression_model and self.uncertainty_method == UncertaintyMethod.MC_DROPOUT and self.dropout_rate > 0:
                layers.append(nn.Dropout(self.dropout_rate))

        # Output layer
        layers.append(nn.Linear(self.hidden_size, current_output_size))

        self.model = nn.Sequential(*layers).to(self.device)

        # Define criterion based on task type and uncertainty method
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
            if self.output_size == 1:  # Binary classification (e.g., outputs logits for BCEWithLogitsLoss)
                self.criterion = nn.BCEWithLogitsLoss()
            else:  # Multi-class classification
                self.criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError("task_type must be 'regression' or 'classification'")

        # Adam optimizer with L2 regularization (weight_decay)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.l2_lambda)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Trains the PyTorch Neural Network.
        Args:
            X (np.ndarray): Training features.
            y (np.ndarray): Training targets.
        """
        if self.input_size is None:
            self.input_size = X.shape[1]

        self.build_model()  # Build model with correct output size and dropout if needed

        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(self.device)
        # For binary classification, ensure y_tensor is 2D for BCEWithLogitsLoss
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

                # Add L1 regularization manually if specified
                if self.l1_lambda > 0:
                    l1_norm = sum(p.abs().sum() for p in self.model.parameters())
                    loss = loss + self.l1_lambda * l1_norm

                loss.backward()
                self.optimizer.step()
            # print(f"  Epoch {epoch+1}/{self.n_epochs}, Loss: {loss.item():.4f}") # For debugging

        # Calculate residual standard deviation for 'constant' uncertainty method (regression only)
        if self._is_regression_model and self.uncertainty_method == UncertaintyMethod.CONSTANT:
            y_pred_train = self.predict(X)  # Use standard predict which returns mean
            _train_residual_std = np.std(y - y_pred_train)  # Local variable, won't overwrite self._train_residual_std from MC-Dropout/Probabilistic
            if np.isnan(_train_residual_std):
                self._train_residual_std = 0.0
            else:
                self._train_residual_std = _train_residual_std

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Makes predictions with the PyTorch Neural Network.
        Args:
            X (np.ndarray): Features for prediction.
        Returns:
            np.ndarray: Predicted values. For classification, returns class labels.
        """
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)

        # Special handling for MC-Dropout during prediction
        if self._is_regression_model and self.uncertainty_method == UncertaintyMethod.MC_DROPOUT:
            self.model.train()  # Enable dropout for MC sampling
            mc_predictions = []
            with torch.no_grad():
                for _ in range(self.n_mc_dropout_samples):
                    outputs = self.model(X_tensor)
                    mc_predictions.append(outputs.cpu().numpy().flatten())
            self.model.eval()  # Set back to eval mode
            return np.mean(mc_predictions, axis=0)  # Return mean of MC samples

        self.model.eval()  # Set model to evaluation mode for deterministic prediction
        with torch.no_grad():
            outputs = self.model(X_tensor)
            if self.task_type == TaskType.CLASSIFICATION:
                if self.output_size == 1:  # Binary classification
                    predictions = (torch.sigmoid(outputs) > 0.5).cpu().numpy().astype(int)
                else:  # Multi-class classification
                    predictions = torch.argmax(outputs, dim=1).cpu().numpy()
            else:  # Regression
                if self.uncertainty_method == UncertaintyMethod.PROBABILISTIC:
                    predictions = outputs[:, 0].cpu().numpy()  # Only return mean for prediction
                else:
                    predictions = outputs.cpu().numpy()
        return predictions.flatten()  # Ensure flat array output

    def predict_uncertainty(self, X: np.ndarray) -> np.ndarray:
        """
        Estimates uncertainty for regression using constant, MC Dropout, or probabilistic methods.
        Args:
            X (np.ndarray): Features for uncertainty estimation.
        Returns:
            np.ndarray: Array of uncertainty estimates (e.g., standard deviation).
        """
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
            self.model.train()  # Enable dropout during inference for MC Dropout
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
        """
        Predicts class probabilities for classification tasks.
        Args:
            X (np.ndarray): Features for probability prediction.
        Returns:
            np.ndarray: Predicted probabilities (for binary: a 2D array [:, 0] neg, [:, 1] pos).
        """
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        if self.task_type != TaskType.CLASSIFICATION:
            raise ValueError("predict_proba is only available for classification tasks.")

        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_tensor)
            if self.output_size == 1:  # Binary classification
                proba = torch.sigmoid(outputs).cpu().numpy().flatten()
                return np.vstack((1 - proba, proba)).T  # Return (N, 2) array
            else:  # Multi-class classification
                return torch.softmax(outputs, dim=1).cpu().numpy()  # Correctly returns (N, num_classes)

    def get_hyperparameter_search_space(self) -> Dict[str, Any]:
        """
        Defines the hyperparameter search space for PyTorchNeuralNetwork.
        Returns:
            Dict[str, Any]: Search space configuration for Optuna.
        """
        space = {
            "hidden_layers": {"type": "int", "low": 1, "high": 3},
            "hidden_size": {"type": "int", "low": 32, "high": 128, "step": 32},
            "learning_rate": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True},
            "n_epochs": {"type": "int", "low": 10, "high": 50, "step": 10},
            "batch_size": {"type": "categorical", "choices": [16, 32, 64]},
            "use_batch_norm": {"type": "categorical", "choices": [True, False]},
            "uncertainty_method": {"type": "categorical", "choices": [e.value for e in UncertaintyMethod]},  # Use enum values
            "l1_lambda": {"type": "float", "low": 1e-6, "high": 1e-2, "log": True},  # L1 regularization
            "l2_lambda": {"type": "float", "low": 1e-6, "high": 1e-2, "log": True},  # L2 regularization
            "activation": {"type": "categorical", "choices": ["ReLU", "Tanh"]},
        }
        # Conditional hyperparameters for uncertainty methods
        # These are added to the space, but their suggestion will be conditional in AutoML._sample_params_for_trial
        space["n_mc_dropout_samples"] = {"type": "int", "low": 50, "high": 200, "step": 50}
        space["dropout_rate"] = {"type": "float", "low": 0.1, "high": 0.5, "step": 0.1}
        return space

    def get_internal_model(self):
        """
        Returns the raw underlying PyTorch model (nn.Sequential).
        """
        return self.model

    def get_classifier_predictions(self, X: np.ndarray, y_true_original: np.ndarray):
        raise NotImplementedError("PyTorchNeuralNetwork is not a composite model and does not have an internal classifier for separate prediction.")


# --- NEW MODEL: FlexibleHiddenLayersNN ---
class FlexibleHiddenLayersNN(BaseModel):
    """
    A PyTorch Neural Network with a dynamic number of active hidden layers.
    It includes an internal feedforward network that predicts 'n' (1 to max_hidden_layers),
    where 'n' determines how many of the final hidden layers are active.
    The first (max_hidden_layers - n) hidden layers act as identity layers.
    Supports optional Batch Normalization and L1/L2 regularization.
    Supports constant, MC-Dropout, and probabilistic uncertainty estimation for regression.
    """

    def __init__(
        self,
        input_size: int = None,
        max_hidden_layers: int = 3,  # Maximum possible hidden layers
        hidden_size: int = 64,  # Size of each potential hidden layer
        output_size: int = 1,
        learning_rate: float = 0.001,
        n_epochs: int = 10,
        batch_size: int = 32,
        task_type: TaskType = TaskType.REGRESSION,
        use_batch_norm: bool = False,
        uncertainty_method: UncertaintyMethod = UncertaintyMethod.CONSTANT,
        n_mc_dropout_samples: int = 100,
        dropout_rate: float = 0.1,
        l1_lambda: float = 0.0,
        l2_lambda: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_size = input_size
        self.max_hidden_layers = max_hidden_layers
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
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda

        self.model = None  # This will be an instance of _FlexibleNN_Module
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
        return "FlexibleNeuralNetwork"

    class _FlexibleNN_Module(nn.Module):
        """Internal PyTorch nn.Module for FlexibleHiddenLayersNN."""

        def __init__(self, input_size, max_hidden_layers, hidden_size, output_size, task_type, use_batch_norm, uncertainty_method, dropout_rate):
            super().__init__()
            self.input_size = input_size
            self.max_hidden_layers = max_hidden_layers
            self.hidden_size = hidden_size
            self.output_size = output_size
            self.task_type = task_type
            self.use_batch_norm = use_batch_norm
            self.uncertainty_method = uncertainty_method
            self.dropout_rate = dropout_rate

            # Output size for the main network's output layer
            final_output_neurons = self.output_size
            if self.task_type == TaskType.REGRESSION and self.uncertainty_method == UncertaintyMethod.PROBABILISTIC:
                final_output_neurons = 2  # Mean and Log-Variance

            # n-predictor: Takes full input features and outputs logits for n (1 to max_hidden_layers)
            self.n_predictor = nn.Sequential(
                nn.Linear(input_size, max(1, input_size // 2)), nn.ReLU(), nn.Linear(max(1, input_size // 2), max_hidden_layers)  # Logits for each possible 'n'
            )

            # Define the potential hidden layers (all max_hidden_layers of them)
            # Each "layer" here is a block of (Linear -> BatchNorm -> ReLU -> Dropout)
            self.hidden_layers_blocks = nn.ModuleList()
            for i in range(max_hidden_layers):
                block_layers = []
                in_features = input_size if i == 0 else hidden_size
                block_layers.append(nn.Linear(in_features, hidden_size))
                if use_batch_norm:
                    block_layers.append(nn.BatchNorm1d(hidden_size))
                block_layers.append(nn.ReLU())
                # Add dropout to active hidden layers if MC-Dropout is enabled and rate > 0
                if task_type == TaskType.REGRESSION and uncertainty_method == UncertaintyMethod.MC_DROPOUT and dropout_rate > 0:
                    block_layers.append(nn.Dropout(dropout_rate))
                self.hidden_layers_blocks.append(nn.Sequential(*block_layers))

            # The final output layer connects from the last 'active' hidden_size
            self.output_layer = nn.Linear(hidden_size, final_output_neurons)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # 1. Predict 'n' (number of active hidden layers)
            n_logits = self.n_predictor(x)

            if self.training:
                # Use Gumbel-Softmax for differentiable selection during training
                # tau is the temperature parameter, lower means closer to one-hot
                n_probs = F.gumbel_softmax(n_logits, tau=0.5, hard=False, dim=1)
                # Convert soft probabilities to a hard integer choice for routing the layers.
                # This makes the architecture selection 'discrete' but the n-predictor is still trained.
                n_chosen_indices = torch.argmax(n_probs, dim=1)  # 0-indexed choice for 'n'
            else:
                # For inference, use hard argmax
                n_chosen_indices = torch.argmax(n_logits, dim=1)

            n_actual = n_chosen_indices + 1  # Convert 0-indexed to 1-indexed n (1 to max_hidden_layers)

            current_output = x  # Input to the first layer in the dynamic sequence

            # Apply hidden layers dynamically
            for i in range(self.max_hidden_layers):
                # Determine for each sample in the batch if this layer `i` should be active or identity
                # A layer `i` is active if its 0-indexed position is greater than or equal to
                # (max_hidden_layers - n_actual).
                active_layer_mask = (i >= (self.max_hidden_layers - n_actual)).unsqueeze(1)  # Shape (batch_size, 1)

                # Get input for the current block (either original input or previous layer's output)
                if i == 0:
                    # First block always uses original `x` as its primary input source
                    input_to_block = x
                else:
                    # Subsequent blocks use the output from the previous `current_output`
                    input_to_block = current_output

                # Apply the current hidden layer block (Linear -> BN -> ReLU -> Dropout)
                active_layer_output = self.hidden_layers_blocks[i](input_to_block)

                # Combine based on the mask: if active_mask is True, use active_layer_output, else current_output (identity)
                current_output = torch.where(active_layer_mask, active_layer_output, current_output)

            # Final output layer
            final_output = self.output_layer(current_output)

            return final_output

    def _build_model(self):
        """Builds the internal PyTorch nn.Module for the FlexibleHiddenLayersNN."""
        # The internal _FlexibleNN_Module handles the dynamic architecture construction
        self.model = self._FlexibleNN_Module(
            input_size=self.input_size,
            max_hidden_layers=self.max_hidden_layers,
            hidden_size=self.hidden_size,
            output_size=self.output_size,
            task_type=self.task_type,
            use_batch_norm=self.use_batch_norm,
            uncertainty_method=self.uncertainty_method,
            dropout_rate=self.dropout_rate,
        ).to(self.device)

        # Set up criterion based on task type and uncertainty method
        if self.task_type == TaskType.REGRESSION:
            if self.uncertainty_method == UncertaintyMethod.PROBABILISTIC:

                def nll_loss(outputs, targets):
                    mean = outputs[:, 0]
                    log_var = outputs[:, 1]
                    loss = 0.5 * torch.mean(torch.exp(-log_var) * (targets - mean) ** 2 + log_var)
                    return loss

                self.criterion = nll_loss
            else:
                self.criterion = nn.MSELoss()
        elif self.task_type == TaskType.CLASSIFICATION:
            if self.output_size == 1:  # Binary classification
                self.criterion = nn.BCEWithLogitsLoss()
            else:  # Multi-class classification
                self.criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError("task_type must be 'regression' or 'classification'")

        # Adam optimizer with L2 regularization (weight_decay)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.l2_lambda)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Trains the FlexibleHiddenLayersNN.
        Args:
            X (np.ndarray): Training features.
            y (np.ndarray): Training targets.
        """
        if self.input_size is None:
            self.input_size = X.shape[1]

        self._build_model()  # Build the internal PyTorch nn.Module

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

                # Add L1 regularization manually
                if self.l1_lambda > 0:
                    l1_norm = sum(p.abs().sum() for p in self.model.parameters())
                    loss = loss + self.l1_lambda * l1_norm

                loss.backward()
                self.optimizer.step()

        # Calculate residual standard deviation for 'constant' uncertainty method (regression only)
        if self._is_regression_model and self.uncertainty_method == UncertaintyMethod.CONSTANT:
            y_pred_train = self.predict(X)  # Use standard predict which returns mean
            _train_residual_std = np.std(y - y_pred_train)
            if np.isnan(_train_residual_std):
                self._train_residual_std = 0.0
            else:
                self._train_residual_std = _train_residual_std

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Makes predictions with the FlexibleHiddenLayersNN.
        Args:
            X (np.ndarray): Features for prediction.
        Returns:
            np.ndarray: Predicted values. For classification, returns class labels.
        """
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

        self.model.eval()  # Set model to evaluation mode for deterministic prediction
        with torch.no_grad():
            outputs = self.model(X_tensor)
            if self.task_type == TaskType.CLASSIFICATION:
                if self.output_size == 1:  # Binary classification
                    predictions = (torch.sigmoid(outputs) > 0.5).cpu().numpy().astype(int)
                else:  # Multi-class classification
                    predictions = torch.argmax(outputs, dim=1).cpu().numpy()
            else:  # Regression
                if self.uncertainty_method == UncertaintyMethod.PROBABILISTIC:
                    predictions = outputs[:, 0].cpu().numpy()  # Only return mean for prediction
                else:
                    predictions = outputs.cpu().numpy()
        return predictions.flatten()  # Ensure flat array output

    def predict_uncertainty(self, X: np.ndarray) -> np.ndarray:
        """
        Estimates uncertainty for regression using constant, MC Dropout, or probabilistic methods.
        Args:
            X (np.ndarray): Features for uncertainty estimation.
        Returns:
            np.ndarray: Array of uncertainty estimates (e.g., standard deviation).
        """
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
            self.model.train()  # Enable dropout during inference for MC Dropout
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
        """
        Predicts class probabilities for classification tasks.
        Args:
            X (np.ndarray): Features for probability prediction.
        Returns:
            np.ndarray: Predicted probabilities (for binary: a 2D array [:, 0] neg, [:, 1] pos).
        """
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        if self.task_type != TaskType.CLASSIFICATION:
            raise ValueError("predict_proba is only available for classification tasks.")

        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_tensor)
            if self.output_size == 1:  # Binary classification
                proba = torch.sigmoid(outputs).cpu().numpy().flatten()
                return np.vstack((1 - proba, proba)).T  # Return (N, 2) array
            else:  # Multi-class classification
                return torch.softmax(outputs, dim=1).cpu().numpy()

    def get_hyperparameter_search_space(self) -> Dict[str, Any]:
        """
        Defines the hyperparameter search space for FlexibleHiddenLayersNN.
        Returns:
            Dict[str, Any]: Search space configuration for Optuna.
        """
        space = {
            "max_hidden_layers": {"type": "int", "low": 1, "high": 5},  # Maximum number of hidden layers
            "hidden_size": {"type": "int", "low": 32, "high": 128, "step": 32},
            "learning_rate": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True},
            "n_epochs": {"type": "int", "low": 10, "high": 50, "step": 10},
            "batch_size": {"type": "categorical", "choices": [16, 32, 64]},
            "use_batch_norm": {"type": "categorical", "choices": [True, False]},
            "uncertainty_method": {"type": "categorical", "choices": [e.value for e in UncertaintyMethod]},
            "l1_lambda": {"type": "float", "low": 1e-6, "high": 1e-2, "log": True},
            "l2_lambda": {"type": "float", "low": 1e-6, "high": 1e-2, "log": True},
            "activation": {"type": "categorical", "choices": ["ReLU", "Tanh"]},
        }
        # Conditional hyperparameters for uncertainty methods, suggested by AutoML if needed
        space["n_mc_dropout_samples"] = {"type": "int", "low": 50, "high": 200, "step": 50}
        space["dropout_rate"] = {"type": "float", "low": 0.1, "high": 0.5, "step": 0.1}
        return space

    def get_internal_model(self):
        """
        Returns the raw underlying PyTorch model (nn.Module).
        """
        return self.model

    def get_classifier_predictions(self, X: np.ndarray, y_true_original: np.ndarray):
        raise NotImplementedError("FlexibleHiddenLayersNN is not a composite model and does not have an internal classifier for separate prediction.")
