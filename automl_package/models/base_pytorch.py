"""Base classes for PyTorch models."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

from automl_package.enums import LearnedRegularizationType, NClassesSelectionMethod, TaskType, UncertaintyMethod
from automl_package.logger import logger
from automl_package.models.base import BaseModel
from automl_package.utils.metrics import Metrics
from automl_package.utils.numerics import aggregate_stats, log_erfc


class PyTorchModelBase(BaseModel, ABC):
    """Base class for PyTorch-based neural network models.

    Handles common initialization, regularization, and training loop logic.
    """

    def __init__(
        self,
        input_size: int | None = None,
        output_size: int = 1,
        learning_rate: float = 0.001,
        n_epochs: int = 10,
        batch_size: int = 32,
        task_type: TaskType = TaskType.REGRESSION,
        use_batch_norm: bool = False,
        uncertainty_method: UncertaintyMethod = UncertaintyMethod.CONSTANT,
        n_mc_dropout_samples: int = 100,
        dropout_rate: float = 0.1,
        learn_regularization_lambdas: bool = False,
        learned_regularization_type: LearnedRegularizationType = LearnedRegularizationType.L1_L2,
        l1_lambda: float = 0.0,
        l2_lambda: float = 0.0,
        lambda_learning_rate: float = 1e-5,
        random_seed: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Initializes the PyTorchModelBase.

        Args:
            input_size (int, optional): The number of input features.
            output_size (int): The number of output features.
            learning_rate (float): Learning rate for the optimizer.
            n_epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
            task_type (TaskType): Type of task (regression or classification).
            use_batch_norm (bool): Whether to use batch normalization.
            uncertainty_method (UncertaintyMethod): Method for uncertainty estimation.
            n_mc_dropout_samples (int): Number of MC dropout samples for uncertainty.
            dropout_rate (float): Dropout rate for MC dropout.
            learn_regularization_lambdas (bool): Whether to learn regularization lambdas.
            learned_regularization_type (LearnedRegularizationType): Type of learned regularization.
            l1_lambda (float): L1 regularization strength.
            l2_lambda (float): L2 regularization strength.
            lambda_learning_rate (float): Learning rate for lambda optimization.
            random_seed (int, optional): Random seed for reproducibility.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.task_type = task_type
        self.use_batch_norm = use_batch_norm
        self.uncertainty_method = uncertainty_method
        self.n_mc_dropout_samples = n_mc_dropout_samples
        self.dropout_rate = dropout_rate
        self.gumbel_tau: float | None = None
        self.n_classes_selection_method: NClassesSelectionMethod | None = None
        self.n_classes_predictor_learning_rate: float | None = None

        self.learn_regularization_lambdas = learn_regularization_lambdas
        self.learned_regularization_type = learned_regularization_type
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.lambda_learning_rate = lambda_learning_rate
        self.include_reg_loss_in_val_loss = False
        self._returns_multiple_outputs: bool = False  # New flag

        self.l1_log_lambda, self.l2_log_lambda = None, None
        self.using_l1_regularization = (
            self.learned_regularization_type in [LearnedRegularizationType.L1_ONLY, LearnedRegularizationType.L1_L2] if self.learn_regularization_lambdas else l1_lambda > 0
        )
        self.using_l2_regularization = (
            self.learned_regularization_type in [LearnedRegularizationType.L2_ONLY, LearnedRegularizationType.L1_L2] if self.learn_regularization_lambdas else l2_lambda > 0
        )

        self.random_seed = random_seed

        self.model: nn.Module | None = None
        self.criterion: nn.Module | None = None
        self.optimizer: optim.Optimizer | None = None
        self.lambda_optimizer: optim.Optimizer | None = None
        self.device: torch.device | None = None  # Initialize to None
        self.is_regression_model = task_type == TaskType.REGRESSION
        self._train_residual_std = 0.0  # For 'constant' uncertainty method

        # Validate uncertainty method based on task type
        if self.is_regression_model:
            if self.uncertainty_method not in [UncertaintyMethod.CONSTANT, UncertaintyMethod.MC_DROPOUT, UncertaintyMethod.PROBABILISTIC]:
                raise ValueError(f"Unsupported uncertainty_method for regression: {self.uncertainty_method.value}. Choose from 'constant', 'mc_dropout', 'probabilistic'.")
            if self.uncertainty_method == UncertaintyMethod.PROBABILISTIC and self.output_size != 1:
                raise ValueError("For 'probabilistic' uncertainty, base output_size must be 1 (it will be expanded to 2 internally).")
        elif self.uncertainty_method != UncertaintyMethod.CONSTANT:
            logger.warning(f"uncertainty_method '{self.uncertainty_method.value}' is not applicable for classification task. Using 'constant'.")
            self.uncertainty_method = UncertaintyMethod.CONSTANT

    @property
    @abstractmethod
    def name(self) -> str:
        """Abstract property for the model's name."""
        raise NotImplementedError

    def build_model(self) -> None:
        """Abstract method to be implemented by subclasses to define the network architecture."""
        raise NotImplementedError("Subclasses must implement build_model()")

    def _setup_optimizers(self, model: nn.Module) -> None:
        self.optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        if self.learn_regularization_lambdas:
            lambda_params = []
            if self.l1_log_lambda is not None:
                lambda_params.append(self.l1_log_lambda)
            if self.l2_log_lambda is not None:
                lambda_params.append(self.l2_log_lambda)
            if lambda_params:  # Only create lambda_optimizer if there are lambdas to optimize
                self.lambda_optimizer = optim.Adam(lambda_params, lr=self.lambda_learning_rate)
        else:
            # If not learning lambdas, apply fixed L2 regularization via weight_decay to the main optimizer
            # Note: weight_decay is applied to all parameters in the optimizer, so if n_predictor has a separate group,
            # it will also have weight_decay applied unless explicitly handled.
            # For now, we apply it to the main group and assume n_predictor doesn't need L2 weight_decay.
            # If n_predictor needs L2, it should be added to its param_group.
            if self.l2_lambda > 0:
                for group in self.optimizer.param_groups:
                    group["weight_decay"] = self.l2_lambda

    def _calculate_regularization_loss(self, base_loss: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
        """Calculates and adds regularization loss to the base loss."""
        loss = base_loss
        if self.learn_regularization_lambdas:
            l1_lambda_val = torch.exp(self.l1_log_lambda) if self.using_l1_regularization else None
            l2_lambda_val = torch.exp(self.l2_log_lambda) if self.using_l2_regularization else None
            d, l1_sum, l2_sum = aggregate_stats(model=model, include_bias=False)

            if self.learned_regularization_type == LearnedRegularizationType.L1_ONLY:
                loss = loss - d * torch.log(l1_lambda_val / 2.0) + l1_lambda_val * l1_sum
            elif self.learned_regularization_type == LearnedRegularizationType.L2_ONLY:
                loss = loss - (d / 2.0) * torch.log(l2_lambda_val / torch.pi) + l2_lambda_val * l2_sum
            else:  # L1_L2
                assert self.learned_regularization_type == LearnedRegularizationType.L1_L2
                log_z = (
                    torch.log(torch.pi / l2_lambda_val) / 2.0 + torch.square(l1_lambda_val) / (4.0 * l2_lambda_val) + log_erfc(l1_lambda_val / (2.0 * torch.sqrt(l2_lambda_val)))
                )
                loss = loss + d * log_z + l1_lambda_val * l1_sum + l2_lambda_val * l2_sum
        elif self.l1_lambda > 0:  # Fixed L1 regularization
            _, l1_sum, _ = aggregate_stats(model=model, include_bias=False)
            loss = loss + self.l1_lambda * l1_sum

        return loss

    def fit(self, x: np.ndarray, y: np.ndarray) -> int:
        """Trains the PyTorch Neural Network.

        :param x: The input features.
        :param y: The target values.
        :return: The number of epochs trained.
        """
        if self.input_size is None:
            self.input_size = 1 if x.ndim == 1 else x.shape[1]

        # Set device here, just before model building and training
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.learn_regularization_lambdas:
            if self.using_l1_regularization:
                self.l1_log_lambda = nn.Parameter(torch.tensor(np.log(1e-4), dtype=torch.float32))
            if self.using_l2_regularization:
                self.l2_log_lambda = nn.Parameter(torch.tensor(np.log(1e-4), dtype=torch.float32))

        if self.random_seed is not None:
            torch.manual_seed(self.random_seed)
            np.random.seed(self.random_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.random_seed)

        self.build_model()  # Build model with correct output size and dropout if needed
        self._setup_optimizers(self.model)  # Setup optimizers after model is built

        # Split data for early stopping if enabled
        x_val, y_val = None, None
        x_val_tensor, y_val_tensor = None, None

        if self.early_stopping_rounds is not None and self.validation_fraction > 0:
            x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=self.validation_fraction, random_state=42)
        else:
            x_train, y_train = x, y

        x_train_tensor = torch.tensor(x_train, dtype=torch.float32).to(self.device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        if self.task_type == TaskType.CLASSIFICATION:
            y_train_tensor = y_train_tensor.long()
        # Ensure y_train_tensor is 2D for both regression and binary classification
        if (self.task_type == TaskType.REGRESSION) or (self.task_type == TaskType.CLASSIFICATION and self.output_size == 1):
            y_train_tensor = y_train_tensor.unsqueeze(1)

        train_dataset = torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        if x_val is not None:
            x_val_tensor = torch.tensor(x_val, dtype=torch.float32).to(self.device)
            y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(self.device)
            if self.task_type == TaskType.CLASSIFICATION:
                y_val_tensor = y_val_tensor.long()
            if (self.task_type == TaskType.REGRESSION) or (self.task_type == TaskType.CLASSIFICATION and self.output_size == 1):
                y_val_tensor = y_val_tensor.unsqueeze(1)

        best_val_loss = float("inf")
        patience_counter = 0
        best_model_state = None
        best_epoch = 0

        for epoch in range(int(self.n_epochs)):
            self.model.train()  # Set model to training mode
            epoch_log_probs = []
            for batch_x, batch_y in train_dataloader:
                self.optimizer.zero_grad()
                if self.lambda_optimizer:
                    self.lambda_optimizer.zero_grad()

                if self._returns_multiple_outputs:
                    outputs_tuple = self.model(batch_x)
                    final_predictions = outputs_tuple[0]
                    log_prob_for_reinforce = outputs_tuple[3] # Assuming log_prob is the 4th element
                else:
                    final_predictions = self.model(batch_x)
                    log_prob_for_reinforce = None

                loss = self.criterion(final_predictions, batch_y)

                loss = self._calculate_regularization_loss(loss, self.model)

                loss.backward()
                self.optimizer.step()
                if self.lambda_optimizer:
                    self.lambda_optimizer.step()

                if self.n_classes_selection_method == NClassesSelectionMethod.REINFORCE and log_prob_for_reinforce is not None:
                    epoch_log_probs.append(log_prob_for_reinforce)

            # Early stopping check
            if self.early_stopping_rounds is not None and x_val is not None:
                self.model.eval()  # Set model to evaluation mode
                with torch.no_grad():
                    # val_outputs now returns (final_predictions, classifier_logits_out, selected_k_values, log_prob_for_reinforce)
                    if self._returns_multiple_outputs:
                        val_outputs_tuple = self.model(x_val_tensor)
                        val_final_predictions = val_outputs_tuple[0]
                    else:
                        val_final_predictions = self.model(x_val_tensor)
                    val_loss = self.criterion(val_final_predictions, y_val_tensor)
                    if self.include_reg_loss_in_val_loss:
                        val_loss = self._calculate_regularization_loss(val_loss, self.model)
                    val_loss = val_loss.item()

                if self.n_classes_selection_method == NClassesSelectionMethod.REINFORCE:
                    self.model.n_classes_strategy.on_epoch_end(validation_loss=val_loss, epoch_log_probs=epoch_log_probs)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = self.model.state_dict()  # Save best model state
                    best_epoch = epoch
                else:
                    patience_counter += 1

                if patience_counter >= self.early_stopping_rounds:
                    logger.info(f"Early stopping at epoch {best_epoch + 1}")
                    break
            else:
                best_epoch = epoch

        # Load best model state if early stopping occurred
        if best_model_state:
            self.model.load_state_dict(best_model_state)

        # Calculate residual standard deviation for 'constant' uncertainty method (regression only)
        if self.is_regression_model and self.uncertainty_method == UncertaintyMethod.CONSTANT:
            y_pred_train = self.predict(x)  # Use standard predict which returns mean
            _train_residual_std = np.std(y - y_pred_train)  # Local variable, won't overwrite self._train_residual_std from MC-Dropout/Probabilistic
            if np.isnan(_train_residual_std):
                self._train_residual_std = 0.0
            else:
                self._train_residual_std = _train_residual_std
        # Log learned lambda values if applicable
        if self.learn_regularization_lambdas:
            if self.l1_log_lambda is not None:
                logger.info(f"Learned L1 Lambda: {torch.exp(self.l1_log_lambda).item():.6f}")
            if self.l2_log_lambda is not None:
                logger.info(f"Learned L2 Lambda: {torch.exp(self.l2_log_lambda).item():.6f}")

        return best_epoch + 1  # Return the number of epochs actually used

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Makes predictions with the PyTorch Neural Network.

        Args:
            x (np.ndarray): Features for prediction.

        Returns:
            np.ndarray: Predicted values. For classification, returns class labels.
        """
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)

        # Special handling for MC-Dropout during prediction
        if self.is_regression_model and self.uncertainty_method == UncertaintyMethod.MC_DROPOUT:
            self.model.train()  # Enable dropout for MC sampling
            mc_predictions = []
            with torch.no_grad():
                for _ in range(self.n_mc_dropout_samples):
                    if self._returns_multiple_outputs:
                        outputs, *_, _ = self.model(x_tensor)
                    else:
                        outputs = self.model(x_tensor)
                    mc_predictions.append(outputs.cpu().numpy().flatten())
            self.model.eval()  # Set back to eval mode
            return np.mean(mc_predictions, axis=0)  # Return mean of MC samples

        self.model.eval()  # Set model to evaluation mode for deterministic prediction
        with torch.no_grad():
            if self._returns_multiple_outputs:
                outputs, *_, _ = self.model(x_tensor)
            else:
                outputs = self.model(x_tensor)
            if self.task_type == TaskType.CLASSIFICATION:
                predictions = (torch.sigmoid(outputs) > 0.5).cpu().numpy().astype(int) if self.output_size == 1 else torch.argmax(outputs, dim=1).cpu().numpy()
            else:  # Regression
                predictions = outputs[:, 0].cpu().numpy() if self.uncertainty_method == UncertaintyMethod.PROBABILISTIC else outputs.cpu().numpy()
        return predictions.flatten()  # Ensure flat array output

    def predict_uncertainty(self, x: np.ndarray) -> np.ndarray:
        """Estimates uncertainty for regression using constant, MC Dropout, or probabilistic methods.

        Args:
            x (np.ndarray): Features for uncertainty estimation.

        Returns:
            np.ndarray: Array of uncertainty estimates (e.g., standard deviation).
        """
        if not self.is_regression_model:
            raise ValueError("predict_uncertainty is only available for regression models.")
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)

        if self.uncertainty_method == UncertaintyMethod.CONSTANT:
            return np.full(x.shape[0], self._train_residual_std)
        if self.uncertainty_method == UncertaintyMethod.PROBABILISTIC:
            self.model.eval()  # Use eval mode for prediction
            with torch.no_grad():
                if self._returns_multiple_outputs:
                    outputs, *_, _ = self.model(x_tensor)
                else:
                    outputs = self.model(x_tensor)
                log_var = outputs[:, 1]
                # Standard deviation is exp(0.5 * log_var)
                uncertainty = torch.exp(0.5 * log_var).cpu().numpy()
            return uncertainty.flatten()
        if self.uncertainty_method == UncertaintyMethod.MC_DROPOUT:
            self.model.train()  # Enable dropout during inference for MC Dropout
            mc_predictions = []
            with torch.no_grad():
                for _ in range(self.n_mc_dropout_samples):
                    if self._returns_multiple_outputs:
                        outputs, *_, _ = self.model(x_tensor)
                    else:
                        outputs = self.model(x_tensor)
                    mc_predictions.append(outputs.cpu().numpy().flatten())
            self.model.eval()  # Set back to eval mode
            return np.std(mc_predictions, axis=0)  # Return std dev of MC samples
        raise ValueError(f"Unknown uncertainty_method: {self.uncertainty_method.value}")

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Predicts class probabilities for classification tasks.

        Args:
            x (np.ndarray): Features for probability prediction.

        Returns:
            np.ndarray: Predicted probabilities (for binary: a 2D array [:, 0] neg, [:, 1] pos).
        """
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        if self.task_type != TaskType.CLASSIFICATION:
            raise ValueError("predict_proba is only available for classification tasks.")

        self.model.eval()
        x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            if self._returns_multiple_outputs:
                outputs, *_, _ = self.model(x_tensor)
            else:
                outputs = self.model(x_tensor)
            if self.output_size == 1:  # Binary classification
                proba = torch.sigmoid(outputs).cpu().numpy().flatten()
                return np.vstack((1 - proba, proba)).T  # Return (N, 2) array
            # Multi-class classification
            return torch.softmax(outputs, dim=1).cpu().numpy()  # Correctly returns (N, num_classes)

    def get_internal_model(self) -> torch.nn.Module | None:
        """Returns the raw underlying PyTorch model (nn.Sequential)."""
        return self.model

    def get_num_parameters(self) -> int:
        """Returns the total number of trainable parameters in the model."""
        if self.model is None:
            return 0
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        if self.learn_regularization_lambdas:
            if self.l1_log_lambda is not None:
                num_params += self.l1_log_lambda.numel()
            if self.l2_log_lambda is not None:
                num_params += self.l2_log_lambda.numel()
        return num_params

    def get_hyperparameter_search_space(self) -> dict[str, Any]:
        """Returns the hyperparameter search space for the base PyTorch model.

        Subclasses should call this method and extend the returned dictionary.
        """
        space = {
            "learning_rate": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True},
            "l1_lambda": {"type": "float", "low": 1e-8, "high": 1e-2, "log": True},
            "l2_lambda": {"type": "float", "low": 1e-8, "high": 1e-2, "log": True},
            "dropout_rate": {"type": "float", "low": 0.0, "high": 0.5, "step": 0.1},
        }
        # Add uncertainty method only for regression tasks
        if self.is_regression_model:
            space["n_mc_dropout_samples"] = {"type": "int", "low": 50, "high": 200, "step": 50}

        return space

    def evaluate(self, x: np.ndarray, y: np.ndarray, save_path: str = "metrics") -> np.ndarray:
        """Evaluates the model on a given dataset and saves the metrics.

        Args:
            x (np.ndarray): Feature matrix for evaluation.
            y (np.ndarray): True labels for evaluation.
            save_path (str): Directory to save the metrics files.

        Returns:
            np.ndarray: The predictions made by the model.
        """
        y_pred = self.predict(x)
        y_proba = self.predict_proba(x) if self.task_type == TaskType.CLASSIFICATION else None
        metrics_calculator = Metrics(task_type=self.task_type.value, model_name=self.name, x_data=x, y_true=y, y_pred=y_pred, y_proba=y_proba)
        metrics_calculator.save_metrics(save_path)
        return y_pred

    def get_classifier_predictions(self, x: np.ndarray, y_true_original: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """This method is not applicable to general PyTorch models as they do not have.

        an internal classifier for the purpose of discretizing regression targets.
        It is intended for composite models like ProbabilisticRegressionModel.
        """
        raise NotImplementedError("get_classifier_predictions is not implemented for this model type.")
