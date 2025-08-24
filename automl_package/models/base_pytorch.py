"""Base classes for PyTorch models."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, mean_squared_error

from automl_package.enums import LearnedRegularizationType, TaskType, UncertaintyMethod
from automl_package.logger import logger
from automl_package.models.base import BaseModel
from automl_package.utils.metrics import Metrics
from automl_package.utils.numerics import aggregate_stats, log_erfc


class PyTorchModelBase(BaseModel, ABC):
    """Base class for PyTorch-based neural network models."""

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
        """Initializes the PyTorchModelBase."""
        super().__init__(early_stopping_rounds=kwargs.get("early_stopping_rounds"), cv_folds=kwargs.get("cv_folds"), **kwargs)
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
        self.random_seed = random_seed
        self.is_regression_model = task_type == TaskType.REGRESSION
        self.learn_regularization_lambdas = learn_regularization_lambdas
        self.learned_regularization_type = learned_regularization_type
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.lambda_learning_rate = lambda_learning_rate
        self.l1_log_lambda, self.l2_log_lambda = None, None
        self.using_l1_regularization = (
            self.learned_regularization_type in [LearnedRegularizationType.L1_ONLY, LearnedRegularizationType.L1_L2] if self.learn_regularization_lambdas else l1_lambda > 0
        )
        self.using_l2_regularization = (
            self.learned_regularization_type in [LearnedRegularizationType.L2_ONLY, LearnedRegularizationType.L1_L2] if self.learn_regularization_lambdas else l2_lambda > 0
        )
        self.model: nn.Module | None = None
        self.criterion: nn.Module | None = None
        self.optimizer: optim.Optimizer | None = None
        self.lambda_optimizer: optim.Optimizer | None = None
        self.device: torch.device | None = None
        self._train_residual_std = 0.0

    @property
    @abstractmethod
    def name(self) -> str:
        """Returns the name of the model."""
        raise NotImplementedError

    @abstractmethod
    def build_model(self) -> None:
        """Builds the model architecture."""
        raise NotImplementedError("Subclasses must implement build_model()")

    def _setup_optimizers(self, model: nn.Module) -> None:
        """Sets up the optimizers for the model."""
        self.optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        if self.learn_regularization_lambdas:
            lambda_params = [p for p in [self.l1_log_lambda, self.l2_log_lambda] if p is not None]
            if lambda_params:
                self.lambda_optimizer = optim.Adam(lambda_params, lr=self.lambda_learning_rate)
        elif self.l2_lambda > 0:
            for group in self.optimizer.param_groups:
                group["weight_decay"] = self.l2_lambda

    def _calculate_regularization_loss(self, base_loss: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
        """Calculates the regularization loss."""
        loss = base_loss
        d, l1_sum, l2_sum = aggregate_stats(model=model, include_bias=False)
        if self.learn_regularization_lambdas:
            l1_lambda_val = torch.exp(self.l1_log_lambda) if self.using_l1_regularization and self.l1_log_lambda is not None else None
            l2_lambda_val = torch.exp(self.l2_log_lambda) if self.using_l2_regularization and self.l2_log_lambda is not None else None

            if self.learned_regularization_type == LearnedRegularizationType.L1_ONLY and l1_lambda_val is not None:
                loss = loss - d * torch.log(l1_lambda_val / 2.0) + l1_lambda_val * l1_sum
            elif self.learned_regularization_type == LearnedRegularizationType.L2_ONLY and l2_lambda_val is not None:
                loss = loss - (d / 2.0) * torch.log(l2_lambda_val / torch.pi) + l2_lambda_val * l2_sum
            elif l1_lambda_val is not None and l2_lambda_val is not None:
                log_z = (
                    torch.log(torch.pi / l2_lambda_val) / 2.0 + torch.square(l1_lambda_val) / (4.0 * l2_lambda_val) + log_erfc(l1_lambda_val / (2.0 * torch.sqrt(l2_lambda_val)))
                )
                loss = loss + d * log_z + l1_lambda_val * l1_sum + l2_lambda_val * l2_sum
        elif self.l1_lambda > 0:
            loss = loss + self.l1_lambda * l1_sum
        return loss

    def _fit_single(
        self, x: np.ndarray, y: np.ndarray, x_val: np.ndarray | None = None, y_val: np.ndarray | None = None, forced_iterations: int | None = None
    ) -> tuple[int, list[float]]:
        """Fits a single model instance.

        Args:
            x (np.ndarray): The training features.
            y (np.ndarray): The training targets.
            x_val (np.ndarray | None): The validation features.
            y_val (np.ndarray | None): The validation targets.
            forced_iterations (int | None): If provided, train for this many iterations, ignoring early stopping.

        Returns:
            tuple[int, list[float]]: A tuple containing:
                - The number of iterations the model was trained for.
                - A list of the validation loss values for each epoch.
        """
        if self.input_size is None:
            self.input_size = 1 if x.ndim == 1 else x.shape[1]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.random_seed is not None:
            torch.manual_seed(self.random_seed)
            np.random.seed(self.random_seed)
        self.build_model()
        if self.learn_regularization_lambdas:
            if self.using_l1_regularization:
                self.l1_log_lambda = nn.Parameter(torch.tensor(np.log(1e-4), dtype=torch.float32))
            if self.using_l2_regularization:
                self.l2_log_lambda = nn.Parameter(torch.tensor(np.log(1e-4), dtype=torch.float32))
        self._setup_optimizers(self.model)

        use_early_stopping = self.early_stopping_rounds is not None and forced_iterations is None
        x_train, y_train, x_val, y_val = self._prepare_train_val_data(x, y, x_val, y_val)

        x_train_tensor = torch.tensor(x_train, dtype=torch.float32).to(self.device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32 if self.is_regression_model else torch.long).to(self.device)
        if self.is_regression_model or (self.task_type == TaskType.CLASSIFICATION and self.output_size == 1):
            y_train_tensor = y_train_tensor.unsqueeze(1)

        train_dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor), batch_size=self.batch_size, shuffle=True)

        x_val_tensor, y_val_tensor = None, None
        if x_val is not None and y_val is not None:
            x_val_tensor = torch.tensor(x_val, dtype=torch.float32).to(self.device)
            y_val_tensor = torch.tensor(y_val, dtype=torch.float32 if self.is_regression_model else torch.long).to(self.device)
            if self.is_regression_model or (self.task_type == TaskType.CLASSIFICATION and self.output_size == 1):
                y_val_tensor = y_val_tensor.unsqueeze(1)

        best_val_loss, patience_counter, best_epoch = float("inf"), 0, 0
        best_model_state = None
        val_loss_history = []
        n_epochs = forced_iterations if forced_iterations is not None else self.n_epochs

        for epoch in range(int(n_epochs)):
            self.model.train()
            for batch_x, batch_y in train_dataloader:
                self.optimizer.zero_grad()
                if self.lambda_optimizer:
                    self.lambda_optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.criterion(outputs[0] if isinstance(outputs, tuple) else outputs, batch_y)
                loss = self._calculate_regularization_loss(loss, self.model)
                loss.backward()
                self.optimizer.step()
                if self.lambda_optimizer:
                    self.lambda_optimizer.step()

            if use_early_stopping and x_val_tensor is not None:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(x_val_tensor)
                    val_loss = self.criterion(val_outputs[0] if isinstance(val_outputs, tuple) else val_outputs, y_val_tensor).item()
                val_loss_history.append(val_loss)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = self.model.state_dict()
                    best_epoch = epoch
                else:
                    patience_counter += 1
                if patience_counter >= self.early_stopping_rounds:
                    logger.info(f"Early stopping at epoch {best_epoch + 1}")
                    break
            else:
                best_epoch = epoch

        if best_model_state and use_early_stopping:
            self.model.load_state_dict(best_model_state)
        if self.is_regression_model and self.uncertainty_method == UncertaintyMethod.CONSTANT:
            y_pred_train = self.predict(x)
            self._train_residual_std = np.std(y - y_pred_train)
        return best_epoch + 1, val_loss_history

    def _evaluate_trial(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Evaluates a trial for hyperparameter optimization."""
        if self.is_regression_model:
            return np.sqrt(mean_squared_error(y_true, y_pred))
        return accuracy_score(y_true, np.round(y_pred))

    def _clone(self) -> "PyTorchModelBase":
        """Creates a new instance of the model with the same parameters."""
        return self.__class__(**self.get_params())

    def get_params(self) -> dict[str, Any]:
        """Gets the parameters of the model."""
        params = super().get_params()
        params.update(
            {
                "input_size": self.input_size,
                "output_size": self.output_size,
                "learning_rate": self.learning_rate,
                "n_epochs": self.n_epochs,
                "batch_size": self.batch_size,
                "task_type": self.task_type,
                "use_batch_norm": self.use_batch_norm,
                "uncertainty_method": self.uncertainty_method,
                "n_mc_dropout_samples": self.n_mc_dropout_samples,
                "dropout_rate": self.dropout_rate,
                "learn_regularization_lambdas": self.learn_regularization_lambdas,
                "learned_regularization_type": self.learned_regularization_type,
                "l1_lambda": self.l1_lambda,
                "l2_lambda": self.l2_lambda,
                "lambda_learning_rate": self.lambda_learning_rate,
                "random_seed": self.random_seed,
            }
        )
        return params

    def get_hyperparameter_search_space(self) -> dict[str, Any]:
        """Gets the hyperparameter search space for the model."""
        space = {
            "learning_rate": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True},
            "l1_lambda": {"type": "float", "low": 1e-8, "high": 1e-2, "log": True},
            "l2_lambda": {"type": "float", "low": 1e-8, "high": 1e-2, "log": True},
            "dropout_rate": {"type": "float", "low": 0.0, "high": 0.5, "step": 0.1},
        }
        if self.is_regression_model:
            space["n_mc_dropout_samples"] = {"type": "int", "low": 50, "high": 200, "step": 50}

        if self.search_space_override:
            space.update(self.search_space_override)
        return space

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Makes predictions on new data."""
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(x_tensor)
            if self.task_type == TaskType.CLASSIFICATION:
                predictions = (torch.sigmoid(outputs) > 0.5).cpu().numpy().astype(int) if self.output_size == 1 else torch.argmax(outputs, dim=1).cpu().numpy()
            else:
                predictions = outputs.cpu().numpy()
        return predictions

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Predicts class probabilities."""
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        if self.task_type != TaskType.CLASSIFICATION:
            raise ValueError("predict_proba is only available for classification tasks.")
        self.model.eval()
        x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = self.model(x_tensor)
            if self.output_size == 1:
                proba = torch.sigmoid(outputs).cpu().numpy().flatten()
                return np.vstack((1 - proba, proba)).T
            return torch.softmax(outputs, dim=1).cpu().numpy()

    def predict_uncertainty(self, x: np.ndarray) -> np.ndarray:
        """Estimates the uncertainty of predictions."""
        if not self.is_regression_model:
            raise ValueError("predict_uncertainty is only available for regression models.")
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        if self.uncertainty_method == UncertaintyMethod.CONSTANT:
            return np.full(x.shape[0], self._train_residual_std)
        raise ValueError(f"Unknown uncertainty_method: {self.uncertainty_method.value}")

    def evaluate(self, x: np.ndarray, y: np.ndarray, save_path: str = "metrics") -> np.ndarray:
        """Evaluates the model and saves metrics."""
        y_pred = self.predict(x)
        y_proba = self.predict_proba(x) if self.task_type == TaskType.CLASSIFICATION else None
        metrics_calculator = Metrics(task_type=self.task_type.value, model_name=self.name, x_data=x, y_true=y, y_pred=y_pred.flatten(), y_proba=y_proba)
        metrics_calculator.save_metrics(save_path)
        return y_pred

    def get_num_parameters(self) -> int:
        """Returns the number of trainable parameters in the model."""
        if self.model is None:
            return 0
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def get_internal_model(self) -> Any:
        """Returns the internal model."""
        return self.model

    def cross_validate(self, x: np.ndarray, y: np.ndarray, cv: int) -> dict[str, Any]:
        """Performs cross-validation."""
        self.cv_folds = cv
        self.fit(x, y)
        return {"test_score": self.cv_score_mean_}

    def get_classifier_predictions(self, x: np.ndarray, y_true_original: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Gets predictions from the internal classifier."""
        raise NotImplementedError("get_classifier_predictions is not implemented for this model type.")
