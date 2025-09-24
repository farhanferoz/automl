"""Base classes for PyTorch models."""

from abc import ABC, abstractmethod
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from automl_package.enums import LearnedRegularizationType, OptimizerType, TaskType, UncertaintyMethod
from automl_package.logger import logger
from automl_package.models.base import BaseModel
from automl_package.models.common.mixins import RegularizationMixin
from automl_package.optimizers import get_optimizer_wrapper
from automl_package.utils.numerics import aggregate_stats, ensure_proba_shape, log_erfc
from automl_package.utils.pytorch_utils import calculate_regularization_loss, get_device


class PyTorchModelBase(BaseModel, RegularizationMixin, ABC):
    """Base class for PyTorch-based neural network models."""

    _defaults: ClassVar[dict[str, Any]] = {
        "input_size": None,
        "output_size": 1,
        "learning_rate": 0.001,
        "n_epochs": 50,
        "batch_size": 32,
        "use_batch_norm": False,
        "n_mc_dropout_samples": 100,
        "dropout_rate": 0.0,
        "learn_regularization_lambdas": False,
        "learned_regularization_type": LearnedRegularizationType.L1_L2,
        "l1_lambda": 0.0,
        "l2_lambda": 0.0,
        "lambda_learning_rate": 1e-5,
        "random_seed": None,
        "optimizer_type": OptimizerType.ADAM,
    }

    def __init__(self, **kwargs: Any) -> None:
        """Initializes the PyTorchModelBase."""
        # Apply defaults from this class specifically
        for key, value in PyTorchModelBase._defaults.items():
            kwargs.setdefault(key, value)

        # Set all attributes on the instance
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Call the parent constructor
        super().__init__(**kwargs)

        # Now, override any attributes set by the parent's constructor if necessary
        self.l1_log_lambda, self.l2_log_lambda = None, None
        self.using_l1_regularization = (
            self.learned_regularization_type in [LearnedRegularizationType.L1_ONLY, LearnedRegularizationType.L1_L2] if self.learn_regularization_lambdas else self.l1_lambda > 0
        )
        self.using_l2_regularization = (
            self.learned_regularization_type in [LearnedRegularizationType.L2_ONLY, LearnedRegularizationType.L1_L2] if self.learn_regularization_lambdas else self.l2_lambda > 0
        )
        self.model: nn.Module | None = None
        self.criterion: nn.Module | None = None
        self.optimizer: optim.Optimizer | None = None
        self.lambda_optimizer: optim.Optimizer | None = None
        self.device: torch.device | None = get_device()
        self._train_residual_std = 0.0
        self.optimizer_wrapper = get_optimizer_wrapper(self.optimizer_type)

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
        self.optimizer = self.optimizer_wrapper.create_optimizer(model.parameters(), lr=self.learning_rate)
        self._setup_lambda_optimizer()

    def _calculate_regularization_loss(self, base_loss: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
        """Calculates the regularization loss by calling the utility function."""
        return calculate_regularization_loss(
            base_loss=base_loss,
            model=model,
            learn_regularization=self.learn_regularization_lambdas,
            learned_regularization_type=self.learned_regularization_type,
            l1_lambda=self.l1_lambda,
            l2_lambda=self.l2_lambda,
            l1_log_lambda=self.l1_log_lambda,
            l2_log_lambda=self.l2_log_lambda,
        )

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
        new_input_size = 1 if x_train.ndim == 1 else x_train.shape[1]
        if self.input_size != new_input_size:
            self.input_size = new_input_size
            self.build_model()

        if self.model is None:
            self.input_size = new_input_size
            self.build_model()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        if self.random_seed is not None:
            torch.manual_seed(self.random_seed)
            np.random.seed(self.random_seed)
        self.build_model()
        self._setup_regularization_parameters()
        self._setup_optimizers(self.model)

        use_early_stopping = self.early_stopping_rounds is not None and forced_iterations is None

        # Convert to numpy arrays
        x_train = np.array(x_train, dtype=np.float32)
        y_train = np.array(y_train, dtype=np.float32 if self.is_regression_model else np.int64)
        if x_val is not None:
            x_val = np.array(x_val, dtype=np.float32)
        if y_val is not None:
            y_val = np.array(y_val, dtype=np.float32 if self.is_regression_model else np.int64)

        is_binary_classification = self.task_type == TaskType.CLASSIFICATION and self.output_size == 1

        x_train_tensor = torch.tensor(x_train, dtype=torch.float32).to(self.device)
        y_train_tensor = torch.tensor(y_train, dtype=(torch.float32 if self.is_regression_model or is_binary_classification else torch.long)).to(self.device)
        if self.is_regression_model or is_binary_classification:
            y_train_tensor = y_train_tensor.unsqueeze(1)

        train_dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor), batch_size=self.batch_size, shuffle=True)

        x_val_tensor, y_val_tensor = None, None
        if x_val is not None and y_val is not None:
            x_val_tensor = torch.tensor(x_val, dtype=torch.float32).to(self.device)
            y_val_tensor = torch.tensor(y_val, dtype=(torch.float32 if self.is_regression_model or is_binary_classification else torch.long)).to(self.device)
            if self.is_regression_model or is_binary_classification:
                y_val_tensor = y_val_tensor.unsqueeze(1)

        best_val_loss, patience_counter, best_epoch = float("inf"), 0, 0
        best_model_state = None
        val_loss_history = []
        n_epochs = forced_iterations if forced_iterations is not None else self.n_epochs

        for epoch in range(int(n_epochs)):
            self.model.train()
            for batch_x, batch_y in train_dataloader:
                if self.lambda_optimizer:
                    self.lambda_optimizer.zero_grad()

                self.optimizer_wrapper.step(
                    model=self.model, loss_fn=self.criterion, regularization_fn=self._calculate_regularization_loss, optimizer=self.optimizer, batch_x=batch_x, batch_y=batch_y
                )

                if self.lambda_optimizer:
                    self.lambda_optimizer.step()
                self._after_step()

            if use_early_stopping and x_val_tensor is not None:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(x_val_tensor)
                    val_loss = self.criterion(val_outputs, y_val_tensor).item()
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
            y_pred_train = self.predict(x_train, filter_data=False)
            self._train_residual_std = np.std(y_train - y_pred_train)
        return best_epoch + 1, val_loss_history

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
                "optimizer_type": self.optimizer_type,
            }
        )
        return params

    def get_hyperparameter_search_space(self) -> dict[str, Any]:
        """Gets the hyperparameter search space for the model."""
        space = {
            "learning_rate": {"type": "float", "low": 1e-4, "high": 1e-2, "log": False},
            "l1_lambda": {"type": "float", "low": 1e-8, "high": 1e-2, "log": False},
            "l2_lambda": {"type": "float", "low": 1e-8, "high": 1e-2, "log": False},
            "dropout_rate": {"type": "float", "low": 0.0, "high": 0.5, "step": 0.1},
        }
        if self.is_regression_model:
            space["n_mc_dropout_samples"] = {"type": "int", "low": 50, "high": 200, "step": 50}

        if self.search_space_override:
            space.update(self.search_space_override)
        return space

    def predict(self, x: np.ndarray | pd.DataFrame, filter_data: bool = True) -> np.ndarray:
        """Makes predictions on new data."""
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        if filter_data:
            x = self._filter_predict_data(x)
        if isinstance(x, pd.DataFrame):
            x = x.values
        x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(x_tensor)
            model_output = outputs[0] if isinstance(outputs, tuple) else outputs

            if self.task_type == TaskType.CLASSIFICATION:
                predictions = (torch.sigmoid(model_output) > 0.5).cpu().numpy().astype(int) if self.output_size == 1 else torch.argmax(model_output, dim=1).cpu().numpy()
            else:  # Regression case
                predictions = (model_output[:, 0] if self.uncertainty_method == UncertaintyMethod.PROBABILISTIC else model_output).cpu().numpy()
        return predictions

    def predict_proba(self, x: np.ndarray | pd.DataFrame, filter_data: bool = True) -> np.ndarray:
        """Predicts class probabilities."""
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        if self.task_type != TaskType.CLASSIFICATION:
            raise ValueError("predict_proba is only available for classification tasks.")
        if filter_data:
            x = self._filter_predict_data(x)
        self.model.eval()
        if isinstance(x, pd.DataFrame):
            x = x.values
        x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = self.model(x_tensor)
            model_output = outputs[0] if isinstance(outputs, tuple) else outputs

            if self.output_size == 1:
                proba = torch.sigmoid(model_output).cpu().numpy()
                n_classes = 2
            else:
                proba = torch.softmax(model_output, dim=1).cpu().numpy()
                n_classes = self.output_size
            return ensure_proba_shape(proba, n_classes)

    def predict_uncertainty(self, x: np.ndarray, filter_data: bool = True) -> np.ndarray:
        """Estimates the uncertainty of predictions."""
        if not self.is_regression_model:
            raise ValueError("predict_uncertainty is only available for regression models.")

        if self.uncertainty_method == UncertaintyMethod.BINNED_RESIDUAL_STD:
            return super().predict_uncertainty(x, filter_data=filter_data)

        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")

        if filter_data:
            x = self._filter_predict_data(x)

        if isinstance(x, pd.DataFrame):
            x = x.values
        x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)

        if self.uncertainty_method == UncertaintyMethod.CONSTANT:
            uncertainty_std = np.full(x.shape[0], self._train_residual_std)
        elif self.uncertainty_method == UncertaintyMethod.MC_DROPOUT:
            self.model.train()  # Activate dropout
            predictions = []
            for _ in range(self.n_mc_dropout_samples):
                with torch.no_grad():
                    outputs = self.model(x_tensor)
                    model_output = outputs[0] if isinstance(outputs, tuple) else outputs
                    predictions.append(model_output.cpu().numpy())
            uncertainty_std = np.std(np.array(predictions), axis=0)
        elif self.uncertainty_method == UncertaintyMethod.PROBABILISTIC:
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(x_tensor)
                model_output = outputs[0] if isinstance(outputs, tuple) else outputs
                # The second column is log_variance, so we exponentiate it to get the variance
                log_variance = model_output[:, 1].cpu().numpy()
                variance = np.exp(log_variance)
                uncertainty_std = np.sqrt(variance)
        else:
            raise ValueError(f"Unknown uncertainty_method: {self.uncertainty_method.value}")
        return uncertainty_std

    def get_num_parameters(self) -> int:
        """Returns the number of trainable parameters in the model."""
        return 0 if self.model is None else sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def get_internal_model(self) -> Any:
        """Returns the internal model."""
        return self.model

    def cross_validate(self, x: np.ndarray, y: np.ndarray, cv: int) -> dict[str, Any]:
        """Performs cross-validation."""
        self.cv_folds = cv
        self.fit(x, y)
        return {"test_score": self.cv_score_mean_}

    def _after_step(self) -> None:
        """A hook that is called after each optimizer step."""

    def get_classifier_predictions(self, x: np.ndarray | pd.DataFrame, y_true_original: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Gets predictions from the internal classifier."""
        raise NotImplementedError("get_classifier_predictions is not implemented for this model type.")
