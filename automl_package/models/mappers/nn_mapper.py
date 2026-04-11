"""A mapper that uses a neural network to map class probabilities to a regression value."""

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from automl_package.enums import BoundaryRegularizationMethod, LearnedRegularizationType, OptimizerType, RegressionStrategy, UncertaintyMethod
from automl_package.logger import logger
from automl_package.models.common.mixins import BoundaryLossMixin, RegularizationMixin
from automl_package.models.common.monotonicity_config_mixin import MonotonicityConfigMixin
from automl_package.models.common.penalties import apply_additional_penalties
from automl_package.models.common.regression_heads import SeparateHeadsRegressionModule, SingleHeadFinalOutputRegressionModule, SingleHeadNOutputsRegressionModule
from automl_package.models.common.training import train_model
from automl_package.models.mappers.base_mapper import BaseMapper
from automl_package.optimizers import get_optimizer_wrapper
from automl_package.utils.losses import nll_loss
from automl_package.utils.numerics import create_bins
from automl_package.utils.pytorch_utils import calculate_regularization_loss, get_device


class NeuralNetworkMapper(BaseMapper, RegularizationMixin, BoundaryLossMixin, MonotonicityConfigMixin):
    """A mapper that uses a neural network to map class probabilities to a regression value.

    This class encapsulates a PyTorch training loop to train the underlying regression head.
    """

    def __init__(
        self,
        n_classes: int,
        regression_strategy: RegressionStrategy,
        mapper_params: dict,
        uncertainty_method: UncertaintyMethod = UncertaintyMethod.CONSTANT,
        early_stopping_rounds: int | None = None,
        boundary_regularization_method: BoundaryRegularizationMethod = BoundaryRegularizationMethod.NONE,
        boundary_loss_weight: float = 1.0,
        class_value_ranges: torch.Tensor | None = None,
        class_boundaries: np.ndarray | None = None,
        optimizer_type: OptimizerType = OptimizerType.ADAM,
        l1_lambda: float = 0.0,
        l2_lambda: float = 0.0,
        learn_regularization_lambdas: bool = False,
        learned_regularization_type: LearnedRegularizationType = LearnedRegularizationType.L1_L2,
        lambda_learning_rate: float = 1e-5,
        apply_boundary_loss_during_validation: bool = False,
        use_middle_class_nll_penalty: bool = False,
        use_monotonic_constraints: bool = False,
        constrain_middle_class: bool = False,
    ) -> None:
        """Initializes the NeuralNetworkMapper.

        Args:
            n_classes: The number of classes.
            regression_strategy: The regression strategy to use.
            mapper_params: Parameters for the mapper, including training and model params.
            uncertainty_method: The uncertainty estimation method.
            early_stopping_rounds: The number of epochs with no improvement after which training will be stopped.
            boundary_regularization_method: The method to use for boundary regularization.
            boundary_loss_weight: The weight of the boundary loss.
            class_value_ranges: The value ranges for each class.
            class_boundaries: The boundaries for each class.
            optimizer_type: The type of optimizer to use.
            l1_lambda: L1 regularization strength.
            l2_lambda: L2 regularization strength.
            learn_regularization_lambdas: Whether to learn the regularization lambdas.
            learned_regularization_type: The type of learned regularization to use.
            lambda_learning_rate: The learning rate for the lambda optimizer.
            apply_boundary_loss_during_validation: Whether to apply boundary loss during validation.
            use_middle_class_nll_penalty: Whether to use a normal distribution penalty for the middle class.
            use_monotonic_constraints: Whether to enforce monotonicity constraints in the regression heads.
            constrain_middle_class: Whether to apply special constraints to the middle class.
        """
        super().__init__(uncertainty_method=uncertainty_method)
        self._bypass_sorting = True
        self.n_classes = n_classes
        self.regression_strategy = regression_strategy
        self.mapper_params = mapper_params
        self.uncertainty_method = uncertainty_method
        self.early_stopping_rounds = early_stopping_rounds
        self.boundary_regularization_method = boundary_regularization_method
        self.boundary_loss_weight = boundary_loss_weight
        self.class_value_ranges = class_value_ranges
        self.class_boundaries = class_boundaries
        self.optimizer_type = optimizer_type
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.learn_regularization_lambdas = learn_regularization_lambdas
        self.learned_regularization_type = learned_regularization_type
        self.lambda_learning_rate = lambda_learning_rate
        self.apply_boundary_loss_during_validation = apply_boundary_loss_during_validation
        self.use_middle_class_nll_penalty = use_middle_class_nll_penalty
        self.use_monotonic_constraints = use_monotonic_constraints
        self.constrain_middle_class = constrain_middle_class
        self.middle_class_dist_params = self.mapper_params.get("middle_class_dist_params")

        if isinstance(self.learned_regularization_type, str):
            self.learned_regularization_type = LearnedRegularizationType[self.learned_regularization_type.upper()]

        self._setup_regularization_parameters()

        self.device = self.mapper_params.get("device", get_device())
        self.epochs = self.mapper_params.get("epochs", 100)
        self.batch_size = self.mapper_params.get("batch_size", 32)
        self.learning_rate = self.mapper_params.get("learning_rate", 1e-3)
        self.regression_output_size = 2 if self.uncertainty_method == UncertaintyMethod.PROBABILISTIC else 1
        self._train_residual_std = [] if self.regression_strategy in [RegressionStrategy.SEPARATE_HEADS, RegressionStrategy.SINGLE_HEAD_N_OUTPUTS] else 0.0
        self.optimizer_wrapper = get_optimizer_wrapper(self.optimizer_type)

        regression_head_params = self.mapper_params.get("regression_head_params", {})

        if self.regression_strategy == RegressionStrategy.SEPARATE_HEADS:
            self.model = SeparateHeadsRegressionModule(
                n_classes=self.n_classes,
                regression_head_params=regression_head_params,
                uncertainty_method=self.uncertainty_method,
                regression_output_size=self.regression_output_size,
                use_monotonic_constraints=self.use_monotonic_constraints,
                constrain_middle_class=self.constrain_middle_class,
            )
        elif self.regression_strategy == RegressionStrategy.SINGLE_HEAD_N_OUTPUTS:
            self.model = SingleHeadNOutputsRegressionModule(
                input_size=self.n_classes,
                n_classes=self.n_classes,
                regression_head_params=regression_head_params,
                uncertainty_method=self.uncertainty_method,
                regression_output_size=self.regression_output_size,
                constrain_middle_class=self.constrain_middle_class,
            )
        elif self.regression_strategy == RegressionStrategy.SINGLE_HEAD_FINAL_OUTPUT:
            self.model = SingleHeadFinalOutputRegressionModule(
                input_size=self.n_classes,
                regression_head_params=regression_head_params,
                uncertainty_method=self.uncertainty_method,
                regression_output_size=self.regression_output_size,
            )
        else:
            raise ValueError(f"Unknown regression_strategy: {self.regression_strategy}")

        self.model.to(self.device)

    @property
    def regression_heads(self):
        """Returns the regression heads module list."""
        return self.model.heads

    def _format_probas(self, probas: np.ndarray) -> np.ndarray:
        """Ensure probabilities are in the correct shape (N, n_classes) for binary case."""
        if self.n_classes == 2 and probas.ndim == 1:
            probas = np.vstack((1 - probas, probas)).T
        elif self.n_classes == 2 and probas.shape[1] == 1:
            probas = np.hstack((1 - probas, probas))
        return probas

    def _calculate_main_regression_loss(self, final_pred: torch.Tensor, y_true_squeezed: torch.Tensor) -> torch.Tensor:
        """Calculates the main regression loss (NLL or MSE) based on the uncertainty method."""
        if self.uncertainty_method == UncertaintyMethod.PROBABILISTIC:
            mean = final_pred[:, 0]
            log_var = final_pred[:, 1]
            loss = nll_loss(torch.stack((mean, log_var), dim=1), y_true_squeezed)
        else:
            loss = nn.MSELoss()(final_pred.squeeze(), y_true_squeezed)
        return loss

    def _calculate_multi_output_loss(self, predictions: tuple[torch.Tensor, torch.Tensor], y_true: torch.Tensor, include_boundary_loss: bool = True) -> torch.Tensor:
        """Calculates loss for strategies that return multiple outputs (final_pred, per_head_outputs)."""
        y_true_squeezed = y_true.squeeze(-1) if y_true.ndim > 1 else y_true
        final_pred, per_head_outputs = predictions
        total_loss = self._calculate_main_regression_loss(final_pred, y_true_squeezed)

        return apply_additional_penalties(
            total_loss=total_loss,
            per_head_outputs=per_head_outputs,
            y_true_squeezed=y_true_squeezed,
            model_instance=self,
            include_boundary_loss=include_boundary_loss,
            class_boundaries=self.class_boundaries,
            class_value_ranges=self.class_value_ranges,
            middle_class_dist_params=self.middle_class_dist_params,
        )

    def _calculate_single_output_loss(self, predictions: torch.Tensor, y_true: torch.Tensor, **_kwargs: Any) -> torch.Tensor:
        """Calculates loss for strategies that return a single final prediction tensor."""
        y_true_squeezed = y_true.squeeze(-1) if y_true.ndim > 1 else y_true
        return self._calculate_main_regression_loss(predictions, y_true_squeezed)

    def _fit(self, probas: np.ndarray, y_original: np.ndarray, **kwargs: Any) -> dict[str, Any]:
        """Fits the neural network mapper. This involves a full training loop."""
        train_indices = kwargs.get("train_indices")
        val_indices = kwargs.get("val_indices")

        probas = self._format_probas(probas=probas)

        self.model.train()
        optimizer = self.optimizer_wrapper.create_optimizer(self.model.parameters(), lr=self.learning_rate)
        self._setup_lambda_optimizer()

        def regularization_fn(loss: torch.Tensor, model: nn.Module) -> torch.Tensor:
            return calculate_regularization_loss(
                base_loss=loss,
                model=model,
                learn_regularization=self.learn_regularization_lambdas,
                learned_regularization_type=self.learned_regularization_type,
                l1_lambda=self.l1_lambda,
                l2_lambda=self.l2_lambda,
                l1_log_lambda=self.l1_log_lambda,
                l2_log_lambda=self.l2_log_lambda,
            )

        use_early_stopping = self.early_stopping_rounds and self.early_stopping_rounds > 0 and train_indices is not None and val_indices is not None

        if use_early_stopping:
            logger.info("NNMapper: Using provided train and validation indices for early stopping.")
            probas_train, y_train = probas[train_indices], y_original[train_indices]
            probas_val, y_val = probas[val_indices], y_original[val_indices]

            probas_val_tensor = torch.tensor(probas_val, dtype=torch.float32).to(self.device)
            y_val_tensor = torch.tensor(y_val, dtype=torch.float32).reshape(-1, 1).to(self.device)
        else:
            logger.info("NNMapper: Not using early stopping. Training on all provided data.")
            probas_train, y_train = probas, y_original
            probas_val_tensor, y_val_tensor = None, None

        probas_tensor = torch.tensor(probas_train, dtype=torch.float32)
        y_tensor = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)

        # Select the appropriate loss function and forward pass arguments based on the strategy
        if self.regression_strategy == RegressionStrategy.SINGLE_HEAD_FINAL_OUTPUT:
            loss_fn = self._calculate_single_output_loss
            forward_pass_kwargs = {}
        else:
            loss_fn = self._calculate_multi_output_loss
            forward_pass_kwargs = {"return_head_outputs": True}

        if self.boundary_regularization_method == BoundaryRegularizationMethod.HARDSIGMOID:
            if self.class_value_ranges is None:
                raise ValueError("class_value_ranges must be provided for the HARDSIGMOID boundary method.")

            _, y_binned_tensor = create_bins(data=y_train, unique_bin_edges=self.class_boundaries)
            y_binned_tensor = torch.tensor(y_binned_tensor, dtype=torch.long, device=self.device)
            forward_pass_kwargs["y_binned_tensor"] = y_binned_tensor
            forward_pass_kwargs["class_value_ranges"] = self.class_value_ranges

        best_epoch, _ = train_model(
            model=self.model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            probas_train=probas_tensor,
            y_train=y_tensor,
            probas_val=probas_val_tensor,
            y_val=y_val_tensor,
            epochs=self.epochs,
            batch_size=self.batch_size,
            early_stopping_rounds=self.early_stopping_rounds,
            device=self.device,
            optimizer_wrapper=self.optimizer_wrapper,
            regularization_fn=regularization_fn,
            lambda_optimizer=self.lambda_optimizer,
            forward_pass_kwargs=forward_pass_kwargs,
            apply_boundary_loss_during_validation=self.apply_boundary_loss_during_validation,
            y_binned_train=forward_pass_kwargs.get("y_binned_tensor"),
        )

        if self.uncertainty_method == UncertaintyMethod.CONSTANT:
            if self.regression_strategy == RegressionStrategy.SEPARATE_HEADS:
                for i in range(self.n_classes):
                    y_pred_train_head = self.model.heads[i](torch.tensor(probas_train[:, i].reshape(-1, 1), dtype=torch.float32).to(self.device)).cpu().detach().numpy()
                    self._train_residual_std.append(np.std(y_train - y_pred_train_head))
            elif self.regression_strategy == RegressionStrategy.SINGLE_HEAD_N_OUTPUTS:
                y_pred_train_per_class = self.model.forward_per_class(torch.tensor(probas_train, dtype=torch.float32).to(self.device)).cpu().detach().numpy()
                for i in range(self.n_classes):
                    self._train_residual_std.append(np.std(y_train - y_pred_train_per_class[:, i, 0]))
            elif self.regression_strategy == RegressionStrategy.SINGLE_HEAD_FINAL_OUTPUT:
                y_pred_train = self._predict(probas_train)
                self._train_residual_std = np.std(y_train - y_pred_train)
            else:
                raise ValueError(f"Regression strategy {self.regression_strategy.value} is not supported.")

        return {"epochs_used": best_epoch}

    def _fit_empty(self) -> None:
        """Handles fitting on empty data."""

    def _predict(self, probas_new: np.ndarray) -> np.ndarray:
        """Makes predictions with the trained neural network mapper."""
        probas_new = self._format_probas(probas=probas_new)

        self.model.eval()
        with torch.no_grad():
            probas_tensor = torch.tensor(probas_new, dtype=torch.float32).to(self.device)
            predictions_tensor = self.model(probas_tensor, return_head_outputs=False)
            if isinstance(predictions_tensor, tuple):
                predictions_tensor = predictions_tensor[0]
            # Return only the mean if uncertainty_method == PROBABILISTIC
            return predictions_tensor[:, 0].cpu().numpy() if self.uncertainty_method == UncertaintyMethod.PROBABILISTIC else predictions_tensor.cpu().numpy()

    def _predict_variance(self, probas_new: np.ndarray) -> np.ndarray:
        """Predicts variance. For probabilistic models, this is the learned variance."""
        if self.uncertainty_method == UncertaintyMethod.PROBABILISTIC:
            probas_new = self._format_probas(probas=probas_new)
            self.model.eval()
            with torch.no_grad():
                probas_tensor = torch.tensor(probas_new, dtype=torch.float32).to(self.device)
                predictions_tensor = self.model(probas_tensor)
                log_var = predictions_tensor[:, 1].cpu().numpy()
                variances = np.exp(log_var)
        else:
            assert self.uncertainty_method == UncertaintyMethod.CONSTANT
            # For CONSTANT uncertainty, return the squared residual std
            variances = np.full(probas_new.shape[0], self._train_residual_std**2)
        return variances

    def predict_mean_and_variance_per_class(self, probas_new: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predicts mean and variance for each class separately.

        Args:
            probas_new: A numpy array of shape (n_samples, n_classes) containing the class probabilities.

        Returns:
            A tuple containing two numpy arrays:
            - The first array contains the mean predictions for each class.
            - The second array contains the variance predictions for each class.
        """
        probas_new = self._format_probas(probas=probas_new)
        self.model.eval()
        with torch.no_grad():
            probas_tensor = torch.tensor(probas_new, dtype=torch.float32).to(self.device)
            predictions_per_class = self.model.forward_per_class(probas_tensor)
            mean_per_class = predictions_per_class[:, :, 0].cpu().numpy()
            if self.uncertainty_method == UncertaintyMethod.PROBABILISTIC:
                log_variance_per_class = predictions_per_class[:, :, 1].cpu().numpy()
                variance_per_class = np.exp(log_variance_per_class)
            elif self.uncertainty_method == UncertaintyMethod.CONSTANT:
                if self.regression_strategy in [RegressionStrategy.SEPARATE_HEADS, RegressionStrategy.SINGLE_HEAD_N_OUTPUTS]:
                    variances = np.array([self._train_residual_std[i] ** 2 for i in range(self.n_classes)])
                    variance_per_class = np.tile(variances, (mean_per_class.shape[0], 1))
                else:
                    variance_per_class = np.full_like(mean_per_class, self._train_residual_std**2)
            else:
                variance_per_class = np.zeros_like(mean_per_class)
            return mean_per_class, variance_per_class

    def get_num_parameters(self) -> int:
        """Returns the total number of trainable parameters in the mapper's model."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
