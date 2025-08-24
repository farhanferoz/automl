"""A mapper that uses a neural network to map class probabilities to a regression value."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from automl_package.enums import RegressionStrategy, UncertaintyMethod
from automl_package.logger import logger
from automl_package.models.common.regression_heads import (
    SeparateHeadsRegressionModule,
    SingleHeadFinalOutputRegressionModule,
    SingleHeadNOutputsRegressionModule,
)
from automl_package.models.mappers.base_mapper import BaseMapper


class NeuralNetworkMapper(BaseMapper):
    """A mapper that uses a neural network to map class probabilities to a regression value.

    This class encapsulates a PyTorch training loop to train the underlying regression head.
    """

    def __init__(
        self,
        n_classes: int,
        regression_strategy: RegressionStrategy,
        mapper_params: dict,
        early_stopping_rounds: int | None = None,
        validation_fraction: float | None = None,
    ) -> None:
        """Initializes the NeuralNetworkMapper.

        Args:
            n_classes: The number of classes.
            regression_strategy: The regression strategy to use.
            mapper_params: Parameters for the mapper, including training and model params.
            early_stopping_rounds: The number of epochs with no improvement after which training will be stopped.
            validation_fraction: The proportion of training data to set aside as validation data for early stopping.
        """
        self.n_classes = n_classes
        self.regression_strategy = regression_strategy
        self.mapper_params = mapper_params
        self.early_stopping_rounds = early_stopping_rounds
        self.validation_fraction = validation_fraction
        self.device = self.mapper_params.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.epochs = self.mapper_params.get("epochs", 100)
        self.batch_size = self.mapper_params.get("batch_size", 32)
        self.learning_rate = self.mapper_params.get("learning_rate", 1e-3)
        # For now, uncertainty is not handled by this mapper, so it's hardcoded.
        self.uncertainty_method = UncertaintyMethod.CONSTANT
        self.regression_output_size = 1

        regression_head_params = self.mapper_params.get("regression_head_params", {})

        if self.regression_strategy == RegressionStrategy.SEPARATE_HEADS:
            self.model = SeparateHeadsRegressionModule(
                n_classes=self.n_classes,
                regression_head_params=regression_head_params,
                uncertainty_method=self.uncertainty_method,
                regression_output_size=self.regression_output_size,
            )
        elif self.regression_strategy == RegressionStrategy.SINGLE_HEAD_N_OUTPUTS:
            self.model = SingleHeadNOutputsRegressionModule(
                input_size=self.n_classes,
                n_classes=self.n_classes,
                regression_head_params=regression_head_params,
                uncertainty_method=self.uncertainty_method,
                regression_output_size=self.regression_output_size,
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

    def _format_probas(self, probas: np.ndarray) -> np.ndarray:
        """Ensure probabilities are in the correct shape (N, n_classes) for binary case."""
        if self.n_classes == 2 and probas.ndim == 1:
            probas = np.vstack((1 - probas, probas)).T
        elif self.n_classes == 2 and probas.shape[1] == 1:
            probas = np.hstack((1 - probas, probas))
        return probas

    def fit(self, probas: np.ndarray, y_original: np.ndarray, train_indices: np.ndarray | None = None, val_indices: np.ndarray | None = None) -> None:
        """Overrides the BaseMapper.fit to handle 2D probability arrays directly.

        This bypasses the sorting logic in the parent class, which is not applicable here.
        """
        if probas.shape[0] != y_original.shape[0]:
            raise ValueError(f"Shape mismatch between probas ({probas.shape[0]}) and y_original ({y_original.shape[0]})")

        probas = self._format_probas(probas=probas)

        # Call _fit directly, bypassing the sorting in BaseMapper
        self._fit(probas, y_original, train_indices, val_indices)

    def _fit(self, probas: np.ndarray, y_original: np.ndarray, train_indices: np.ndarray | None = None, val_indices: np.ndarray | None = None) -> None:
        """Fits the neural network mapper. This involves a full training loop."""
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        loss_fn = nn.MSELoss()

        if self.early_stopping_rounds and train_indices is not None and val_indices is not None:
            logger.info("NNMapper: Using provided train and validation indices.")
            logger.info(f"NNMapper: Number of train indices: {len(train_indices)}")
            logger.info(f"NNMapper: Number of val indices: {len(val_indices)}")
            probas_train, y_train = probas[train_indices], y_original[train_indices]
            probas_val, y_val = probas[val_indices], y_original[val_indices]

            probas_val_tensor = torch.tensor(probas_val, dtype=torch.float32).to(self.device)
            y_val_tensor = torch.tensor(y_val, dtype=torch.float32).reshape(-1, 1).to(self.device)
        else:
            logger.info("NNMapper: Not using provided train and validation indices.")
            probas_train, y_train = probas, y_original
            probas_val_tensor, y_val_tensor = None, None

        # Create DataLoader
        probas_tensor = torch.tensor(probas_train, dtype=torch.float32)
        y_tensor = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
        dataset = TensorDataset(probas_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        best_val_loss = float("inf")
        epochs_no_improve = 0
        best_model_state = None

        for _epoch in range(self.epochs):
            self.model.train()
            for probas_batch_cpu, y_batch_cpu in loader:
                probas_batch = probas_batch_cpu.to(self.device)
                y_batch = y_batch_cpu.to(self.device)

                optimizer.zero_grad()
                predictions = self.model(probas_batch)
                loss = loss_fn(predictions, y_batch)
                loss.backward()
                optimizer.step()

            if probas_val_tensor is not None:
                self.model.eval()
                with torch.no_grad():
                    val_preds = self.model(probas_val_tensor)
                    val_loss = loss_fn(val_preds, y_val_tensor)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                    best_model_state = self.model.state_dict()
                else:
                    epochs_no_improve += 1

                if self.early_stopping_rounds and epochs_no_improve >= self.early_stopping_rounds:
                    logger.info(f"NNMapper: Early stopping at epoch {_epoch + 1}")
                    if best_model_state:
                        self.model.load_state_dict(best_model_state)
                    break
        if self.early_stopping_rounds and best_model_state:
            self.model.load_state_dict(best_model_state)

    def _fit_empty(self, probas: np.ndarray, y_original: np.ndarray) -> None:
        """Handles fitting on empty data."""

    def predict(self, probas_new: np.ndarray) -> np.ndarray:
        """Makes predictions with the trained neural network mapper."""
        probas_new = self._format_probas(probas=probas_new)

        self.model.eval()
        with torch.no_grad():
            probas_tensor = torch.tensor(probas_new, dtype=torch.float32).to(self.device)
            predictions_tensor = self.model(probas_tensor)
            return predictions_tensor.cpu().numpy().flatten()

    def predict_variance(self, probas_new: np.ndarray) -> np.ndarray:
        """Predicts variance. Not implemented for this mapper yet.

        Returns an array of zeros.
        """
        return np.zeros(probas_new.shape[0])

    def get_num_parameters(self) -> int:
        """Returns the total number of trainable parameters in the mapper's model."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
