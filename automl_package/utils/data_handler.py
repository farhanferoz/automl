"""Data handling utilities."""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DataHandler:
    """Handles data scaling and splitting."""

    def __init__(self, scale_x: bool = True, scale_y: bool = True) -> None:
        """Initializes the DataHandler."""
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.x_scaler = StandardScaler() if scale_x else None
        self.y_scaler = StandardScaler() if scale_y else None

    def fit_transform(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Fits and transforms the data."""
        x_scaled = self.x_scaler.fit_transform(x) if self.scale_x else x
        y_scaled = self.y_scaler.fit_transform(y.reshape(-1, 1)).flatten() if self.scale_y else y
        return x_scaled, y_scaled

    def inverse_transform_y(self, y_pred_scaled: np.ndarray) -> np.ndarray:
        """Inverse transforms the scaled y predictions."""
        if not self.scale_y:
            return y_pred_scaled
        return self.y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()


def create_train_val_split(x: np.ndarray, y: np.ndarray, validation_fraction: float, random_state: int | None) -> tuple[np.ndarray, np.ndarray]:
    """Creates a train-validation split and returns the indices."""
    train_indices = np.arange(x.shape[0])
    val_indices = np.array([])
    if validation_fraction > 0:
        indices = np.arange(x.shape[0])
        try:
            train_indices, val_indices = train_test_split(indices, test_size=validation_fraction, random_state=random_state, stratify=y if y is not None else None)
        except ValueError:
            train_indices, val_indices = train_test_split(indices, test_size=validation_fraction, random_state=random_state, stratify=None)
    return train_indices, val_indices
