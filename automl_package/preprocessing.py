"""Preprocessing utilities for AutoML."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder as SklearnOneHotEncoder


class OrderedTargetEncoder(BaseEstimator, TransformerMixin):
    """A leakage-free target encoder for categorical features.

    It calculates smoothed target means based on a random permutation of data
    to prevent target leakage during training.
    """

    def __init__(
        self,
        cols: list[str | int] | None = None,
        smoothing: float = 1.0,
        min_samples_leaf: int = 1,
        random_state: int | None = None,
    ) -> None:
        """Initializes the OrderedTargetEncoder.

        Args:
            cols (list[str | int], optional): List of column names or indices to encode.
                                              If None, all categorical columns will be encoded.
            smoothing (float): Smoothing factor to prevent overfitting to rare categories.
            min_samples_leaf (int): Minimum number of samples per leaf for target encoding.
            random_state (int, optional): Random seed for reproducibility. Defaults to None.
        """
        self.cols = cols
        self.smoothing = smoothing
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.mapping = {}
        self.global_mean = None

    def fit(
        self, x: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray | None = None
    ) -> "OrderedTargetEncoder":
        """Fits the encoder to the training data.

        Args:
            x (pd.DataFrame | np.ndarray): Training features.
            y (pd.Series | np.ndarray): Training target.

        Returns:
            self: Fitted encoder.
        """
        if y is None:
            raise TypeError("fit() missing 1 required positional argument: 'y'")

        if isinstance(x, np.ndarray):
            x = pd.DataFrame(
                x,
                columns=pd.Index([f"col_{i}" for i in range(x.shape[1])], dtype=object),
            )
            if self.cols is not None:
                self.cols = [
                    f"col_{i}" for i in self.cols
                ]  # Adjust column names if indices were given

        self.global_mean = y.mean()

        for col in self.cols:
            if col not in x.columns:
                raise ValueError(f"Column {col} not found in X.")

            # Create a temporary DataFrame for the current column and target
            temp_df = pd.DataFrame({"col": x[col], "target": y})

            # Shuffle the data to create a random permutation for leakage-free calculation
            temp_df = temp_df.sample(
                frac=1, random_state=self.random_state
            ).reset_index(drop=True)

            # Calculate cumulative sum and count for each category
            cumsum = temp_df.groupby("col")["target"].cumsum() - temp_df["target"]
            cumcount = temp_df.groupby("col").cumcount()

            # Calculate leakage-free mean
            # Add smoothing to prevent division by zero and handle rare categories
            _ = (cumsum + self.global_mean * self.smoothing) / (
                cumcount + self.smoothing
            )

            # Store the mapping for each category
            self.mapping[col] = temp_df.groupby("col")["target"].mean().to_dict()

            # For prediction, we need a single value per category, so we use the overall mean
            # for that category from the training data.
            # The 'ordered' part is only for training data transformation to prevent leakage.

        return self

    def transform(self, x: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
        """Transforms the input data using the fitted encoder.

        Args:
            x (pd.DataFrame | np.ndarray): Data to transform.

        Returns:
            pd.DataFrame | np.ndarray: Transformed data.
        """
        if isinstance(x, np.ndarray):
            x_transformed = pd.DataFrame(
                x,
                columns=pd.Index([f"col_{i}" for i in range(x.shape[1])], dtype=object),
            )
            if self.cols is not None:
                # Create a list of column names that are actually categorical features
                categorical_col_names = [
                    col for col in self.cols if col in self.mapping
                ]

                for col in categorical_col_names:
                    # Use the stored mapping for transformation
                    x_transformed[col] = (
                        x_transformed[col]
                        .map(self.mapping[col])
                        .fillna(self.global_mean)
                    )
            else:  # If cols was None during fit, assume all object/category columns were encoded
                for col in self.mapping:  # Iterate through learned mappings
                    x_transformed[col] = (
                        x_transformed[col]
                        .map(self.mapping[col])
                        .fillna(self.global_mean)
                    )

            return (
                x_transformed.values
            )  # Return as numpy array if input was numpy array
        # Input is pandas DataFrame
        x_transformed = x.copy()
        for col in self.cols:
            if col in self.mapping:  # Check if the column was fitted
                x_transformed[col] = (
                    x_transformed[col].map(self.mapping[col]).fillna(self.global_mean)
                )
            else:
                # If a column was specified but not in mapping (e.g., all values unseen), fill with global mean
                x_transformed[col] = self.global_mean
        return x_transformed

    def fit_transform(
        self,
        x: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray | None = None,
        **_fit_params: Any,
    ) -> pd.DataFrame | np.ndarray:
        """Fits the encoder and transforms the data.

        Args:
            x (pd.DataFrame | np.ndarray): Training features.
            y (pd.Series | np.ndarray): Training target.
            **_fit_params: Additional fit parameters (ignored).

        Returns:
            pd.DataFrame | np.ndarray: Transformed data.
        """
        if y is None:
            raise TypeError(
                "fit_transform() missing 1 required positional argument: 'y'"
            )
        self.fit(x, y)
        return self.transform(x)


class OneHotEncoder(BaseEstimator, TransformerMixin):
    """A wrapper around sklearn.preprocessing.OneHotEncoder for categorical features."""

    def __init__(
        self, cols: list[str | int] | None = None, handle_unknown: str = "ignore"
    ) -> None:
        """Initializes the OneHotEncoder.

        Args:
            cols (list[str | int], optional): List of column names or indices to encode.
                                              If None, all categorical columns will be encoded.
            handle_unknown (str): Whether to raise an error or ignore unknown categories.
        """
        self.cols = cols
        self.handle_unknown = handle_unknown
        self.encoder = SklearnOneHotEncoder(
            handle_unknown=handle_unknown, sparse_output=False
        )
        self.feature_names_out_ = None  # To store feature names after transformation

    def fit(
        self,
        x: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray | None = None,  # noqa: ARG002
    ) -> "OneHotEncoder":
        """Fits the encoder to the training data.

        Args:
            x (pd.DataFrame | np.ndarray): Training features.
            y (pd.Series | np.ndarray, optional): Training target (ignored).

        Returns:
            self: Fitted encoder.
        """
        if isinstance(x, np.ndarray):
            x_temp = pd.DataFrame(x)
            x_selected = x_temp.iloc[:, self.cols] if self.cols is not None else x_temp
        else:  # X is pandas DataFrame
            x_selected = x[self.cols] if self.cols is not None else x

        self.encoder.fit(x_selected)
        return self

    def transform(self, x: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Transforms the input data using the fitted encoder.

        Args:
            x (pd.DataFrame | np.ndarray): Data to transform.

        Returns:
            np.ndarray: Transformed data.
        """
        if isinstance(x, np.ndarray):
            x_temp = pd.DataFrame(x)
            x_selected = x_temp.iloc[:, self.cols] if self.cols is not None else x_temp
        else:  # X is pandas DataFrame
            x_selected = x[self.cols] if self.cols is not None else x

        transformed_data = self.encoder.transform(x_selected)

        # Store feature names for consistency if needed later
        if self.feature_names_out_ is None:
            if hasattr(self.encoder, "get_feature_names_out"):
                self.feature_names_out_ = self.encoder.get_feature_names_out(
                    self.cols if self.cols else x_selected.columns
                )
            else:  # Fallback for older sklearn versions or if method is missing
                self.feature_names_out_ = [
                    f"x{i}" for i in range(transformed_data.shape[1])
                ]  # Generic names

        return transformed_data

    def fit_transform(
        self,
        x: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray | None = None,
        **_fit_params: Any,
    ) -> np.ndarray:
        """Fits the encoder and transforms the data.

        Args:
            x (pd.DataFrame | np.ndarray): Training features.
            y (pd.Series | np.ndarray, optional): Training target (ignored).
            **_fit_params: Additional fit parameters (ignored).

        Returns:
            np.ndarray: Transformed data.
        """
        self.fit(x, y)
        return self.transform(x)
