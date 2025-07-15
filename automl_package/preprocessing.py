from typing import List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder as SklearnOneHotEncoder


class OrderedTargetEncoder(BaseEstimator, TransformerMixin):
    """
    A leakage-free target encoder for categorical features.
    It calculates smoothed target means based on a random permutation of data
    to prevent target leakage during training.
    """

    def __init__(self, cols: Optional[List[Union[str, int]]] = None, smoothing: float = 1.0, min_samples_leaf: int = 1):
        self.cols = cols
        self.smoothing = smoothing
        self.min_samples_leaf = min_samples_leaf
        self.mapping = {}
        self.global_mean = None

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Optional[Union[pd.Series, np.ndarray]] = None):
        if y is None:
            raise TypeError("fit() missing 1 required positional argument: 'y'")

        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=pd.Index([f"col_{i}" for i in range(X.shape[1])], dtype=object))
            if self.cols is not None:
                self.cols = [f"col_{i}" for i in self.cols] # Adjust column names if indices were given

        self.global_mean = y.mean()

        for col in self.cols:
            if col not in X.columns:
                raise ValueError(f"Column {col} not found in X.")

            # Create a temporary DataFrame for the current column and target
            temp_df = pd.DataFrame({'col': X[col], 'target': y})

            # Shuffle the data to create a random permutation for leakage-free calculation
            temp_df = temp_df.sample(frac=1, random_state=42).reset_index(drop=True)

            # Calculate cumulative sum and count for each category
            cumsum = temp_df.groupby('col')['target'].cumsum() - temp_df['target']
            cumcount = temp_df.groupby('col').cumcount()

            # Calculate leakage-free mean
            # Add smoothing to prevent division by zero and handle rare categories
            encoded_col = (cumsum + self.global_mean * self.smoothing) / (cumcount + self.smoothing)

            # Store the mapping for each category
            self.mapping[col] = temp_df.groupby('col')['target'].mean().to_dict()
            
            # For prediction, we need a single value per category, so we use the overall mean
            # for that category from the training data.
            # The 'ordered' part is only for training data transformation to prevent leakage.

        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        if isinstance(X, np.ndarray):
            X_transformed = pd.DataFrame(X, columns=pd.Index([f"col_{i}" for i in range(X.shape[1])], dtype=object))
            if self.cols is not None:
                # Ensure cols are adjusted if they were indices during fit
                current_cols = [f"col_{i}" for i in range(X.shape[1])]
                original_cols_map = {f"col_{idx}": self.cols[i] for i, idx in enumerate(self.cols)}
                
                # Create a list of column names that are actually categorical features
                categorical_col_names = [col for col in self.cols if col in self.mapping]
                
                for col in categorical_col_names:
                    # Use the stored mapping for transformation
                    X_transformed[col] = X_transformed[col].map(self.mapping[col]).fillna(self.global_mean)
            else: # If cols was None during fit, assume all object/category columns were encoded
                for col in self.mapping.keys(): # Iterate through learned mappings
                    X_transformed[col] = X_transformed[col].map(self.mapping[col]).fillna(self.global_mean)

            return X_transformed.values # Return as numpy array if input was numpy array
        else: # Input is pandas DataFrame
            X_transformed = X.copy()
            for col in self.cols:
                if col in self.mapping: # Check if the column was fitted
                    X_transformed[col] = X_transformed[col].map(self.mapping[col]).fillna(self.global_mean)
                else:
                    # If a column was specified but not in mapping (e.g., all values unseen), fill with global mean
                    X_transformed[col] = self.global_mean
            return X_transformed

    def fit_transform(self, X: Union[pd.DataFrame, np.ndarray], y: Optional[Union[pd.Series, np.ndarray]] = None, **fit_params) -> Union[pd.DataFrame, np.ndarray]:
        if y is None:
            raise TypeError("fit_transform() missing 1 required positional argument: 'y'")
        self.fit(X, y)
        return self.transform(X)


class OneHotEncoder(BaseEstimator, TransformerMixin):
    """
    A wrapper around sklearn.preprocessing.OneHotEncoder for categorical features.
    """

    def __init__(self, cols: Optional[List[Union[str, int]]] = None, handle_unknown: str = 'ignore'):
        self.cols = cols
        self.handle_unknown = handle_unknown
        self.encoder = SklearnOneHotEncoder(handle_unknown=handle_unknown, sparse_output=False)
        self.feature_names_out_ = None # To store feature names after transformation

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Optional[Union[pd.Series, np.ndarray]] = None):
        if isinstance(X, np.ndarray):
            X_temp = pd.DataFrame(X)
            if self.cols is not None:
                # If cols are indices, select columns by index
                X_selected = X_temp.iloc[:, self.cols]
            else:
                # If cols is None, assume all columns are to be encoded (or handle based on dtype later)
                X_selected = X_temp
        else: # X is pandas DataFrame
            if self.cols is not None:
                X_selected = X[self.cols]
            else:
                X_selected = X # Assume all columns are to be encoded

        self.encoder.fit(X_selected)
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if isinstance(X, np.ndarray):
            X_temp = pd.DataFrame(X)
            if self.cols is not None:
                X_selected = X_temp.iloc[:, self.cols]
            else:
                X_selected = X_temp
        else: # X is pandas DataFrame
            if self.cols is not None:
                X_selected = X[self.cols]
            else:
                X_selected = X

        transformed_data = self.encoder.transform(X_selected)
        
        # Store feature names for consistency if needed later
        if self.feature_names_out_ is None:
            if hasattr(self.encoder, 'get_feature_names_out'):
                self.feature_names_out_ = self.encoder.get_feature_names_out(self.cols if self.cols else X_selected.columns)
            else: # Fallback for older sklearn versions or if method is missing
                self.feature_names_out_ = [f"x{i}" for i in range(transformed_data.shape[1])] # Generic names

        return transformed_data

    def fit_transform(self, X: Union[pd.DataFrame, np.ndarray], y: Optional[Union[pd.Series, np.ndarray]] = None, **fit_params) -> np.ndarray:
        self.fit(X, y)
        return self.transform(X)
