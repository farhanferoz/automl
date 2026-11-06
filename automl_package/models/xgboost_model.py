"""XGBoost model wrapper for AutoML."""

from typing import Any, ClassVar, Never

import numpy as np
import xgboost as xgb

from automl_package.enums import CategoricalCast, ExplainerType, Metric, TaskType, UncertaintyMethod
from automl_package.logger import logger
from automl_package.models.base import BaseModel
from automl_package.models.common.column_roles import ColumnRolesMixin
from automl_package.models.common.common import get_loss_history
from automl_package.utils.numerics import ensure_proba_shape


class XGBoostModel(ColumnRolesMixin, BaseModel):
    """XGBoost model wrapper.

    Note: Methods in this class are intentionally named identically to those in BaseModel
    as they implement the BaseModel interface. This is not a redeclaration error.

    ``offset_feature`` is handed to XGBoost as ``base_margin`` (``base_margin_eval_set`` for the validation set, ``base_margin`` again in
    ``predict``, which includes it). ``categorical_features`` become pandas ``Categorical`` columns with the levels fixed at fit
    (``enable_categorical=True``). See `ColumnRolesMixin`.
    """

    _supports_sample_weight: ClassVar[bool] = True  # native row weights, on the early-stopping set too

    _CATEGORICAL_CAST = CategoricalCast.PANDAS_CATEGORICAL

    def __init__(self, random_seed: int | None = None, **kwargs: Any) -> None:
        """Initializes the XGBoostModel.

        Args:
            random_seed (int, optional): Random seed for reproducibility.
            task_type (TaskType): The type of task (regression or classification).
            **kwargs: Additional keyword arguments for the BaseModel.
        """
        super().__init__(**kwargs)
        self.random_seed = random_seed
        self.model: xgb.XGBRegressor | xgb.XGBClassifier | None = None
        self._train_residual_std = 0.0
        self.num_iterations_used = 0
        self.params.setdefault("verbosity", 0)

        if self.task_type == TaskType.REGRESSION:
            if self.uncertainty_method == UncertaintyMethod.PROBABILISTIC:
                logger.warning(
                    f"The {UncertaintyMethod.PROBABILISTIC.name} uncertainty method is not supported by the scikit-learn API of XGBoost. "
                    f"Falling back to {UncertaintyMethod.BINNED_RESIDUAL_STD.name} uncertainty method."
                )
                self.uncertainty_method = UncertaintyMethod.BINNED_RESIDUAL_STD
            self.objective = "reg:squarederror"
            self.eval_metric = Metric.RMSE.label
        elif self.task_type == TaskType.CLASSIFICATION:
            # Objective and metric will be set dynamically in _fit_single
            self.objective = None
            self.eval_metric = None

    @property
    def name(self) -> str:
        """Returns the name of the model."""
        return "XGBoostModel"

    def _fit_single(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        forced_iterations: int | None = None,
        sample_weight: np.ndarray | None = None,
        sample_weight_val: np.ndarray | None = None,
    ) -> tuple[int, list[float]]:
        """Fits a single model instance.

        Args:
            x_train (np.ndarray): The training features.
            y_train (np.ndarray): The training targets.
            x_val (np.ndarray | None): The validation features.
            y_val (np.ndarray | None): The validation targets.
            forced_iterations (int | None): If provided, train for this many iterations, ignoring early stopping.
            sample_weight (np.ndarray | None): Row weights of the training part (XGBoost ``sample_weight``).
            sample_weight_val (np.ndarray | None): Row weights of the early-stopping set (``sample_weight_eval_set``).

        Returns:
            tuple[int, list[float]]: A tuple containing:
                - The number of iterations the model was trained for.
                - A list of the validation loss values for each epoch.
        """
        eval_set = None
        use_early_stopping = self.early_stopping_rounds is not None and forced_iterations is None

        features_train, offset_train = self._split_column_roles(x_train, fit=True)
        fit_params: dict[str, Any] = {}
        if offset_train is not None:
            fit_params["base_margin"] = offset_train
        if sample_weight is not None:
            fit_params["sample_weight"] = sample_weight

        params = self.params.copy()
        if self.categorical_features:
            params["enable_categorical"] = True
        if use_early_stopping and x_val is not None:
            x_val, offset_val = self._split_column_roles(x_val)
            eval_set = [(x_val, y_val)]
            if offset_val is not None:
                fit_params["base_margin_eval_set"] = [offset_val]
            if sample_weight_val is not None:
                fit_params["sample_weight_eval_set"] = [sample_weight_val]
            params["early_stopping_rounds"] = self.early_stopping_rounds
        else:
            if "early_stopping_rounds" in params:
                del params["early_stopping_rounds"]
            self.train_indices = np.arange(x_train.shape[0])
            self.val_indices = None

        # Dynamically set objective for classification
        if self.task_type == TaskType.CLASSIFICATION:
            num_classes = len(np.unique(y_train))
            if num_classes > 2:
                self.objective = "multi:softprob"
                self.eval_metric = "mlogloss"
                params["num_class"] = num_classes
            else:
                self.objective = "binary:logistic"
                self.eval_metric = Metric.LOG_LOSS.label

        model_instance = xgb.XGBClassifier if self.task_type == TaskType.CLASSIFICATION else xgb.XGBRegressor
        params.setdefault("n_estimators", 500)
        self.model = model_instance(objective=self.objective, eval_metric=self.eval_metric, random_state=self.random_seed, **params)

        if forced_iterations is not None:
            self.model.n_estimators = forced_iterations

        if eval_set is not None:
            fit_params["eval_set"] = eval_set
            fit_params["verbose"] = False

        self.model.fit(features_train, y_train, **fit_params)

        if self.is_regression_model:
            y_pred_train = self.predict(x_train, filter_data=False)
            self._train_residual_std = np.std(y_train - y_pred_train)
            if np.isnan(self._train_residual_std):
                self._train_residual_std = 0.0

        chose_by_early_stopping = use_early_stopping and self.model.best_iteration is not None
        # XGBoost counts rounds from 0, so the number of rounds kept is best_iteration + 1 when early stopping chose it.
        rounds_kept = int(self.model.best_iteration) + 1 if chose_by_early_stopping else int(self.model.n_estimators)
        loss_history = get_loss_history(self.model, use_early_stopping)
        if forced_iterations is None:  # a refit with forced rounds must not overwrite the record of the pass that chose them
            self.best_iteration_ = rounds_kept
            self.hit_cap_ = rounds_kept == int(self.model.n_estimators)  # the best round is the last one allowed: more rounds may still help

        return rounds_kept, loss_history

    def _clone(self) -> "XGBoostModel":
        """Creates a new instance of the model with the same parameters."""
        return XGBoostModel(**self.get_params())

    def get_params(self) -> dict[str, Any]:
        """Gets parameters for this estimator.

        Returns:
            dict: Parameter names mapped to their values.
        """
        params = super().get_params()
        params.update({"random_seed": self.random_seed, "task_type": self.task_type})
        return params

    def cross_validate(self, x: np.ndarray, y: np.ndarray, cv: int) -> dict[str, Any]:
        """Performs cross-validation and returns the scores."""
        self.cv_folds = cv
        self.fit(x, y)
        return {"test_score": self.cv_score_mean_}

    def _predict(self, x: np.ndarray, filter_data: bool = True) -> np.ndarray:
        """Makes predictions on new data."""
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        if filter_data:
            x = self._filter_predict_data(x)
        features, offset = self._split_column_roles(x)
        return self.model.predict(features) if offset is None else self.model.predict(features, base_margin=offset)  # includes the starting score

    def predict_uncertainty(self, x: np.ndarray, filter_data: bool = True) -> np.ndarray:
        """Estimates uncertainty for predictions."""
        if not self.is_regression_model:
            raise ValueError("predict_uncertainty is only available for regression models.")
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")

        if self.uncertainty_method == UncertaintyMethod.BINNED_RESIDUAL_STD:
            return super().predict_uncertainty(x, filter_data)

        # Default to CONSTANT uncertainty
        if filter_data:
            x = self._filter_predict_data(x)
        return np.full(x.shape[0], self._train_residual_std)

    def predict_proba(self, x: np.ndarray, filter_data: bool = True) -> np.ndarray:
        """Predicts class probabilities for classification tasks.

        Args:
            x (np.ndarray): Features for probability prediction.
            filter_data (bool): If True, filter the input data using the feature selection mask.

        Returns:
            np.ndarray: Predicted probabilities.
        """
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        if not hasattr(self.model, "predict_proba"):
            raise ValueError("predict_proba is not available for the current XGBoost configuration (likely regression).")
        if filter_data:
            x = self._filter_predict_data(x)
        proba = self.model.predict_proba(self._split_column_roles(x)[0])
        return ensure_proba_shape(proba, self.model.n_classes_)

    def get_hyperparameter_search_space(self) -> dict[str, Any]:
        """Defines the hyperparameter search space for XGBoost.

        Returns:
            Dict[str, Any]: A dictionary defining the hyperparameter search space.
        """
        space = {
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
            "max_depth": {"type": "int", "low": 3, "high": 9, "step": 2},
            "subsample": {"type": "float", "low": 0.6, "high": 1.0, "step": 0.1},
            "colsample_bytree": {"type": "float", "low": 0.6, "high": 1.0, "step": 0.1},
            "gamma": {"type": "float", "low": 0.0, "high": 0.2, "step": 0.05},
            "reg_alpha": {"type": "float", "low": 1e-6, "high": 1.0, "log": True},  # L1 regularization
            "reg_lambda": {"type": "float", "low": 1e-6, "high": 1.0, "log": True},  # L2 regularization
        }
        if self.early_stopping_rounds is None:
            space["n_estimators"] = {"type": "int", "low": 5, "high": 550, "step": 50}
        if self.search_space_override:
            space.update(self.search_space_override)
        return space

    def get_internal_model(self) -> xgb.XGBRegressor | xgb.XGBClassifier | None:
        """Returns the raw underlying XGBoost model."""
        return self.model

    def get_shap_explainer_info(self) -> dict[str, Any]:
        """Gets the SHAP explainer type and the model to be explained."""
        return {"explainer_type": ExplainerType.TREE, "model": self.get_internal_model()}

    def get_num_parameters(self) -> int:
        """Returns the number of estimators in the XGBoost model.

        Returns:
            int: The number of estimators.
        """
        if self.model is None:
            return 0
        if hasattr(self.model, "best_iteration") and self.model.best_iteration is not None:
            return self.model.best_iteration
        return self.model.n_estimators

    def get_classifier_predictions(self, x: np.ndarray, y_true_original: np.ndarray) -> Never:
        """Not implemented for XGBoostModel.

        Raises:
            NotImplementedError: XGBoostModel is not a composite model.
        """
        raise NotImplementedError("XGBoostModel is not a composite model and does not have an internal classifier for separate prediction.")
