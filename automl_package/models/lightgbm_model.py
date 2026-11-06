"""LightGBM model wrapper for AutoML."""

from typing import Any, NoReturn

import lightgbm as lgb
import numpy as np

from automl_package.enums import CategoricalCast, ComputeDevice, ExplainerType, ForecastLoss, Metric, TaskType, UncertaintyMethod
from automl_package.logger import logger
from automl_package.models.base import BaseModel
from automl_package.models.common.column_roles import ColumnRolesMixin
from automl_package.models.common.common import get_loss_history
from automl_package.utils.losses import poisson_deviance, tree_model_gaussian_nll_eval_metric, tree_model_gaussian_nll_objective
from automl_package.utils.numerics import ensure_proba_shape
from automl_package.utils.pytorch_utils import resolve_compute_device

LIGHTGBM_DEVICE_PARAM = "device_type"  # LightGBM's parameter naming the training device
LIGHTGBM_CUDA_DEVICE = "cuda"  # its CUDA tree learner (only in a build compiled with USE_CUDA)
BINARY_CLASS_COUNT = 2  # more classes than this is a multiclass objective
# Training rows from which AUTO puts LightGBM on the GPU. Measured 2026-11-06 on the GB10 (300 rounds, depth 8, 64 columns, vs 4 CPU
# threads; /home/ff235/cache/automl/build_reports/gpu_native_builds.md): the CUDA learner's time barely grows with rows, the CPU's
# grows in proportion; GPU 2.5x slower at 300k rows, 1.1-1.3x faster at 1M.
LIGHTGBM_GPU_MIN_ROWS = 1_000_000


class LightGBMModel(ColumnRolesMixin, BaseModel):
    """LightGBM model wrapper.

    ``offset_feature`` is handed to LightGBM as ``init_score`` (``eval_init_score`` for the validation set); LightGBM's own ``predict``
    excludes it, so ``predict`` adds it back. ``categorical_features`` go to ``categorical_feature`` with the codes as given.
    See `ColumnRolesMixin`.

    ``loss=ForecastLoss.POISSON_DEVIANCE`` (regression, ``UncertaintyMethod.CONSTANT``): the target is a log count ``t`` (e.g. ``log1p(y)``), LightGBM fits
    objective ``poisson`` on the label ``exp(t)`` with the offset as ``init_score``, and early stopping reads LightGBM's ``poisson`` metric on the
    validation rows (with ``eval_init_score``), the same objective up to a constant. ``predict`` still returns the linear predictor (raw score plus
    offset) in target space, so a caller's ``expm1(prediction)`` is unchanged, and the held-out fit term is the Poisson deviance of ``exp(t)`` against
    ``exp(prediction)``. The default ``SQUARED_ERROR`` is the original behaviour.
    """

    _CATEGORICAL_CAST = CategoricalCast.NATIVE_CODES

    def __init__(
        self,
        random_seed: int | None = None,
        loss: ForecastLoss = ForecastLoss.SQUARED_ERROR,
        compute_device: ComputeDevice = ComputeDevice.AUTO,
        gpu_min_rows: int = LIGHTGBM_GPU_MIN_ROWS,
        **kwargs: Any,
    ) -> None:
        """Initializes the LightGBMModel.

        Args:
            random_seed: seed handed to LightGBM.
            loss: the fit term (enums.ForecastLoss); POISSON_DEVIANCE needs a regression task with constant uncertainty.
            compute_device: where it trains (enums.ComputeDevice). AUTO takes the GPU when `get_device()` finds CUDA and the fit has at
                least `gpu_min_rows` rows (below that the CPU is faster); GPU and CPU are taken as asked. The GPU needs LightGBM's CUDA
                build, which install.sh compiles on a GPU machine (scripts/gpu_builds.toml).
            gpu_min_rows: training rows from which AUTO uses the GPU.
            **kwargs: passed on to the base class; LightGBM's own parameters go in ``params``.
        """
        super().__init__(**kwargs)
        self.random_seed = random_seed
        self.loss = ForecastLoss(loss)
        self.compute_device = ComputeDevice(compute_device)
        self.gpu_min_rows = gpu_min_rows
        self._last_fit_device: ComputeDevice | None = None
        self.model = None
        self._train_residual_std = 0.0
        self.num_iterations_used = 0

        self.params.setdefault("verbose", -1)
        if self.task_type == TaskType.REGRESSION:
            if self.uncertainty_method == UncertaintyMethod.PROBABILISTIC:
                self.objective = tree_model_gaussian_nll_objective
                self.metric = "None"  # Use custom eval metric
                self.params["num_class"] = 2
            elif self.loss is ForecastLoss.POISSON_DEVIANCE:
                self.objective = "poisson"
                self.metric = "poisson"
            else:
                self.objective = "regression"
                self.metric = Metric.RMSE.label
        elif self.task_type == TaskType.CLASSIFICATION:
            # Objective and metric will be set dynamically in _fit_single
            self.objective = None
            self.metric = None
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")
        if self.loss is ForecastLoss.POISSON_DEVIANCE and not (self.is_regression_model and self.uncertainty_method == UncertaintyMethod.CONSTANT):
            raise ValueError(f"loss=POISSON_DEVIANCE needs task_type=REGRESSION and uncertainty_method=CONSTANT (got {self.task_type}, {self.uncertainty_method}).")

    @property
    def name(self) -> str:
        """Returns the name of the model."""
        return "LightGBMModel"

    def fit_device(self, n_rows: int) -> ComputeDevice:
        """Where a fit on `n_rows` rows trains (see `compute_device`); logged whenever the choice changes."""
        large_enough = self.compute_device is ComputeDevice.GPU or n_rows >= self.gpu_min_rows
        device = ComputeDevice.GPU if large_enough and resolve_compute_device(self.compute_device) is ComputeDevice.GPU else ComputeDevice.CPU
        if device is not self._last_fit_device:
            logger.info(f"LightGBMModel trains on the {device.value.upper()} (asked {self.compute_device.value}, {n_rows:,} rows, "
                        f"GPU from {self.gpu_min_rows:,} rows under auto)")
            self._last_fit_device = device
        return device

    def _fit_single(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        forced_iterations: int | None = None,
    ) -> tuple[int, list[float]]:
        """Fits a single model instance."""
        eval_set = None
        callbacks = []
        use_early_stopping = self.early_stopping_rounds is not None and forced_iterations is None

        features_train, offset_train = self._split_column_roles(x_train, fit=True)
        fit_params: dict[str, Any] = {}
        if offset_train is not None:
            fit_params["init_score"] = offset_train
        if self.categorical_features:
            fit_params["categorical_feature"] = self.categorical_features

        if use_early_stopping and x_val is not None:
            x_val, offset_val = self._split_column_roles(x_val)
            eval_set = [(x_val, self._fit_label(y_val))]
            if offset_val is not None:
                fit_params["eval_init_score"] = [offset_val]
            callbacks.append(lgb.early_stopping(self.early_stopping_rounds, verbose=False))

        params = self.params.copy()
        fit_device = self.fit_device(len(features_train))
        if fit_device is ComputeDevice.GPU:
            params[LIGHTGBM_DEVICE_PARAM] = LIGHTGBM_CUDA_DEVICE
        params.setdefault("n_estimators", 500)

        # Dynamically set objective for classification
        if self.task_type == TaskType.CLASSIFICATION:
            num_classes = len(np.unique(y_train))
            if num_classes > BINARY_CLASS_COUNT:
                self.objective = "multiclass"
                self.metric = "multi_logloss"
                params["num_class"] = num_classes
            else:
                self.objective = "binary"
                self.metric = Metric.LOG_LOSS.label

        model_instance = lgb.LGBMClassifier if self.task_type == TaskType.CLASSIFICATION else lgb.LGBMRegressor
        self.model = model_instance(
            objective=self.objective,
            metric=self.metric,
            random_state=self.random_seed,
            **params,
        )

        if forced_iterations is not None:
            self.model.n_estimators = forced_iterations

        fit_params.update({"eval_set": eval_set, "callbacks": callbacks})
        if self.uncertainty_method == UncertaintyMethod.PROBABILISTIC and eval_set is not None:
            fit_params["eval_metric"] = tree_model_gaussian_nll_eval_metric

        try:
            self.model.fit(features_train, self._fit_label(y_train), **fit_params)
        except lgb.basic.LightGBMError as error:
            if fit_device is ComputeDevice.GPU and "CUDA" in str(error):
                raise RuntimeError(
                    f"LightGBM was asked for the GPU but this build cannot use it ({error}). Remedy: rebuild the environment with ./install.sh, "
                    "which compiles LightGBM for this GPU (scripts/gpu_builds.toml), or pass compute_device=ComputeDevice.CPU."
                ) from error
            raise

        if self.is_regression_model and self.uncertainty_method == UncertaintyMethod.CONSTANT:
            y_pred_train = self.predict(x_train, filter_data=False)
            self._train_residual_std = np.std(y_train - y_pred_train)

        best_iteration = self.model.best_iteration_ if use_early_stopping and self.model.best_iteration_ is not None else self.model.n_estimators
        loss_history = get_loss_history(self.model, use_early_stopping)
        if forced_iterations is None:  # a refit with forced rounds must not overwrite the record of the pass that chose them
            self.best_iteration_ = int(best_iteration)
            self.hit_cap_ = self.best_iteration_ == int(self.model.n_estimators)  # the best round is the last one allowed: more rounds may still help

        return best_iteration, loss_history

    def _fit_label(self, y: np.ndarray) -> np.ndarray:
        """The label LightGBM fits: the target itself, or ``exp(target)`` (the count) under the Poisson loss."""
        return np.exp(y) if self.loss is ForecastLoss.POISSON_DEVIANCE else y

    def _held_out_fit_score(self, x_val: np.ndarray, y_val: np.ndarray) -> float:
        """Held-out fit term: the Poisson deviance under ``POISSON_DEVIANCE`` (lower is better), else the base class's score."""
        if self.loss is not ForecastLoss.POISSON_DEVIANCE:
            return super()._held_out_fit_score(x_val, y_val)
        return float(np.mean(poisson_deviance(np.asarray(y_val, dtype=float), self._predict_for_scoring(x_val))))

    def _clone(self) -> "LightGBMModel":
        """Creates a new instance of the model with the same parameters."""
        return LightGBMModel(**self.get_params())

    def get_params(self) -> dict[str, Any]:
        """Gets parameters for this estimator."""
        params = super().get_params()
        params.update({"task_type": self.task_type, "random_seed": self.random_seed, "loss": self.loss})
        return params

    def _predict(self, x: np.ndarray, filter_data: bool = True) -> np.ndarray:
        """Makes predictions on new data."""
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        if filter_data:
            x = self._filter_predict_data(x)
        features, offset = self._split_column_roles(x)

        if self.uncertainty_method == UncertaintyMethod.PROBABILISTIC:
            # The sklearn wrapper returns a 2D array for multi-class objectives
            raw_predictions = self.model.predict(features, raw_score=True)
            return raw_predictions[:, 0]  # Return only the mean
        predictions = self.model.predict(features, raw_score=self.loss is ForecastLoss.POISSON_DEVIANCE)  # the linear predictor, not exp() of it
        return predictions if offset is None else predictions + offset  # LightGBM's predict excludes init_score: add the starting score back

    def predict_proba(self, x: np.ndarray, filter_data: bool = True) -> np.ndarray:
        """Predicts class probabilities."""
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        if not hasattr(self.model, "predict_proba"):
            raise ValueError("predict_proba is not available for this model.")
        if filter_data:
            x = self._filter_predict_data(x)
        proba = self.model.predict_proba(self._split_column_roles(x)[0])
        return ensure_proba_shape(proba, self.model.n_classes_)

    def predict_uncertainty(self, x: np.ndarray, filter_data: bool = True) -> np.ndarray:
        """Estimates uncertainty for predictions."""
        if not self.is_regression_model:
            raise ValueError("predict_uncertainty is only available for regression models.")
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")

        if self.uncertainty_method == UncertaintyMethod.BINNED_RESIDUAL_STD:
            return super().predict_uncertainty(x, filter_data)

        if filter_data:
            x = self._filter_predict_data(x)

        if self.uncertainty_method == UncertaintyMethod.PROBABILISTIC:
            raw_predictions = self.model.predict(self._split_column_roles(x)[0], raw_score=True)
            log_variance = raw_predictions[:, 1]
            variance = np.exp(log_variance)
            return np.sqrt(variance)

        return np.full(x.shape[0], self._train_residual_std)

    def get_hyperparameter_search_space(self) -> dict[str, Any]:
        """Gets the hyperparameter search space for the model."""
        space = {
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
            "num_leaves": {"type": "int", "low": 20, "high": 150, "step": 10},
            "max_depth": {"type": "int", "low": 5, "high": 15, "step": 2},
            "min_child_samples": {"type": "int", "low": 20, "high": 40, "step": 5},
            "subsample": {"type": "float", "low": 0.6, "high": 1.0, "step": 0.1},
            "colsample_bytree": {"type": "float", "low": 0.6, "high": 1.0, "step": 0.1},
            "reg_alpha": {"type": "float", "low": 1e-6, "high": 1.0, "log": True},
            "reg_lambda": {"type": "float", "low": 1e-6, "high": 1.0, "log": True},
        }
        if self.early_stopping_rounds is None:
            space["n_estimators"] = {"type": "int", "low": 5, "high": 550, "step": 50}
        if self.search_space_override:
            space.update(self.search_space_override)
        return space

    def get_internal_model(self) -> Any:
        """Returns the raw underlying LightGBM model."""
        return self.model

    def get_shap_explainer_info(self) -> dict[str, Any]:
        """Gets the SHAP explainer type and the model to be explained."""
        return {"explainer_type": ExplainerType.TREE, "model": self.get_internal_model()}

    def get_num_parameters(self) -> int:
        """Returns the number of estimators in the LightGBM model."""
        if self.model is None:
            num_parameters = 0
        elif hasattr(self.model, "best_iteration_") and self.model.best_iteration_ is not None:
            num_parameters = self.model.best_iteration_
        else:
            num_parameters = self.model.n_estimators
        return num_parameters

    def get_classifier_predictions(self, x: np.ndarray, y_true_original: np.ndarray) -> NoReturn:
        """Not implemented for LightGBMModel."""
        raise NotImplementedError("LightGBMModel is not a composite model.")

    def cross_validate(self, x: np.ndarray, y: np.ndarray, cv: int) -> dict[str, Any]:
        """Performs cross-validation."""
        self.cv_folds = cv
        self.fit(x, y)
        return {"test_score": self.cv_score_mean_}
