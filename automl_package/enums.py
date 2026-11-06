"""Enums for AutoML package."""

from enum import Enum


class TaskType(Enum):
    """Enum for the type of machine learning task."""

    REGRESSION = "regression"
    CLASSIFICATION = "classification"


class UncertaintyMethod(Enum):
    """Enum for different uncertainty estimation methods."""

    CONSTANT = "constant"
    MC_DROPOUT = "mc_dropout"
    PROBABILISTIC = "probabilistic"
    BINNED_RESIDUAL_STD = "binned_residual_std"


class RegressionStrategy(Enum):
    """Enum for different regression strategies in probabilistic models."""

    SEPARATE_HEADS = "separate_heads"
    SINGLE_HEAD_N_OUTPUTS = "single_head_n_outputs"
    SINGLE_HEAD_FINAL_OUTPUT = "single_head_final_output"


class BoundaryRegularizationMethod(Enum):
    """Enum for boundary enforcement methods."""

    NONE = "none"
    PENALTY = "penalty"
    HARDSIGMOID = "hardsigmoid"


class MapperType(Enum):
    """Enum for different probability mapping strategies."""

    LINEAR = ("linear", False)
    LOOKUP_MEDIAN = ("lookup_median", False)
    LOOKUP_MEAN = ("lookup_mean", False)
    SPLINE = ("spline", False)
    AUTO = ("auto", False)
    NN_SEPARATE_HEADS = ("nn_separate_heads", True)
    NN_SINGLE_HEAD_N_OUTPUTS = ("nn_single_head_n_outputs", True)
    NN_SINGLE_HEAD_FINAL_OUTPUT = ("nn_single_head_final_output", True)

    def __init__(self, label: str, is_nn: bool) -> None:
        """Initializes the MapperType enum member.

        Args:
            label: The string value for the enum.
            is_nn: A boolean indicating if the mapper is a neural network.
        """
        self.label = label
        self.is_nn = is_nn


class ModelName(Enum):
    """Enum for different model names supported by AutoML."""

    JAX_LINEAR_REGRESSION = "JAXLinearRegression"
    NORMAL_EQUATION_LINEAR_REGRESSION = "NormalEquationLinearRegression"
    PYTORCH_LINEAR_REGRESSION = "PyTorchLinearRegression"
    PYTORCH_NEURAL_NETWORK = "PyTorchNeuralNetwork"
    FLEXIBLE_NEURAL_NETWORK = "FlexibleNeuralNetwork"
    XGBOOST = "XGBoostModel"
    LIGHTGBM = "LightGBMModel"
    CATBOOST = "CatBoostModel"
    CLASSIFIER_REGRESSION = "ClassifierRegression"
    PROBABILISTIC_REGRESSION = "ProbabilisticRegression"
    LSTM_FORECASTER = "PyTorchLSTMForecaster"


class LearnedRegularizationType(Enum):
    """Enum for different types of learned regularization."""

    L1_ONLY = "l1_only"
    L2_ONLY = "l2_only"
    L1_L2 = "l1_l2"


class LayerSelectionMethod(Enum):
    """Enum for different layer selection methods in flexible neural networks."""

    NONE = "none"
    SOFT_GATING = "soft_gating"
    GUMBEL_SOFTMAX = "gumbel_softmax"
    STE = "ste"
    REINFORCE = "reinforce"
    NESTED = "nested"  # per-sample depth draws (a training schedule, not a selector) — capacity-ladder F2


class DepthRegularization(Enum):
    """Enum for depth complexity control in flexible neural networks."""

    NONE = "none"
    DEPTH_PENALTY = "depth_penalty"
    ELBO = "elbo"
    COST_AWARE_ELBO = "cost_aware_elbo"


class CapacitySelection(Enum):
    """The ONE capacity-selection API, shared by every capacity-dial family.

    Capacity-programme Task FP-3, `docs/flexnn/package/plan.md`.
    Ships only the members whose mechanism exists today -- an enum member is a promise that it
    works, not a placeholder. A member is added by the task that builds its mechanism, never ahead
    of it (the rule that retires the old `WidthSelectionMethod.DISTILLED` trap: an enum member whose
    implementation raises `NotImplementedError`).

    `GLOBAL_CHEAP` and `GLOBAL_SWEEP` were added 2026-07-21 by `docs/probreg/shared/plan.md` PA, which built both
    mechanisms for ProbReg. **The k family implements them; width and depth do NOT yet** -- those
    arrive with `docs/flexnn/width/plan.md` WSEL-3/WSEL-4 and `docs/flexnn/depth/plan.md` DSEL-6/DSEL-7. A member existing
    here is a promise about THIS enum's contract, not a claim that every family implements it.
    """

    FIXED = "fixed"  # no distilled-router selection; the model's own default single-pass behavior
    # (an explicit width/depth/k, or its trained in-training selection strategy) is used unchanged.
    PER_INPUT = "per_input"  # a `DistilledCapacityRouter` fitted via `fit_router()` chooses the
    # capacity per sample; `predict()`/`predict_uncertainty()` route with no caller flag.
    GLOBAL_CHEAP = "global_cheap"  # ONE capacity for the whole dataset, read cheaply off an
    # already-trained model's held-out per-rung curve (cheapest-within-tolerance at twice a
    # bootstrap SE) -- e.g. ProbReg M1, `ProbabilisticRegressionModel.fit_global_selector`.
    GLOBAL_SWEEP = "global_sweep"  # ONE capacity for the whole dataset, found by training a
    # SEPARATE ordinary model per candidate value and applying the SAME selection rule -- the
    # expensive reference the cheap arms are measured against -- e.g. ProbReg M3,
    # `ProbabilisticRegressionModel.fit_sweep_selector`.


class NClassesRegularization(Enum):
    """Enum for n_classes complexity control in probabilistic regression.

    K_PENALTY and ELBO are **RETIRED under the nested ladder** (MASTER Decision 29,
    `docs/flexnn/PLAN.md`): they are only meaningful alongside an in-training
    `NClassesSelectionMethod`, and those are themselves retired. Under `NESTED` they raise at
    construction and are absent from `get_hyperparameter_search_space`; they are reachable ONLY
    behind the explicit `allow_retired_capacity_selection` opt-out, for a labelled comparison arm,
    and any run using it records that fact in its results JSON.

    The live path is `NClassesSelectionMethod.NESTED` training + `DistilledCapacityRouter`-routed
    inference (`ProbabilisticRegressionModel.fit_router`), which needs no regularizer at all.

    *(This docstring previously described these as a "labeled COMPARISON ARM" per Decision 13.
    Decision 29 superseded that: demoted became unreachable-by-default.)*
    """

    NONE = "none"
    K_PENALTY = "k_penalty"
    ELBO = "elbo"


class NClassesSelectionMethod(Enum):
    """Enum for different n_classes selection methods in probabilistic regression.

    SOFT_GATING/GUMBEL_SOFTMAX/STE/REINFORCE are in-training selection and are **RETIRED**
    (MASTER Decision 29): under the nested ladder they raise at construction and are absent from
    `get_hyperparameter_search_space`, reachable only behind the explicit
    `allow_retired_capacity_selection` opt-out for a labelled comparison arm. Retirement is not
    deletion -- the code stays, and any run using the opt-out records that fact in its results JSON.
    *(Previously "a labeled COMPARISON ARM kept fully functional"; Decision 29 superseded that.)*
    NESTED is the
    recommended path: per-sample k drawn as a training SCHEDULE (capacity-ladder Task F9, ported
    from `automl_package/examples/_capacity_ladder_nested.py`), with per-input k at inference
    supplied post-hoc by `ProbabilisticRegressionModel.fit_router` +
    `automl_package.models.common.distilled_router.DistilledCapacityRouter` -- selection is
    DISTILLED, never in-training.
    """

    NONE = "none"
    SOFT_GATING = "soft_gating"
    GUMBEL_SOFTMAX = "gumbel_softmax"
    STE = "ste"
    REINFORCE = "reinforce"
    NESTED = "nested"  # per-sample k draws (a training schedule, not a selector) -- capacity-ladder F9; RECOMMENDED path w/ DistilledCapacityRouter (Decision 13)


class CategoricalEncodingStrategy(Enum):
    """Enum for different categorical encoding strategies."""

    ONE_HOT = "one_hot"
    ORDERED_TARGET = "ordered_target"


class Metric(Enum):
    """Enum for different evaluation metrics."""

    RMSE = ("rmse", True)
    MSE = ("mse", True)
    MAE = ("mae", True)
    NLL = ("nll", True)
    CRPS = ("crps", True)
    ACCURACY = ("accuracy", False)
    LOG_LOSS = ("log_loss", True)

    def __init__(self, label: str, is_smaller_better: bool) -> None:
        """Initializes the Metric enum member.

        Args:
            label: The string value for the enum.
            is_smaller_better: A boolean indicating if a smaller metric value is better.
        """
        self.label = label
        self.is_smaller_better = is_smaller_better


class ActivationFunction(Enum):
    """Enum for different activation functions."""

    RELU = "relu"
    TANH = "tanh"
    SIGMOID = "sigmoid"
    LEAKY_RELU = "leaky_relu"
    ELU = "elu"
    SELU = "selu"
    SOFTPLUS = "softplus"
    SWISH = "swish"
    MISH = "mish"
    GELU = "gelu"
    PRELU = "prelu"
    RRELU = "rrelu"
    HARDSHRINK = "hardshrink"
    SOFTSHRINK = "softshrink"
    TANHSHRINK = "tanhshrink"
    SOFTMIN = "softmin"
    SOFTMAX = "softmax"
    LOG_SOFTMAX = "log_softmax"
    ADAPTIVE_LOG_SOFTMAX_WITH_LOSS = "adaptive_log_softmax_with_loss"
    GLU = "glu"
    LOGSIGMOID = "logsigmoid"
    HARDTANH = "hardtanh"
    THRESHOLD = "threshold"
    RELU6 = "relu6"
    CELU = "celu"
    SILU = "silu"
    HARDSWISH = "hardswish"
    IDENTITY = "identity"
    LINEAR = "linear"


class Penalty(Enum):
    """Enum for different regularization penalties."""

    L1 = "l1"
    L2 = "l2"
    ELASTICNET = "elasticnet"


class FunctionType(Enum):
    """Enum for different function types."""

    SIN = "sin"
    POLYNOMIAL = "polynomial"
    LINEAR = "linear"


class DataSplitStrategy(Enum):
    """Enum for different data split strategies."""

    RANDOM = "random"
    DISTINCT_DATES = "distinct_dates"
    TIME_ORDERED = "time_ordered"


class ExplainerType(Enum):
    """Enum for different SHAP explainer types."""

    TREE = "tree"
    DEEP = "deep"
    LINEAR = "linear"
    KERNEL = "kernel"
    CATBOOST_PROBABILISTIC_PROXY = "catboost_probabilistic_proxy"


class ProbabilisticRegressionOptimizationStrategy(Enum):
    """Enum for different optimization strategies in ProbabilisticRegressionModel."""

    REGRESSION_ONLY = "regression_only"
    COMPOSITE_LOSS = "composite_loss"
    GRADIENT_STOP = "gradient_stop"  # "classifier learns nothing" control: no CE, probs detached, classifier keeps its initial weights
    CE_STOP_GRAD = "ce_stop_grad"  # classifier trained on bin-CE only; probs detached before heads
    TWO_STAGE = "two_stage"  # classifier trained alone on bin-CE (stopped on held-out CE), frozen; then heads trained on its detached probs


class TwoStagePhase(Enum):
    """The phase a `ProbabilisticRegressionOptimizationStrategy.TWO_STAGE` fit is in."""

    CLASSIFIER = "classifier"  # classifier alone, by cross-entropy on the class labels
    HEADS = "heads"  # heads on the frozen classifier's detached probabilities, by the regression fit term


class HpoSampler(Enum):
    """Optuna sampler a hyperparameter search draws its trials with."""

    TPE = "tpe"
    RANDOM = "random"


class KSelectionRule(Enum):
    """How a sweep over k picks the winner from the held-out error curve."""

    CHEAPEST_WITHIN_TOLERANCE = "cheapest_within_tolerance"  # smallest k within 2 bootstrap SE of the best (`utils/capacity_selection.py`)
    MINIMUM = "minimum"  # the k with the lowest mean held-out error


class ProbRegLossType(Enum):
    """Enum for loss function used in ProbabilisticRegressionModel."""

    GAUSSIAN_LTV = "gaussian_ltv"  # heads give mean and spread; one Gaussian via the law of total variance


class OptimizerType(Enum):
    """Enum for different optimizer types."""

    ADAM = "adam"
    HESSIAN_FREE = "hessian_free"


class Monotonicity(Enum):
    """Enum for different monotonicity constraints."""

    NONE = "none"
    POSITIVE = "positive"
    NEGATIVE = "negative"


class RnnBackend(Enum):
    """Kernel library that runs a recurrent layer on CUDA (no effect on CPU).

    Measured 2026-11-03 on the GB10 (cuDNN 9.15 and 9.26, batch 1024, window 56): cuDNN's LSTM takes 93-174 ms per
    training step at hidden sizes 160 and 192 against 10-17 ms at 128, 224 and 256, while PyTorch's own kernels take
    13-15 ms there and are at most ~37% slower than cuDNN at the other sizes.
    """

    CUDNN = "cudnn"  # NVIDIA's fused kernels (PyTorch's default)
    NATIVE = "native"  # PyTorch's own kernels (cuDNN switched off for the recurrent layer only)


class CategoricalCast(Enum):
    """How a boosting library wants the integer category codes of a categorical column handed over."""

    NATIVE_CODES = "native_codes"  # LightGBM: the codes as given (float, NaN = missing), named in `categorical_feature`
    INTEGER_CODES = "integer_codes"  # CatBoost: integer codes, NaN mapped to the code -1 (its own category)
    PANDAS_CATEGORICAL = "pandas_categorical"  # XGBoost: pandas Categorical with the category levels fixed at fit


class CategoricalEncoding(Enum):
    """How a model reads its ``categorical_features`` columns (integer category codes, NaN = missing; `BaseModel.categorical_encoding`)."""

    NATIVE = "native"  # the boosting library splits on the codes itself (LightGBM, XGBoost, CatBoost)
    ONE_HOT = "one_hot"  # one 0/1 column per category level seen in the training rows; a missing or unseen value is all zeros


class OutOfRangeAction(Enum):
    """What a model does with predictions outside the range of its training targets (see ``utils/prediction_envelope.py``)."""

    FLAG = "flag"  # report only; predictions are returned unchanged
    CLIP = "clip"  # clip each output column to its training [min, max] and log how many entries moved
    RAISE = "raise"  # raise ValueError when any prediction is outside the range


class OutputBound(Enum):
    """Whether a neural network's regression output is confined to the range of its training targets."""

    NONE = "none"  # the output is unconstrained (default)
    TRAINING_RANGE = "training_range"  # per output column, out = c + h * tanh((raw - c) / h) with [c - h, c + h] the training targets' min..max


class ForecastLoss(Enum):
    """Fit term (the loss early stopping and the search read on held-out rows) of a forecasting model."""

    SQUARED_ERROR = "squared_error"  # mean squared error between the model output and the target (default)
    POISSON_DEVIANCE = "poisson_deviance"  # Poisson deviance of the count exp(target [+ offset]) against exp(output [+ offset]); `utils.losses.poisson_deviance`


class TargetScaling(Enum):
    """How a multi-output model scales its targets before fitting (the forecasts always come back in target units)."""

    NONE = "none"  # targets as given
    PER_OUTPUT = "per_output"  # each output column standardised by the mean and standard deviation of the rows being fitted; NaN stays NaN


class InputScaling(Enum):
    """How a model scales its input columns before fitting (`BaseModel.input_scaling`)."""

    NONE = "none"  # inputs as given
    STANDARD = "standard"  # each numeric column standardised by the training part's mean and sd; binary 0/1, categorical and offset columns left alone
    QUANTILE = "quantile"  # each numeric column mapped to a standard normal through the training part's quantiles (rank-based: skew and outliers
    # removed, values beyond the training range capped at the extreme quantile); binary 0/1, categorical and offset columns left alone


class MissingInputs(Enum):
    """What a model does with a missing (NaN) input value (`BaseModel.missing_inputs`)."""

    RAISE = "raise"  # a NaN in the inputs raises ValueError naming the remedy
    TRAINING_MEAN = "training_mean"  # replaced by the training part's column mean (before any scaling)
    NATIVE = "native"  # passed through: the model handles NaN itself (gradient-boosted trees, the LSTM)


class ComputeDevice(Enum):
    """Where a model that is not a PyTorch network trains (gradient-boosted trees); PyTorch models use `get_device()`."""

    AUTO = "auto"  # the GPU when `get_device()` finds CUDA (honours AUTOML_DEVICE), else the CPU
    CPU = "cpu"
    GPU = "gpu"


class LRSchedule(Enum):
    """How a PyTorch network's learning rate moves over the epochs of one training pass (`PyTorchModelBase.lr_schedule`)."""

    NONE = "none"  # constant (after any warmup)
    REDUCE_ON_PLATEAU = "reduce_on_plateau"  # multiplied by `lr_schedule_factor` after `lr_schedule_patience` epochs without a new best held-out fit term
    COSINE = "cosine"  # cosine decay towards 0 over the epoch budget `n_epochs` (after any warmup)
