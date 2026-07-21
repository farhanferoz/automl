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
    SKLEARN_LOGISTIC_REGRESSION = "SKLearnLogisticRegression"
    CLASSIFIER_REGRESSION = "ClassifierRegression"
    PROBABILISTIC_REGRESSION = "ProbabilisticRegression"


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

    Capacity-programme Task FP-3, `docs/plans/capacity_programme/flexnn-package.md`.
    Ships only the members whose mechanism exists today -- an enum member is a promise that it
    works, not a placeholder. A member is added by the task that builds its mechanism, never ahead
    of it (the rule that retires the old `WidthSelectionMethod.DISTILLED` trap: an enum member whose
    implementation raises `NotImplementedError`).

    `GLOBAL_CHEAP` and `GLOBAL_SWEEP` were added 2026-07-21 by `probreg.md` PA, which built both
    mechanisms for ProbReg. **The k family implements them; width and depth do NOT yet** -- those
    arrive with `width.md` WSEL-3/WSEL-4 and `depth-selection.md` DSEL-6/DSEL-7. A member existing
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
    `docs/plans/capacity_programme/MASTER.md`): they are only meaningful alongside an in-training
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
    GRADIENT_STOP = "gradient_stop"
    CE_STOP_GRAD = "ce_stop_grad"  # classifier trained on bin-CE only; probs detached before heads


class ProbRegLossType(Enum):
    """Enum for loss function used in ProbabilisticRegressionModel."""

    GAUSSIAN_LTV = "gaussian_ltv"  # law of total variance + Gaussian NLL (current default)
    MDN = "mdn"                    # Bishop 1994 mixture density network NLL


class OptimizerType(Enum):
    """Enum for different optimizer types."""

    ADAM = "adam"
    HESSIAN_FREE = "hessian_free"


class Monotonicity(Enum):
    """Enum for different monotonicity constraints."""

    NONE = "none"
    POSITIVE = "positive"
    NEGATIVE = "negative"
