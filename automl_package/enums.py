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


class NClassesSelectionMethod(Enum):
    """Enum for different n_classes selection methods in probabilistic regression."""

    NONE = "none"
    SOFT_GATING = "soft_gating"
    GUMBEL_SOFTMAX = "gumbel_softmax"
    STE = "ste"
    REINFORCE = "reinforce"


class CategoricalEncodingStrategy(Enum):
    """Enum for different categorical encoding strategies."""

    ONE_HOT = "one_hot"
    ORDERED_TARGET = "ordered_target"


class Metric(Enum):
    """Enum for different evaluation metrics."""

    RMSE = ("rmse", True)
    MSE = ("mse", True)
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