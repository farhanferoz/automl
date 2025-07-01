from enum import Enum


class UncertaintyMethod(Enum):
    """Enum for different uncertainty estimation methods."""

    CONSTANT = "constant"
    MC_DROPOUT = "mc_dropout"
    PROBABILISTIC = "probabilistic"


class RegressionStrategy(Enum):
    """Enum for different regression strategies in ProbabilisticRegressionModel."""

    SEPARATE_HEADS = "separate_heads"
    SINGLE_HEAD_N_OUTPUTS = "single_head_n_outputs"
    SINGLE_HEAD_FINAL_OUTPUT = "single_head_final_output"


class MapperType(Enum):
    """Enum for different mapping types in ClassifierRegressionModel."""

    LINEAR = "linear"
    LOOKUP_MEAN = "lookup_mean"
    LOOKUP_MEDIAN = "lookup_median"
    SPLINE = "spline"


class TaskType(Enum):
    """Enum for different machine learning task types."""

    REGRESSION = "regression"
    CLASSIFICATION = "classification"


class ModelName(Enum):
    """Enum for names of different machine learning models."""

    JAX_LINEAR_REGRESSION = "JAXLinearRegression"
    PYTORCH_NEURAL_NETWORK = "PyTorchNeuralNetwork"
    FLEXIBLE_NEURAL_NETWORK = "FlexibleNeuralNetwork"
    XGBOOST = "XGBoost"
    LIGHTGBM = "LightGBM"
    SKLEARN_LOGISTIC_REGRESSION = "SKLearnLogisticRegression"
    CATBOOST = "CatBoost"
    CLASSIFIER_REGRESSION = "ClassifierRegression"
    PROBABILISTIC_REGRESSION = "ProbabilisticRegression"
