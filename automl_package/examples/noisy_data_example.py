"""Example script for demonstrating AutoML capabilities with noisy data."""

from typing import Any

import numpy as np
from bokeh.models import Legend, LegendItem
from bokeh.palettes import Category10
from bokeh.plotting import figure, show
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from automl_package.enums import LearnedRegularizationType, TaskType
from automl_package.models.catboost_model import CatBoostModel
from automl_package.models.flexible_neural_network import FlexibleHiddenLayersNN
from automl_package.models.lightgbm_model import LightGBMModel
from automl_package.models.linear_regression import JAXLinearRegression
from automl_package.models.neural_network import PyTorchNeuralNetwork
from automl_package.models.normal_equation_linear_regression import NormalEquationLinearRegression
from automl_package.models.probabilistic_regression import ProbabilisticRegressionModel
from automl_package.models.pytorch_linear_regression import PyTorchLinearRegression
from automl_package.models.xgboost_model import XGBoostModel


def generate_noisy_data(n_samples: int = 1000, noise_level: float = 0.5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generates synthetic data with a non-linear relationship and significant noise."""
    np.random.seed(42)
    x = np.random.rand(n_samples, 1) * 10
    # A non-linear relationship
    y_true = np.sin(x).ravel() + x.ravel() * 0.5
    # Add significant noise
    noise = np.random.randn(n_samples) * noise_level
    y = y_true + noise
    return x, y, y_true


def plot_results_bokeh(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray, y_true_test: np.ndarray, results: dict[str, dict[str, Any]]) -> None:
    """Generates an interactive Bokeh plot for model comparison."""
    p = figure(
        width=900,
        height=600,
        title="Model Comparison on Noisy Non-Linear Data",
        x_axis_label="Feature",
        y_axis_label="Target",
    )

    # Training Data
    train_data_glyph = p.scatter(x_train.flatten(), y_train, color="gray", alpha=0.5, size=5)

    # Test Data
    test_data_glyph = p.scatter(x_test.flatten(), y_test, color="blue", alpha=0.5, size=5)

    # True Relationship
    sorted_indices_true = np.argsort(x_test, axis=0).flatten()
    true_rel_glyph = p.line(
        x_test[sorted_indices_true].flatten(),
        y_true_test[sorted_indices_true],
        line_dash="dashed",
        line_color="black",
        line_width=2,
    )

    # The Category10 palette supports up to 10 distinct colors.
    # We will use the largest palette and cycle through the colors if there are more than 10 models.
    colors = Category10[10]
    legend_items = [
        LegendItem(label="Training Data", renderers=[train_data_glyph]),
        LegendItem(label="Test Data", renderers=[test_data_glyph]),
        LegendItem(label="True Relationship", renderers=[true_rel_glyph]),
    ]

    for i, (name, result) in enumerate(results.items()):
        sorted_indices = np.argsort(x_test, axis=0).flatten()
        line_glyph = p.line(
            x_test[sorted_indices].flatten(),
            result["predictions"][sorted_indices],
            line_color=colors[i % len(colors)],
            line_width=2,
        )
        legend_items.append(LegendItem(label=f"{name} (MSE: {result['mse']:.4f})", renderers=[line_glyph]))

    # Create a custom legend to allow toggling
    legend = Legend(
        items=legend_items,
        location="top_left",
        click_policy="hide",
        background_fill_alpha=0.5,
        label_text_font_size="8pt",
    )
    p.add_layout(legend)

    show(p)


def run_experiment() -> None:
    """Runs the full experiment to compare model performances."""
    x, y, y_true = generate_noisy_data()
    x_train, x_test, y_train, y_test, y_true_train, y_true_test = train_test_split(x, y, y_true, test_size=0.2, random_state=42)

    # Scale features and target
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    x_train_scaled = scaler_x.fit_transform(x_train)
    x_test_scaled = scaler_x.transform(x_test)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

    # --- Model Initialization ---

    # PyTorch Neural Network
    pytorch_nn = PyTorchNeuralNetwork(
        input_size=x_train.shape[1],
        hidden_layers=2,
        hidden_size=64,
        task_type=TaskType.REGRESSION,
        n_epochs=100,
        learning_rate=0.01,
        early_stopping_rounds=20,
        validation_fraction=0.2,
        learn_regularization_lambdas=True,
        learned_regularization_type=LearnedRegularizationType.L1_L2,
        lambda_learning_rate=0.001,
        random_seed=1,
    )

    # Flexible Neural Network
    flexible_nn = FlexibleHiddenLayersNN(
        input_size=x_train.shape[1],
        max_hidden_layers=5,
        hidden_size=64,
        task_type=TaskType.REGRESSION,
        n_epochs=200,
        learning_rate=0.01,
        early_stopping_rounds=20,
        validation_fraction=0.2,
        gumbel_tau=0.1,
        n_predictor_layers=2,
        learn_regularization_lambdas=True,
        learned_regularization_type=LearnedRegularizationType.L1_L2,
        lambda_learning_rate=0.001,
        gumbel_tau_anneal_rate=0.9,
        n_predictor_learning_rate=0.01,
    )

    # Probabilistic Regression Model
    probabilistic_regression = ProbabilisticRegressionModel(
        input_size=x_train.shape[1],
        n_classes=3,
        n_epochs=200,
        early_stopping_rounds=20,
        validation_fraction=0.2,
        learn_regularization_lambdas=True,
        learned_regularization_type=LearnedRegularizationType.L1_L2,
        lambda_learning_rate=0.001,
        random_seed=1,
    )

    # CatBoost Model
    catboost_model = CatBoostModel(
        task_type=TaskType.REGRESSION,
        iterations=100,
        learning_rate=0.1,
        early_stopping_rounds=20,
        random_seed=1,
    )

    # LightGBM Model
    lightgbm_model = LightGBMModel(
        task_type=TaskType.REGRESSION,
        n_estimators=100,
        learning_rate=0.1,
        early_stopping_rounds=20,
        random_seed=1,
    )

    # Linear Regression
    linear_regression_model = JAXLinearRegression(
        learning_rate=0.01,
        n_iterations=1000,
        task_type=TaskType.REGRESSION,
        random_seed=1,
    )

    # Normal Equation Linear Regression
    normal_equation_linear_regression_model = NormalEquationLinearRegression(
        task_type=TaskType.REGRESSION,
    )

    # PyTorch Linear Regression
    pytorch_linear_regression_model = PyTorchLinearRegression(
        input_size=x_train.shape[1],
        task_type=TaskType.REGRESSION,
        n_epochs=100,
        learning_rate=0.01,
        early_stopping_rounds=20,
        validation_fraction=0.2,
        random_seed=1,
    )

    # XGBoost Model
    xgboost_model = XGBoostModel(
        task_type=TaskType.REGRESSION,
        n_epochs=100,
        learning_rate=0.1,
        early_stopping_rounds=20,
        random_seed=1,
    )

    models = {
        "PyTorch NN": pytorch_nn,
        "Flexible NN": flexible_nn,
        "Probabilistic Reg": probabilistic_regression,
        "CatBoost": catboost_model,
        "LightGBM": lightgbm_model,
        "Linear Regression": linear_regression_model,
        "Normal Equation Linear Regression": normal_equation_linear_regression_model,
        "PyTorch Linear Regression": pytorch_linear_regression_model,
        "XGBoost": xgboost_model,
    }

    results = {}

    # --- Training and Evaluation ---

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(x_train_scaled, y_train_scaled)
        y_pred_scaled = model.evaluate(x_test_scaled, y_test_scaled, save_path=f"{name}_metrics")
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

        mse = mean_squared_error(y_test, y_pred)
        results[name] = {"mse": mse, "predictions": y_pred}
        print(f"{name} MSE: {mse:.4f}")
        print(f"{name} Number of Parameters: {model.get_num_parameters()}")

    # --- Visualization ---

    plot_results_bokeh(x_train, y_train, x_test, y_test, y_true_test, results)


if __name__ == "__main__":
    run_experiment()
