"""Example script for demonstrating AutoML capabilities with noisy data."""

import numpy as np
from bokeh.models import Legend, LegendItem
from bokeh.palettes import Category10
from bokeh.plotting import figure, show
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from automl_package.enums import LearnedRegularizationType, TaskType
from automl_package.models.flexible_neural_network import FlexibleHiddenLayersNN
from automl_package.models.neural_network import PyTorchNeuralNetwork
from automl_package.models.probabilistic_regression import ProbabilisticRegressionModel


def generate_noisy_data(n_samples=1000, noise_level=0.5):
    """Generates synthetic data with a non-linear relationship and significant noise."""
    np.random.seed(42)
    X = np.random.rand(n_samples, 1) * 10
    # A non-linear relationship
    y_true = np.sin(X).ravel() + X.ravel() * 0.5
    # Add significant noise
    noise = np.random.randn(n_samples) * noise_level
    y = y_true + noise
    return X, y, y_true


def plot_results_bokeh(X_train, y_train, X_test, y_test, y_true_test, results):
    """Generates an interactive Bokeh plot for model comparison."""
    p = figure(width=900, height=600, title="Model Comparison on Noisy Non-Linear Data", x_axis_label="Feature", y_axis_label="Target")

    # Training Data
    train_data_glyph = p.scatter(X_train.flatten(), y_train, color="gray", alpha=0.5, size=5)

    # Test Data
    test_data_glyph = p.scatter(X_test.flatten(), y_test, color="blue", alpha=0.5, size=5)

    # True Relationship
    sorted_indices_true = np.argsort(X_test, axis=0).flatten()
    true_rel_glyph = p.line(X_test[sorted_indices_true].flatten(), y_true_test[sorted_indices_true], line_dash="dashed", line_color="black", line_width=2)

    # The Category10 palette supports up to 10 distinct colors.
    # We will use the largest palette and cycle through the colors if there are more than 10 models.
    colors = Category10[10]
    legend_items = [
        LegendItem(label="Training Data", renderers=[train_data_glyph]),
        LegendItem(label="Test Data", renderers=[test_data_glyph]),
        LegendItem(label="True Relationship", renderers=[true_rel_glyph]),
    ]

    for i, (name, result) in enumerate(results.items()):
        sorted_indices = np.argsort(X_test, axis=0).flatten()
        line_glyph = p.line(X_test[sorted_indices].flatten(), result["predictions"][sorted_indices], line_color=colors[i % len(colors)], line_width=2)
        legend_items.append(LegendItem(label=f'{name} (MSE: {result["mse"]:.4f})', renderers=[line_glyph]))

    # Create a custom legend to allow toggling
    legend = Legend(items=legend_items, location="top_left", click_policy="hide", background_fill_alpha=0.5, label_text_font_size="8pt")
    p.add_layout(legend)

    show(p)


def run_experiment():
    """Runs the full experiment to compare model performances."""
    X, y, y_true = generate_noisy_data()
    X_train, X_test, y_train, y_test, y_true_train, y_true_test = train_test_split(X, y, y_true, test_size=0.2, random_state=42)

    # Scale features and target
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

    # --- Model Initialization ---

    # PyTorch Neural Network with learned lambdas
    pytorch_nn_learned_lambdas = PyTorchNeuralNetwork(
        input_size=X_train.shape[1],
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

    # PyTorch Neural Network with fixed lambdas
    pytorch_nn_fixed_lambdas = PyTorchNeuralNetwork(
        input_size=X_train.shape[1],
        hidden_layers=2,
        hidden_size=64,
        task_type=TaskType.REGRESSION,
        n_epochs=100,
        learning_rate=0.01,
        early_stopping_rounds=20,
        validation_fraction=0.2,
        l1_lambda=0.001,
        l2_lambda=0.001,
        learn_regularization_lambdas=False,
        random_seed=1,
    )

    # Flexible Neural Network with learned lambdas
    flexible_nn_learned_lambdas = FlexibleHiddenLayersNN(
        input_size=X_train.shape[1],
        max_hidden_layers=5,  # Increased from 2
        hidden_size=64,
        task_type=TaskType.REGRESSION,
        n_epochs=200,
        learning_rate=0.01,
        early_stopping_rounds=20,
        validation_fraction=0.2,
        gumbel_tau=0.1,  # Initial higher value for annealing
        n_predictor_layers=2,  # Increased capacity
        learn_regularization_lambdas=True,
        learned_regularization_type=LearnedRegularizationType.L1_L2,
        lambda_learning_rate=0.001,
        gumbel_tau_anneal_rate=0.9,  # More aggressive annealing
        n_predictor_learning_rate=0.01,
    )

    # Flexible Neural Network with fixed lambdas
    flexible_nn_fixed_lambdas = FlexibleHiddenLayersNN(
        input_size=X_train.shape[1],
        max_hidden_layers=5,
        hidden_size=64,
        task_type=TaskType.REGRESSION,
        n_epochs=200,
        learning_rate=0.01,
        early_stopping_rounds=20,
        validation_fraction=0.2,
        gumbel_tau=0.1,  # Initial higher value for annealing
        n_predictor_layers=2,  # Increased capacity
        learn_regularization_lambdas=True,
        learned_regularization_type=LearnedRegularizationType.L1_L2,
        lambda_learning_rate=0.001,
        gumbel_tau_anneal_rate=0.9,  # More aggressive annealing
        n_predictor_learning_rate=0.01,
    )

    # Probabilistic Regression Model with learned lambdas
    probabilistic_regression_learned_lambdas = ProbabilisticRegressionModel(
        input_size=X_train.shape[1],
        n_classes=3,
        n_epochs=200,
        early_stopping_rounds=20,
        validation_fraction=0.2,
        learn_regularization_lambdas=True,
        learned_regularization_type=LearnedRegularizationType.L1_L2,
        lambda_learning_rate=0.001,
        random_seed=1,
    )

    # Probabilistic Regression Model with fixed lambdas
    probabilistic_regression_fixed_lambdas = ProbabilisticRegressionModel(
        input_size=X_train.shape[1],
        n_classes=3,
        n_epochs=200,
        early_stopping_rounds=20,
        validation_fraction=0.2,
        l1_lambda=0.001,
        l2_lambda=0.001,
        learn_regularization_lambdas=False,
        random_seed=1,
    )

    models = {
        "PyTorch NN (Learned Lambdas)": pytorch_nn_learned_lambdas,
        "PyTorch NN (Fixed Lambdas)": pytorch_nn_fixed_lambdas,
        "Flexible NN (Learned Lambdas)": flexible_nn_learned_lambdas,
        "Flexible NN (Fixed Lambdas)": flexible_nn_fixed_lambdas,
        "Probabilistic Reg (Learned Lambdas)": probabilistic_regression_learned_lambdas,
        "Probabilistic Reg (Fixed Lambdas)": probabilistic_regression_fixed_lambdas,
    }

    results = {}

    # --- Training and Evaluation ---

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_scaled, y_train_scaled)
        y_pred_scaled = model.evaluate(X_test_scaled, y_test_scaled, save_path=f"{name}_metrics")
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

        mse = mean_squared_error(y_test, y_pred)
        results[name] = {"mse": mse, "predictions": y_pred}
        print(f"{name} MSE: {mse:.4f}")
        print(f"{name} Number of Parameters: {model.get_num_parameters()}")

    # --- Visualization ---

    plot_results_bokeh(X_train, y_train, X_test, y_test, y_true_test, results)


if __name__ == "__main__":
    run_experiment()
