# ruff: noqa: ERA001
"""Showcase of Overfitting in Various Models.

This script demonstrates how different models handle a simple dataset,
highlighting the risk of overfitting with more complex models if not
properly configured. It compares three models:
1. A standard neural network (PyTorchNeuralNetwork).
2. A regression model built from a classifier (ClassifierRegressionModel).
3. A probabilistic regression model that estimates a distribution (ProbabilisticRegressionModel).

The script will:
1. Generate a simple synthetic dataset (linear function with noise).
2. Train the three models on this data.
3. Evaluate the models by calculating Mean Squared Error (MSE).
4. Generate and save a plot comparing the predictions of all models against
   the true underlying function.
"""

import json
import logging
import os
import shutil

import numpy as np
import pandas as pd
from bokeh.io import output_file, save
from bokeh.palettes import Category10
from bokeh.plotting import figure
from models.classifier_regression import ClassifierRegressionModel
from models.neural_network import PyTorchNeuralNetwork
from sklearn.metrics import mean_squared_error

from automl_package.enums import DataSplitStrategy, MapperType, RegressionStrategy, TaskType, UncertaintyMethod, ProbabilisticRegressionOptimizationStrategy
from automl_package.models.base import BaseModel
from automl_package.models.catboost_model import CatBoostModel
from automl_package.models.lightgbm_model import LightGBMModel
from automl_package.models.linear_regression import LinearRegressionModel
from automl_package.models.normal_equation_linear_regression import NormalEquationLinearRegression
from automl_package.models.probabilistic_regression import ProbabilisticRegressionModel
from automl_package.models.pytorch_linear_regression import PyTorchLinearRegression
from automl_package.models.xgboost_model import XGBoostModel
from automl_package.utils.data_handler import DataHandler

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def create_price_delta_data(
    n_samples: int = 1000,
    n_features: int = 2,
    n_noise_features: int = 8,
    signal_strength: float = 1.0,
    noise_level: float = 0.5,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Creates a synthetic dataset based on price deltas with noise features.

    Args:
        n_samples (int): The number of data points to generate.
        n_features (int): The number of independent time series to use for signal features.
        n_noise_features (int): The number of independent time series to use as noise features.
        signal_strength (float): A multiplier for the signal part of the target.
        noise_level (float): The standard deviation of the Gaussian noise added to the target.
        seed (int): The random seed for reproducibility.

    Returns:
        tuple[pd.DataFrame, pd.Series, pd.Series]: A tuple containing:
            - The full feature matrix (signal + noise features) as a DataFrame.
            - The noisy target variable as a Series.
            - The true, noiseless target variable as a Series.
    """
    np.random.seed(seed)
    sinusoid_amplitude = 1.0
    sinusoid_period = 100
    sinusoid_variation_factor = 0.1

    total_features = n_features + n_noise_features
    # 1. Generate independent time series (prices) using a random walk
    prices = np.zeros((n_samples, total_features))
    for t in range(1, n_samples):
        # First series has a sinusoidal component
        prices[t, 0] = (
            prices[t - 1, 0]
            + np.random.randn() * 0.1
            + np.sin(
                t
                / (
                    sinusoid_period
                    + np.random.uniform(
                        -sinusoid_period * sinusoid_variation_factor,
                        sinusoid_period * sinusoid_variation_factor,
                    )
                )
            )
            * (
                sinusoid_amplitude
                + np.random.uniform(
                    -sinusoid_amplitude * sinusoid_variation_factor,
                    sinusoid_amplitude * sinusoid_variation_factor,
                )
            )
        )
        # Other series are simple random walks
        for j in range(1, total_features):
            prices[t, j] = prices[t - 1, j] + np.random.randn() * 0.1

    # 2. Create features from the deltas of all series
    all_feature_deltas = np.diff(prices, axis=0)

    # 3. Generate the target delta based on a weighted sum of signal feature deltas only
    signal_feature_deltas = all_feature_deltas[:, :n_features]
    weights = np.random.randn(n_features)
    weighted_sum = np.sum(signal_feature_deltas * weights, axis=1) * signal_strength

    # 4. Define the true target as the weighted sum (linear relationship)
    y_true = weighted_sum

    # 5. Add Gaussian noise to the target
    noise = np.random.normal(0, noise_level, n_samples - 1)
    y_noisy = y_true + noise

    # Create a timestamp column
    timestamps = pd.to_datetime(pd.date_range(start="2023-01-01", periods=n_samples - 1, freq="D"))

    # The first sample is lost due to diff, so we return n_samples - 1
    df = pd.DataFrame(all_feature_deltas, columns=[f"feature_{i}" for i in range(total_features)])
    df["timestamp"] = timestamps
    return df, pd.Series(y_noisy), pd.Series(y_true)


def plot_time_series_results(y_data: np.ndarray, y_true: np.ndarray, predictions: dict, results: dict, output_path: str) -> None:
    """Generates and saves a Bokeh plot comparing model predictions for time series."""
    p = figure(width=1200, height=700, title="Overfitting Showcase: Model Comparison (Test Set)", x_axis_label="Time Step", y_axis_label="Output (y)")

    time_steps = np.arange(len(y_data))

    # Plot noisy data points
    p.scatter(time_steps, y_data, color="gray", alpha=0.6, size=8, legend_label="Test Data")

    # Plot the true underlying function
    p.line(time_steps, y_true, line_dash="dashed", line_color="black", line_width=3, legend_label="True Function (Noiseless Target)")

    colors = Category10[10]

    # Plot predictions for each model
    for i, (name, y_pred) in enumerate(predictions.items()):
        color = colors[i % len(colors)]
        y_pred_flat = y_pred.flatten()
        p.line(time_steps, y_pred_flat, line_color=color, line_width=2.5, legend_label=f"{name} (MSE: {results[name]['rmse']**2:.4f})")

    p.legend.location = "top_right"
    p.legend.click_policy = "hide"
    p.legend.background_fill_alpha = 0.5
    p.legend.label_text_font_size = "8pt"

    output_file(output_path)
    save(p)
    logging.info(f"Saved comparison plot to {output_path}")


def run_showcase() -> None:
    """Runs the full overfitting showcase."""
    n_samples = 2000
    user_defined_n_classes = 3
    early_stopping_rounds = 50
    validation_fraction = 0.2
    test_fraction = 0.2
    cv_folds = None
    n_epochs = 500
    hidden_layers = 2
    hidden_size = 64
    learning_rate = 0.005
    feature_selection_threshold = None
    uncertainty_method = UncertaintyMethod.PROBABILISTIC
    optimize_hyperparameters = False
    n_trials = 50
    random_seed = 42

    output_dir = "overfitting_showcase_results"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    x, y, y_true = create_price_delta_data(n_samples=n_samples, n_features=2, n_noise_features=8, noise_level=0.1, seed=random_seed)

    timestamps = x["timestamp"].values
    x_features = x.drop(columns=["timestamp"])

    # Time-ordered split
    split_index = int(len(x_features) * (1 - test_fraction))
    x_train, x_test = x_features[:split_index], x_features[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    _, y_true_test = y_true[:split_index], y_true[split_index:]
    timestamps_train, _ = timestamps[:split_index], timestamps[split_index:]

    data_handler = DataHandler()
    x_train_scaled, y_train_scaled = data_handler.fit_transform(x_train.values, y_train.values)
    x_test_scaled, y_test_scaled = data_handler.transform(x_test.values, y_test.values)

    models_to_test: dict[str, BaseModel] = {
        "PyTorch_NN_Random_Split": PyTorchNeuralNetwork(
            input_size=x_train_scaled.shape[1],
            hidden_layers=hidden_layers,
            hidden_size=hidden_size,
            n_epochs=n_epochs,
            learning_rate=learning_rate,
            uncertainty_method=uncertainty_method,
            early_stopping_rounds=early_stopping_rounds,
            validation_fraction=validation_fraction,
            test_fraction=0.0,
            cv_folds=cv_folds,
            split_strategy=DataSplitStrategy.RANDOM,
            feature_selection_threshold=feature_selection_threshold,
            optimize_hyperparameters=optimize_hyperparameters,
            n_trials=n_trials,
            random_seed=random_seed,
            output_dir=os.path.join(output_dir, "PyTorch_NN_Random_Split"),
        ),
        "PyTorch_NN_Time_Ordered_Split": PyTorchNeuralNetwork(
            input_size=x_train_scaled.shape[1],
            hidden_layers=hidden_layers,
            hidden_size=hidden_size,
            n_epochs=n_epochs,
            learning_rate=learning_rate,
            uncertainty_method=uncertainty_method,
            early_stopping_rounds=early_stopping_rounds,
            validation_fraction=validation_fraction,
            test_fraction=0.0,
            cv_folds=cv_folds,
            split_strategy=DataSplitStrategy.TIME_ORDERED,
            feature_selection_threshold=feature_selection_threshold,
            optimize_hyperparameters=optimize_hyperparameters,
            n_trials=n_trials,
            random_seed=random_seed,
            output_dir=os.path.join(output_dir, "PyTorch_NN_Time_Ordered_Split"),
        ),
        "Classifier_Regression": ClassifierRegressionModel(
            base_classifier_class=PyTorchNeuralNetwork,
            base_classifier_params={
                "input_size": x_train_scaled.shape[1],
                "output_size": user_defined_n_classes,
                "n_epochs": n_epochs,
                "hidden_layers": hidden_layers,
                "hidden_size": hidden_size,
                "learning_rate": learning_rate,
                "random_seed": random_seed,
            },
            n_classes=user_defined_n_classes,
            early_stopping_rounds=early_stopping_rounds,
            validation_fraction=validation_fraction,
            test_fraction=0.0,
            cv_folds=cv_folds,
            uncertainty_method=uncertainty_method,
            split_strategy=DataSplitStrategy.RANDOM,
            mapper_type=MapperType.LINEAR,
            auto_include_nn_mappers=True,
            feature_selection_threshold=feature_selection_threshold,
            calculate_feature_importance=False,
            optimize_hyperparameters=optimize_hyperparameters,
            n_trials=n_trials,
            random_seed=random_seed,
            output_dir=os.path.join(output_dir, "Classifier_Regression"),
        ),
        "CatBoost": CatBoostModel(
            n_estimators=n_epochs,
            learning_rate=learning_rate,
            uncertainty_method=uncertainty_method,
            early_stopping_rounds=early_stopping_rounds,
            validation_fraction=validation_fraction,
            test_fraction=0.0,
            cv_folds=cv_folds,
            feature_selection_threshold=feature_selection_threshold,
            optimize_hyperparameters=optimize_hyperparameters,
            n_trials=n_trials,
            random_seed=random_seed,
            output_dir=os.path.join(output_dir, "CatBoost"),
            task_type=TaskType.REGRESSION,
        ),
        "LightGBM": LightGBMModel(
            n_estimators=n_epochs,
            learning_rate=learning_rate,
            uncertainty_method=uncertainty_method,
            early_stopping_rounds=early_stopping_rounds,
            validation_fraction=validation_fraction,
            test_fraction=0.0,
            cv_folds=cv_folds,
            feature_selection_threshold=feature_selection_threshold,
            optimize_hyperparameters=optimize_hyperparameters,
            n_trials=n_trials,
            random_seed=random_seed,
            output_dir=os.path.join(output_dir, "LightGBM"),
            task_type=TaskType.REGRESSION,
        ),
        "LinearRegression": LinearRegressionModel(
            early_stopping_rounds=early_stopping_rounds,
            validation_fraction=validation_fraction,
            test_fraction=0.0,
            feature_selection_threshold=feature_selection_threshold,
            optimize_hyperparameters=optimize_hyperparameters,
            n_trials=n_trials,
            cv_folds=cv_folds,
            uncertainty_method=uncertainty_method,
            output_dir=os.path.join(output_dir, "LinearRegression"),
        ),
        "NormalEquationLinearRegression": NormalEquationLinearRegression(
            feature_selection_threshold=feature_selection_threshold,
            optimize_hyperparameters=optimize_hyperparameters,
            n_trials=n_trials,
            uncertainty_method=uncertainty_method,
            output_dir=os.path.join(output_dir, "NormalEquationLinearRegression"),
        ),
        "PyTorchLinearRegression": PyTorchLinearRegression(
            input_size=x_train_scaled.shape[1],
            n_epochs=n_epochs,
            learning_rate=learning_rate,
            uncertainty_method=uncertainty_method,
            early_stopping_rounds=early_stopping_rounds,
            validation_fraction=validation_fraction,
            test_fraction=0.0,
            cv_folds=cv_folds,
            feature_selection_threshold=feature_selection_threshold,
            optimize_hyperparameters=optimize_hyperparameters,
            n_trials=n_trials,
            random_seed=random_seed,
            output_dir=os.path.join(output_dir, "PyTorchLinearRegression"),
        ),
        "XGBoost": XGBoostModel(
            n_estimators=n_epochs,
            learning_rate=learning_rate,
            uncertainty_method=uncertainty_method,
            early_stopping_rounds=early_stopping_rounds,
            validation_fraction=validation_fraction,
            test_fraction=0.0,
            cv_folds=cv_folds,
            random_seed=random_seed,
            feature_selection_threshold=feature_selection_threshold,
            optimize_hyperparameters=optimize_hyperparameters,
            n_trials=n_trials,
            output_dir=os.path.join(output_dir, "XGBoost"),
            task_type=TaskType.REGRESSION,
        ),
        # "FlexibleNeuralNetwork": FlexibleNeuralNetwork(
        #     input_size=x_train_scaled.shape[1],
        #     output_size=1,
        #     n_epochs=n_epochs,
        #     learning_rate=learning_rate,
        #     early_stopping_rounds=early_stopping_rounds,
        #     validation_fraction=validation_fraction,
        #     feature_selection_threshold=feature_selection_threshold,
        #     random_seed=random_seed,
        #     output_dir=os.path.join(output_dir, "FlexibleNeuralNetwork"),
        # ),
        # "IndependentWeightsFlexibleNeuralNetwork": IndependentWeightsFlexibleNN(
        #     input_size=x_train_scaled.shape[1],
        #     output_size=1,
        #     n_epochs=n_epochs,
        #     learning_rate=learning_rate,
        #     early_stopping_rounds=early_stopping_rounds,
        #     validation_fraction=validation_fraction,
        #     feature_selection_threshold=feature_selection_threshold,
        #     random_seed=random_seed,
        #     output_dir=os.path.join(output_dir, "IndependentWeightsFlexibleNeuralNetwork"),
        # ),
    }

    for strategy in RegressionStrategy:
        models_to_test[f"Probabilistic_Regression_{strategy.value}"] = ProbabilisticRegressionModel(
            input_size=x_train_scaled.shape[1],
            n_classes=user_defined_n_classes,
            regression_strategy=strategy,
            base_classifier_params={"hidden_layers": hidden_layers, "hidden_size": hidden_size, "random_seed": random_seed},
            n_epochs=n_epochs,
            learning_rate=learning_rate,
            uncertainty_method=uncertainty_method,
            early_stopping_rounds=early_stopping_rounds,
            validation_fraction=validation_fraction,
            test_fraction=0.0,
            cv_folds=cv_folds,
            split_strategy=DataSplitStrategy.RANDOM,
            feature_selection_threshold=feature_selection_threshold,
            optimize_hyperparameters=optimize_hyperparameters,
            optimization_strategy=ProbabilisticRegressionOptimizationStrategy.GRADIENT_STOP,
            n_trials=n_trials,
            add_classification_loss=True,
            random_seed=random_seed,
            output_dir=os.path.join(output_dir, f"Probabilistic_Regression_{strategy.value}"),
        )

    results = {}
    predictions = {}
    uncertainties = {}

    logging.info("--- Running Model Training and Evaluation ---")
    for name, model in models_to_test.items():
        if not name.startswith("Probabilistic_Regression_"):
            continue
        logging.info(f"Training {name}...")
        model_output_dir = os.path.join(output_dir, name)
        os.makedirs(model_output_dir, exist_ok=True)

        model.fit(x_train_scaled, y_train_scaled, timestamps=timestamps_train)
        y_pred_scaled, y_std_scaled = model.evaluate(x_test_scaled, y_test_scaled, "test", model_output_dir)

        y_pred = data_handler.inverse_transform_y(y_pred_scaled)
        y_std = y_std_scaled * data_handler.y_scaler.scale_ if y_std_scaled is not None else None

        predictions[name] = y_pred
        uncertainties[name] = y_std

        # Read metrics from the file saved by the evaluate method
        with open(os.path.join(model_output_dir, "test_metrics.json")) as f:
            results[name] = json.load(f)

        unscaled_mse = mean_squared_error(y_test, y_pred)
        results[name]["unscaled_mse"] = unscaled_mse
        logging.info(f"  -> Test MSE (scaled): {results[name]['rmse']**2:.4f}")
        logging.info(f"  -> Test MSE (unscaled): {unscaled_mse:.4f}")
        if "nll" in results[name]:
            logging.info(f"  -> Test NLL: {results[name]['nll']:.4f}")

    logging.info("\n--- Model Performance Summary (Test Set) ---")
    # Convert results dict to DataFrame for display
    results_for_df = {}
    for name, metrics in results.items():
        results_for_df[name] = {"MSE (scaled)": metrics.get("rmse", np.nan) ** 2, "MSE (unscaled)": metrics.get("unscaled_mse", np.nan), "NLL": metrics.get("nll", np.nan)}
    results_df = pd.DataFrame.from_dict(results_for_df, orient="index").sort_values(by="MSE (unscaled)")
    logging.info(f"\n{results_df}")

    # --- Visualization ---
    plot_path = os.path.join(output_dir, "overfitting_comparison.html")
    plot_time_series_results(y_test.values, y_true_test.values, predictions, results, plot_path)

    logging.info("\nShowcase complete.")


if __name__ == "__main__":
    run_showcase()
