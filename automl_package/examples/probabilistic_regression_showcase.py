"""Showcase for Probabilistic and Classifier Regression models."""

import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from automl_package.enums import (
    FunctionType,
    MapperType,
    NClassesSelectionMethod,
    RegressionStrategy,
    TaskType,
    UncertaintyMethod,
)
from automl_package.logger import logger
from automl_package.models.classifier_regression import ClassifierRegressionModel
from automl_package.models.neural_network import PyTorchNeuralNetwork
from automl_package.models.probabilistic_regression import ProbabilisticRegressionModel
from automl_package.utils.metrics import calculate_nll


# --- 1. Enhanced Synthetic Data Generation ---
def generate_synthetic_data(
    n_samples: int = 1000,
    noise_level: float = 0.1,
    function_type: str = "sin",
    random_seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generates synthetic data for regression tasks with controllable noise.

    Args:
        n_samples (int): Number of samples to generate.
        noise_level (float): Standard deviation of the Gaussian noise added to the true function.
        function_type (str): Type of underlying true function ("sin", "polynomial", or "linear").
        random_seed (int, optional): Random seed for reproducibility. Defaults to None.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            - X (np.ndarray): Input features.
            - y (np.ndarray): Noisy target values.
            - y_true (np.ndarray): True (noiseless) target values.
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    x_data = np.random.rand(n_samples, 1) * 10 - 5  # X from -5 to 5

    if function_type == "sin":
        y_true = np.sin(x_data) * 2 + 0.5 * x_data  # A non-linear function
    elif function_type == "polynomial":
        y_true = 0.1 * x_data**3 - 0.5 * x_data**2 + 2 * x_data + 5  # A polynomial function
    else:  # linear
        y_true = 2 * x_data + 1

    noise = np.random.normal(0, noise_level, n_samples).reshape(-1, 1)
    y_noisy = y_true + noise
    return x_data, y_noisy, y_true


# --- Baseline Classical Regression Model ---
class SimpleNN(nn.Module):
    """A simple feedforward neural network for regression."""

    def __init__(self, input_size: int, output_size: int, activation: nn.Module) -> None:
        """Initializes the SimpleNN.

        Args:
            input_size (int): Number of input features.
            output_size (int): Number of output features (1 for regression).
            activation (nn.Module): The activation function to use.
        """
        super().__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.activation = activation()
        self.fc2 = nn.Linear(64, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the SimpleNN."""
        return self.fc2(self.activation(self.fc1(x)))


def train_and_evaluate_simple_nn(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    input_size: int,
    output_size: int,
    activation: nn.Module,
    n_epochs: int = 100,
    lr: float = 0.01,
) -> tuple[float, np.ndarray]:
    """Trains and evaluates a SimpleNN model.

    Args:
        x_train (np.ndarray): Training features.
        y_train (np.ndarray): Training targets.
        x_test (np.ndarray): Test features.
        y_test (np.ndarray): Test targets.
        input_size (int): Number of input features.
        output_size (int): Number of output features.
        activation (nn.Module): The activation function to use.
        n_epochs (int): Number of training epochs.
        lr (float): Learning rate for the optimizer.

    Returns:
        Tuple[float, np.ndarray]:
            - mse (float): Mean Squared Error on the test set.
            - y_pred (np.ndarray): Predictions on the test set.
    """
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    # y_test_tensor is not directly used in the training loop, only for final MSE calculation

    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    model = SimpleNN(input_size, output_size, activation)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for _epoch in range(n_epochs):  # Renamed to _epoch as it's not used
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        y_pred_tensor = model(x_test_tensor)
        y_pred = y_pred_tensor.numpy()
        mse = mean_squared_error(y_test, y_pred)
    return mse, y_pred


# --- Main Showcase Function ---
def run_probabilistic_regression_showcase() -> None:
    """Runs the showcase demonstrating probabilistic and classifier regression."""
    output_dir = "probabilistic_regression_showcase_results"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    noise_levels = [0.05, 0.5, 1.5]  # Low, Medium, High noise
    n_samples = 1000
    function_type = FunctionType.SIN  # Using sine function for non-linearity

    results = {}
    all_predictions = {}

    random_seed = 42
    for noise_level in noise_levels:
        logger.info(f"\n--- Running experiments for Noise Level: {noise_level} ---")
        x_data, y_data, y_true = generate_synthetic_data(
            n_samples=n_samples,
            noise_level=noise_level,
            function_type=function_type,
            random_seed=random_seed,
        )
        x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=random_seed)

        input_size = x_train.shape[1]
        output_size = y_train.shape[1]

        # --- 1. Simple NN Baseline ---
        logger.info("Training Simple NN Baseline...")
        mse_simple_nn, y_pred_simple_nn = train_and_evaluate_simple_nn(x_train, y_train, x_test, y_test, input_size, output_size, nn.ReLU)
        results[f"Noise_{noise_level:.2f}_SimpleNN"] = mse_simple_nn
        all_predictions[f"Noise_{noise_level:.2f}_SimpleNN"] = y_pred_simple_nn
        logger.info(f"Simple NN MSE: {mse_simple_nn:.4f}")

        # --- 2. Classifier Regression ---
        logger.info("Training Classifier Regression Models...")
        n_classes_range = range(2, 8)  # From 2 to 7 classes
        mapper_types = [MapperType.LOOKUP_MEDIAN, MapperType.LINEAR]

        best_cr_mse = float("inf")
        best_cr_n_classes = -1
        best_cr_mapper_type = None

        for n_cls in n_classes_range:
            for m_type in mapper_types:
                logger.info(f"  Training Classifier Regression (n_classes={n_cls}, mapper={m_type.label})...")
                cr_model = ClassifierRegressionModel(
                    n_classes=n_cls,
                    base_classifier_class=PyTorchNeuralNetwork,
                    base_classifier_params={
                        "input_size": input_size,
                        "output_size": n_cls,
                        "task_type": TaskType.CLASSIFICATION,
                    },
                    mapper_type=m_type,
                    n_epochs=100,
                    learning_rate=0.01,
                    early_stopping_rounds=10,
                    validation_fraction=0.1,
                    random_seed=random_seed,
                )
                cr_model.fit(x_train, y_train)
                y_pred_cr = cr_model.predict(x_test)
                mse_cr = mean_squared_error(y_test, y_pred_cr)
                results[f"Noise_{noise_level:.2f}_CR_n{n_cls}_m{m_type.label}"] = mse_cr
                all_predictions[f"Noise_{noise_level:.2f}_CR_n{n_cls}_m{m_type.label}"] = y_pred_cr
                logger.info(f"    MSE: {mse_cr:.4f}")

                if mse_cr < best_cr_mse:
                    best_cr_mse = mse_cr
                    best_cr_n_classes = n_cls
                    best_cr_mapper_type = m_type

        logger.info(f"  Best Classifier Regression for Noise {noise_level:.2f}: n_classes={best_cr_n_classes}, mapper={best_cr_mapper_type.label}, MSE={best_cr_mse:.4f}")

        # --- 3. Probabilistic Regression ---
        logger.info("Training Probabilistic Regression Models...")
        regression_strategies = [
            RegressionStrategy.SEPARATE_HEADS,
            RegressionStrategy.SINGLE_HEAD_FINAL_OUTPUT,
        ]
        n_classes_range_pr = range(2, 8)  # From 2 to 7 classes

        best_pr_mse = float("inf")
        best_pr_n_classes = -1
        best_pr_reg_strategy = None

        for n_cls in n_classes_range_pr:
            for reg_strategy in regression_strategies:
                logger.info(f"  Training Probabilistic Regression (n_classes={n_cls}, strategy={reg_strategy.value})...")
                pr_model = ProbabilisticRegressionModel(
                    input_size=input_size,
                    n_classes=n_cls,
                    uncertainty_method=UncertaintyMethod.PROBABILISTIC,
                    n_classes_selection_method=NClassesSelectionMethod.NONE,
                    regression_strategy=reg_strategy,
                    base_classifier_params={"hidden_layers": 1, "hidden_size": 64},
                    regression_head_params={"hidden_layers": 0, "hidden_size": 32},
                    n_epochs=100,
                    learning_rate=0.01,
                    early_stopping_rounds=10,
                    validation_fraction=0.1,
                    random_seed=random_seed,
                )
                pr_model.fit(x_train, y_train)
                y_pred_pr = pr_model.predict(x_test)
                mse_pr = mean_squared_error(y_test, y_pred_pr)

                y_pred_mean_pr = y_pred_pr
                y_pred_std_pr = pr_model.predict_uncertainty(x_test)
                y_test_flat = y_test.flatten()
                nll = calculate_nll(y_test_flat, y_pred_mean_pr, y_pred_std_pr)

                results[f"Noise_{noise_level:.2f}_PR_n{n_cls}_s{reg_strategy.value}_MSE"] = mse_pr
                results[f"Noise_{noise_level:.2f}_PR_n{n_cls}_s{reg_strategy.value}_NLL"] = nll
                all_predictions[f"Noise_{noise_level:.2f}_PR_n{n_cls}_s{reg_strategy.value}_Pred"] = y_pred_pr
                logger.info(f"    MSE: {mse_pr:.4f}, NLL: {nll:.4f}")

                if mse_pr < best_pr_mse:
                    best_pr_mse = mse_pr
                    best_pr_n_classes = n_cls
                    best_pr_reg_strategy = reg_strategy

        logger.info(f"  Best Probabilistic Regression for Noise {noise_level:.2f}: n_classes={best_pr_n_classes}, strategy={best_pr_reg_strategy.value}, MSE={best_pr_mse:.4f}")

    # --- Reporting and Visualization ---
    logger.info("\n--- Overall Results Summary ---")
    results_df = pd.DataFrame.from_dict(results, orient="index", columns=["Value"])
    print(results_df)

    # Plotting predictions for selected noise levels and models
    for noise_level in noise_levels:
        plt.figure(figsize=(12, 8))
        plt.scatter(x_data.flatten(), y_data.flatten(), alpha=0.3, label="Noisy Data")
        plt.plot(
            x_data.flatten(),
            y_true.flatten(),
            color="black",
            linestyle="--",
            label="True Function",
        )

        # Sort x_test for plotting smooth lines
        sort_indices = np.argsort(x_test.flatten())
        x_test_sorted = x_test.flatten()[sort_indices]

        # Plot Simple NN
        y_pred_simple_nn_sorted = all_predictions[f"Noise_{noise_level:.2f}_SimpleNN"][sort_indices]
        plt.plot(
            x_test_sorted,
            y_pred_simple_nn_sorted,
            label=f"Simple NN (MSE: {results[f'Noise_{noise_level:.2f}_SimpleNN']:.4f})",
            color="red",
        )

        # Plot best Classifier Regression
        best_cr_key = None
        min_cr_mse = float("inf")
        for n_cls in n_classes_range:
            for m_type in mapper_types:
                key = f"Noise_{noise_level:.2f}_CR_n{n_cls}_m{m_type.label}"
                if key in results and results[key] < min_cr_mse:
                    min_cr_mse = results[key]
                    best_cr_key = key

        if best_cr_key:
            y_pred_best_cr_sorted = all_predictions[best_cr_key][sort_indices]
            plt.plot(
                x_test_sorted,
                y_pred_best_cr_sorted,
                label=f"Best CR (n_classes={best_cr_key.split('_')[3]}, mapper={best_cr_key.split('_')[4]}, MSE: {min_cr_mse:.4f})",
                color="green",
            )

        # Plot best Probabilistic Regression
        best_pr_key = None
        min_pr_mse = float("inf")
        for n_cls in n_classes_range_pr:
            for reg_strategy in regression_strategies:
                key = f"Noise_{noise_level:.2f}_PR_n{n_cls}_s{reg_strategy.value}_MSE"
                if key in results and results[key] < min_pr_mse:
                    min_pr_mse = results[key]
                    best_pr_key = key.replace("_MSE", "")

        if best_pr_key:
            pr_key_pred = f"{best_pr_key}_Pred"
            pr_key_nll = f"{best_pr_key}_NLL"
            if pr_key_pred in all_predictions:
                y_pred_pr_data = all_predictions[pr_key_pred]
                y_pred_mean_pr_sorted = y_pred_pr_data[sort_indices, 0]
                y_pred_std_pr_sorted = np.sqrt(y_pred_pr_data[sort_indices, 1])

                plt.plot(
                    x_test_sorted,
                    y_pred_mean_pr_sorted,
                    label=f"Best PR (n_classes={best_pr_key.split('_')[3]}, strategy={best_pr_key.split('_')[4]}, MSE: {min_pr_mse:.4f}, NLL: {results[pr_key_nll]:.4f})",
                    color="blue",
                )
                plt.fill_between(
                    x_test_sorted,
                    y_pred_mean_pr_sorted - y_pred_std_pr_sorted,
                    y_pred_mean_pr_sorted + y_pred_std_pr_sorted,
                    color="blue",
                    alpha=0.1,
                )

        plt.title(f"Model Predictions at Noise Level: {noise_level}")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f"predictions_noise_{noise_level:.2f}.png"))
        plt.close()

    logger.info("\nShowcase complete. Check the 'probabilistic_regression_showcase_results' directory for plots and detailed metrics.")


if __name__ == "__main__":
    run_probabilistic_regression_showcase()
