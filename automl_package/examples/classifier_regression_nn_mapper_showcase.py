"""Showcase for the ClassifierRegressionModel with a NeuralNetworkMapper."""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

from automl_package.enums import MapperType, TaskType
from automl_package.models.classifier_regression import ClassifierRegressionModel
from automl_package.models.neural_network import PyTorchNeuralNetwork
from automl_package.utils.metrics import Metrics


def generate_data() -> tuple[np.ndarray, np.ndarray]:
    """Generates synthetic data for regression."""
    x = np.linspace(-10, 10, 500).reshape(-1, 1)
    y = np.sin(x).ravel() + np.random.normal(0, 0.2, x.shape[0])
    return x, y


def run_showcase() -> None:
    """Runs the showcase for ClassifierRegressionModel with NeuralNetworkMapper."""
    print("--- Running ClassifierRegressionModel with NeuralNetworkMapper Showcase ---")

    # 1. Generate Data
    x, y = generate_data()
    random_seed = 42
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=random_seed)

    # 2. Configure the model
    base_classifier_class = PyTorchNeuralNetwork

    base_classifier_params = {
        "hidden_layers": 2,
        "hidden_size": 64,
        "use_batch_norm": True,
        "dropout_rate": 0.1,
        "n_epochs": 50,
        "learning_rate": 0.01,
    }

    nn_mapper_params = {
        "epochs": 100,
        "learning_rate": 0.01,
        "batch_size": 32,
        "regression_head_params": {
            "hidden_layers": 1,
            "hidden_size": 32,
            "use_batch_norm": True,
        },
    }

    # 3. Instantiate the ClassifierRegressionModel
    model = ClassifierRegressionModel(
        n_classes=10,
        base_classifier_class=base_classifier_class,
        base_classifier_params=base_classifier_params,
        mapper_type=MapperType.NN_SEPARATE_HEADS,
        nn_mapper_params=nn_mapper_params,
        early_stopping_rounds=10,
        validation_fraction=0.2,
        random_seed=random_seed,
    )

    print(f"Training model: {model.name}")

    # 4. Fit the model
    model.fit(x_train, y_train)

    # 5. Evaluate the model
    print("\n--- Evaluating on Test Set ---")
    y_pred = model.predict(x_test)

    metrics_calculator = Metrics(
        task_type=TaskType.REGRESSION,
        model_name=model.name,
        x_data=x_test,
        y_true=y_test,
        y_pred=y_pred,
    )
    metrics = metrics_calculator.calculate()
    print("Metrics:", metrics)

    # 6. Plot results
    plt.figure(figsize=(10, 6))
    plt.scatter(x_test, y_test, label="Actual Data", alpha=0.6, s=10)
    sorted_indices = np.argsort(x_test.ravel())
    plt.plot(x_test[sorted_indices], y_pred[sorted_indices], color="red", label="Predictions")
    plt.title("ClassifierRegressionModel with NN Mapper Predictions")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plot_path = "classifier_regression_nn_mapper_showcase.png"
    plt.savefig(plot_path)
    print(f"\nPlot saved to {plot_path}")


if __name__ == "__main__":
    run_showcase()
