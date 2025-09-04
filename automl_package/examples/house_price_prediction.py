import json
import logging
import os
import shutil

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from automl_package.enums import DataSplitStrategy, TaskType
from automl_package.models.base import BaseModel
from automl_package.models.catboost_model import CatBoostModel
from automl_package.models.classifier_regression import ClassifierRegressionModel
from automl_package.models.linear_regression import LinearRegressionModel
from automl_package.models.neural_network import PyTorchNeuralNetwork
from automl_package.models.normal_equation_linear_regression import NormalEquationLinearRegression
from automl_package.models.pytorch_linear_regression import PyTorchLinearRegression
from automl_package.utils.data_handler import DataHandler, create_train_val_split

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def load_house_price_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Loads the house price dataset."""
    data_path = os.path.join(os.path.dirname(__file__), "data", "FlatSaleData.csv")
    df = pd.read_csv(data_path)

    features = ["Floor", "Parking Spaces", "Size (sq ft)", "is Pan Peninsula", "is Wardian", "E14 Index Delta (Jan 2023)"]
    target = "Actual Price (GBP)"

    X = df[features]
    y = df[target]

    return X, y, df


def get_linear_model_params(model: BaseModel, feature_names: list[str], output_dir: str) -> None:
    """Gets and saves the parameters of a linear model."""
    params = {}
    internal_model = model.get_internal_model()

    # Handle ShapModel wrapper
    if hasattr(internal_model, "model"):
        original_model = internal_model.model
    else:
        original_model = internal_model

    if hasattr(original_model, "coef_") and hasattr(original_model, "intercept_"):
        params["intercept"] = np.array(original_model.intercept_).item()
        coefs = np.array(original_model.coef_).flatten()
        coefficients = {feature_names[i]: coefs[i] for i in range(len(coefs))}
        params["coefficients"] = coefficients
    else:
        logging.warning(f"Could not get parameters for model {model.name}")
        return

    logging.info(f"  -> Model Parameters: {params}")

    with open(os.path.join(output_dir, "model_parameters.json"), "w") as f:
        json.dump(params, f, indent=4, cls=NumpyEncoder)


def generate_output_file(
    model: BaseModel, data_handler: DataHandler, all_data: pd.DataFrame, indices: np.ndarray, file_path: str, feature_names: list[str]
) -> None:
    """Generates a CSV file with predictions and errors."""
    subset_data = all_data.iloc[indices].copy()
    subset_x = subset_data[feature_names]

    subset_x_scaled, _ = data_handler.transform(subset_x.values, None)

    predictions_scaled = model.predict(subset_x_scaled)
    predictions = data_handler.inverse_transform_y(predictions_scaled)

    output_df = all_data.iloc[indices].copy()
    output_df["Predicted price"] = predictions
    output_df["Prediction Error"] = output_df["Actual Price (GBP)"] - output_df["Predicted price"]
    output_df["Prediction Error (%)"] = (output_df["Prediction Error"] / output_df["Actual Price (GBP)"]) * 100

    output_df.to_csv(file_path, index=False)


def run_house_price_prediction() -> None:
    """Runs the house price prediction showcase."""
    random_seed = 42
    output_dir = "house_price_prediction_results"

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    X, y, full_df = load_house_price_data()

    # Create a separate feature set for linear models
    X_linear = X.copy()
    X_linear["Size (sq ft)^2"] = X_linear["Size (sq ft)"] ** 2

    # Data Splitting
    train_indices, val_indices, test_indices = create_train_val_split(
        x=X.values, validation_fraction=0.2, test_fraction=0.2, split_strategy=DataSplitStrategy.RANDOM, random_state=random_seed, timestamps=None
    )
    train_val_indices = np.concatenate((train_indices, val_indices))

    # Simplified hyperparameters for non-linear models
    nn_hidden_layers = 1
    nn_hidden_size = 32
    catboost_estimators = 200
    catboost_depth = 4
    early_stopping_rounds = 50

    models_to_test: dict[str, BaseModel] = {
        "LinearRegression": LinearRegressionModel(
            split_strategy=DataSplitStrategy.RANDOM,
            output_dir=os.path.join(output_dir, "LinearRegression"),
            validation_fraction=0.2,
            test_fraction=0.0,
            random_seed=random_seed,
            early_stopping_rounds=early_stopping_rounds,
        ),
        "NormalEquationLinearRegression": NormalEquationLinearRegression(
            output_dir=os.path.join(output_dir, "NormalEquationLinearRegression"),
            test_fraction=0.0,
            random_seed=random_seed,
        ),
        "PyTorchLinearRegression": PyTorchLinearRegression(
            input_size=X_linear.shape[1],
            n_epochs=500,
            learning_rate=0.01,
            split_strategy=DataSplitStrategy.RANDOM,
            output_dir=os.path.join(output_dir, "PyTorchLinearRegression"),
            validation_fraction=0.2,
            test_fraction=0.0,
            random_seed=random_seed,
            early_stopping_rounds=early_stopping_rounds,
        ),
        "PyTorch_NN": PyTorchNeuralNetwork(
            input_size=X.shape[1],
            hidden_layers=nn_hidden_layers,
            hidden_size=nn_hidden_size,
            n_epochs=500,
            split_strategy=DataSplitStrategy.RANDOM,
            output_dir=os.path.join(output_dir, "PyTorch_NN"),
            validation_fraction=0.2,
            test_fraction=0.0,
            random_seed=random_seed,
            early_stopping_rounds=early_stopping_rounds,
        ),
        "CatBoost": CatBoostModel(
            n_estimators=catboost_estimators,
            depth=catboost_depth,
            learning_rate=0.05,
            early_stopping_rounds=early_stopping_rounds,
            split_strategy=DataSplitStrategy.RANDOM,
            output_dir=os.path.join(output_dir, "CatBoost"),
            task_type=TaskType.REGRESSION,
            validation_fraction=0.2,
            test_fraction=0.0,
            random_seed=random_seed,
        ),
        "Classifier_Regression": ClassifierRegressionModel(
            base_classifier_class=PyTorchNeuralNetwork,
            base_classifier_params={
                "input_size": X.shape[1],
                "output_size": 10,  # 10 classes for regression
                "n_epochs": 500,
                "hidden_layers": nn_hidden_layers,
                "hidden_size": nn_hidden_size,
                "early_stopping_rounds": early_stopping_rounds,
            },
            n_classes=10,
            split_strategy=DataSplitStrategy.RANDOM,
            output_dir=os.path.join(output_dir, "Classifier_Regression"),
            validation_fraction=0.2,
            test_fraction=0.0,
            random_seed=random_seed,
            early_stopping_rounds=early_stopping_rounds,
        ),
    }

    results = {}

    logging.info("--- Running Model Training and Evaluation ---")
    for name, model in models_to_test.items():
        logging.info(f"Training {name}...")
        model_output_dir = os.path.join(output_dir, name)
        os.makedirs(model_output_dir, exist_ok=True)

        if "Linear" in name:
            x_data = X_linear
            data_handler = DataHandler(scale_binary_features=False)
            x_train_val, y_train_val = data_handler.fit_transform(x_data.iloc[train_val_indices].values, y.iloc[train_val_indices].values)
            all_data_for_output = X_linear.copy()
            all_data_for_output['Actual Price (GBP)'] = y
        else:
            x_data = X
            data_handler = DataHandler(scale_binary_features=False)
            x_train_val, y_train_val = data_handler.fit_transform(x_data.iloc[train_val_indices].values, y.iloc[train_val_indices].values)
            all_data_for_output = full_df

        model.fit(pd.DataFrame(x_train_val, columns=x_data.columns), y_train_val)

        # Generate output files
        generate_output_file(model, data_handler, all_data_for_output, train_val_indices, os.path.join(model_output_dir, "train_predictions.csv"), x_data.columns.tolist())
        generate_output_file(model, data_handler, all_data_for_output, test_indices, os.path.join(model_output_dir, "test_predictions.csv"), x_data.columns.tolist())

        # Evaluate on test set
        x_test_scaled, _ = data_handler.transform(x_data.iloc[test_indices].values, None)
        y_pred_scaled = model.predict(x_test_scaled)
        y_pred = data_handler.inverse_transform_y(y_pred_scaled)
        test_mse = mean_squared_error(y.iloc[test_indices], y_pred)
        results[name] = {"MSE": test_mse}
        logging.info(f"  -> Test MSE: {test_mse:.4f}")

        if "Linear" in name:
            get_linear_model_params(model, x_data.columns.tolist(), model_output_dir)

    logging.info("\n--- Model Performance Summary ---")
    results_df = pd.DataFrame.from_dict(results, orient="index").sort_values(by="MSE")
    logging.info(f"\n{results_df}")

    logging.info("\nShowcase complete.")


if __name__ == "__main__":
    run_house_price_prediction()