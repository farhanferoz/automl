"""Dynamic k Comparison: fixed-k vs unregularized dynamic-k vs ELBO dynamic-k vs k-penalty dynamic-k.

Runs on heteroscedastic sine dataset (noise grows with |x|).
Reports MSE, NLL, calibration, and mean selected k.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from automl_package.enums import (
    NClassesSelectionMethod,
    RegressionStrategy,
    UncertaintyMethod,
)
from automl_package.models.probabilistic_regression import ProbabilisticRegressionModel

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def make_heteroscedastic_data(n=1000, seed=42):
    np.random.seed(seed)
    x = np.random.uniform(-5, 5, n).reshape(-1, 1)
    y_true = np.sin(x) * 2 + 0.5 * x
    noise_std = 0.1 + 0.4 * np.abs(x)
    y = y_true + np.random.normal(0, noise_std)
    return x, y.ravel(), y_true.ravel(), noise_std.ravel()


def compute_nll(y_true, y_pred, y_std):
    log_var = 2 * np.log(np.clip(y_std, 1e-6, None))
    return float(0.5 * np.mean(log_var + ((y_true - y_pred) ** 2) / np.exp(log_var)))


def compute_calibration(y_true, y_pred, y_std):
    """Fraction of test points within ±1σ (ideal ≈ 0.6827)."""
    return float(np.mean(np.abs(y_true - y_pred) <= y_std))


def get_mean_k(model, x):
    """Get mean selected k from dynamic model."""
    import torch
    x_t = torch.tensor(x, dtype=torch.float32).to(model.device)
    model.model.eval()
    with torch.no_grad():
        _, _, k_actual, _, _ = model.model(x_t)
    return float(k_actual.float().mean().item())


def run_comparison():
    x, y, y_true, noise_std = make_heteroscedastic_data(n=1000)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    _, noise_test = train_test_split(noise_std, test_size=0.3, random_state=42)

    results = []
    common = dict(
        input_size=1,
        uncertainty_method=UncertaintyMethod.PROBABILISTIC,
        regression_strategy=RegressionStrategy.SEPARATE_HEADS,
        n_epochs=100, learning_rate=0.01, early_stopping_rounds=15,
        validation_fraction=0.2, random_seed=42,
        calculate_feature_importance=False,
    )

    configs = [
        # Fixed k sweep
        ("Fixed k=3", dict(n_classes=3, n_classes_selection_method=NClassesSelectionMethod.NONE)),
        ("Fixed k=5", dict(n_classes=5, n_classes_selection_method=NClassesSelectionMethod.NONE)),
        ("Fixed k=7", dict(n_classes=7, n_classes_selection_method=NClassesSelectionMethod.NONE)),
        ("Fixed k=10", dict(n_classes=10, n_classes_selection_method=NClassesSelectionMethod.NONE)),
        # Dynamic k (unregularized)
        ("Dynamic (none)", dict(n_classes=3, max_n_classes_for_probabilistic_path=10,
                                n_classes_selection_method=NClassesSelectionMethod.GUMBEL_SOFTMAX,
                                n_classes_regularization="none")),
        # Dynamic k (ELBO)
        ("Dynamic (ELBO)", dict(n_classes=3, max_n_classes_for_probabilistic_path=10,
                                n_classes_selection_method=NClassesSelectionMethod.GUMBEL_SOFTMAX,
                                n_classes_regularization="elbo")),
        # Dynamic k (k-penalty)
        ("Dynamic (penalty)", dict(n_classes=3, max_n_classes_for_probabilistic_path=10,
                                   n_classes_selection_method=NClassesSelectionMethod.GUMBEL_SOFTMAX,
                                   n_classes_regularization="k_penalty", k_penalty_weight=0.02)),
        # Dynamic with different strategies
        ("Dynamic ELBO+SoftGating", dict(n_classes=3, max_n_classes_for_probabilistic_path=10,
                                         n_classes_selection_method=NClassesSelectionMethod.SOFT_GATING,
                                         n_classes_regularization="elbo")),
    ]

    for name, cfg in configs:
        print(f"Training {name}...")
        model = ProbabilisticRegressionModel(**{**common, **cfg})
        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)
        y_std = model.predict_uncertainty(x_test)

        mse = float(np.mean((y_test - y_pred) ** 2))
        nll = compute_nll(y_test, y_pred, y_std)
        cal = compute_calibration(y_test, y_pred, y_std)
        noise_corr = float(np.corrcoef(noise_test.ravel(), y_std.ravel())[0, 1])

        is_dynamic = cfg.get("n_classes_selection_method") != NClassesSelectionMethod.NONE
        mean_k = get_mean_k(model, x_test) if is_dynamic else cfg["n_classes"]

        results.append({
            "Model": name,
            "MSE": round(mse, 4),
            "NLL": round(nll, 4),
            "Cal(1σ)": round(cal, 3),
            "Noise r": round(noise_corr, 3),
            "Mean k": round(mean_k, 2),
        })

    df = pd.DataFrame(results)
    print("\n" + "=" * 80)
    print("DYNAMIC k COMPARISON — Heteroscedastic Sine (n=1000)")
    print("=" * 80)
    print(df.to_string(index=False))
    print("\nIdeal calibration at 1σ: 0.683")
    print("Higher noise correlation = better uncertainty estimation")
    return df


if __name__ == "__main__":
    run_comparison()
