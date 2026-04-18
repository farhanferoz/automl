"""Compare ProbReg mixture evaluation vs collapsed-Gaussian evaluation on multimodal data.

For bimodal targets y = x ± 1.5, the law-of-total-variance collapse gives a
single wide Gaussian that over-estimates uncertainty and underestimates log-likelihood.
Evaluating via the full mixture-of-Gaussians (predict_distribution) should produce
better NLL and CRPS.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split

from automl_package.enums import NClassesSelectionMethod, RegressionStrategy, UncertaintyMethod
from automl_package.models.probabilistic_regression import ProbabilisticRegressionModel
from automl_package.utils.distributions import GaussianDistribution
from automl_package.utils.scoring import calculate_crps


def generate_bimodal(n: int = 1500, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3.0, 3.0, n).reshape(-1, 1).astype(np.float32)
    sign = rng.choice([-1.0, 1.0], size=n).reshape(-1, 1)
    y = (x + sign * 1.5 + rng.normal(0, 0.1, (n, 1))).ravel().astype(np.float32)
    return x, y


def main() -> None:
    x, y = generate_bimodal()
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.3, random_state=0)

    model = ProbabilisticRegressionModel(
        input_size=1, n_classes=2, max_n_classes_for_probabilistic_path=5,
        uncertainty_method=UncertaintyMethod.PROBABILISTIC,
        n_classes_selection_method=NClassesSelectionMethod.NONE,
        regression_strategy=RegressionStrategy.SEPARATE_HEADS,
        n_epochs=80, learning_rate=0.01,
        early_stopping_rounds=20, validation_fraction=0.2,
        random_seed=42, calculate_feature_importance=False,
    )
    model.fit(x_tr, y_tr)

    # Collapsed-Gaussian evaluation
    mu = model.predict(x_te)
    sigma = model.predict_uncertainty(x_te)
    gauss = GaussianDistribution(mu, sigma)
    nll_g = -float(np.mean(gauss.log_prob(y_te)))
    crps_g = calculate_crps(y_te, gauss)

    # Mixture evaluation
    mix = model.predict_distribution(x_te)
    nll_m = -float(np.mean(mix.log_prob(y_te)))
    crps_m = calculate_crps(y_te, mix)

    print(f"{'metric':<8}  {'gaussian':>10}  {'mixture':>10}  {'delta':>10}")
    print(f"{'NLL':<8}  {nll_g:>10.4f}  {nll_m:>10.4f}  {nll_m - nll_g:>+10.4f}")
    print(f"{'CRPS':<8}  {crps_g:>10.4f}  {crps_m:>10.4f}  {crps_m - crps_g:>+10.4f}")
    print("(negative delta = mixture is better)")


if __name__ == "__main__":
    main()
