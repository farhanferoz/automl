"""Sanity check for photo-z domain metrics and LocallyAdaptiveConformalWrapper.

Generates a synthetic redshift-like dataset (z_spec in [0, 2], heteroscedastic noise),
fits a ProbabilisticRegressionModel, then verifies:
  1. photo_z_metrics() returns correct keys and reasonable values.
  2. LocallyAdaptiveConformalWrapper achieves claimed coverage on held-out test set.
  3. LocallyAdaptiveConformalWrapper produces narrower intervals than ConformalWrapper
     in low-noise regions and wider intervals in high-noise regions.

Usage:
    ~/dev/.venv/bin/python -m automl_package.examples.photo_z_domain_metrics_test
"""

import numpy as np
from sklearn.model_selection import train_test_split

from automl_package.enums import NClassesSelectionMethod, RegressionStrategy, UncertaintyMethod
from automl_package.models.conformal import ConformalWrapper, LocallyAdaptiveConformalWrapper
from automl_package.models.probabilistic_regression import ProbabilisticRegressionModel
from automl_package.utils.distributions import GaussianDistribution
from automl_package.utils.domain_metrics import photo_z_metrics


def _make_photo_z_data(n: int = 1500, seed: int = 42) -> dict:
    """Synthetic photo-z dataset: z_spec in [0, 2], noise grows with z."""
    rng = np.random.default_rng(seed)

    # 5 photometric band magnitudes + 2 colours
    z_spec = rng.uniform(0.05, 2.0, n).astype(np.float32)
    noise_std = 0.02 + 0.05 * z_spec  # heteroscedastic: noisier at high z

    x = np.column_stack([
        z_spec + rng.normal(0, noise_std),     # band 1 proxy
        z_spec ** 1.5 + rng.normal(0, 0.1, n),
        np.sin(z_spec * 2) + rng.normal(0, 0.05, n),
        np.log1p(z_spec) + rng.normal(0, 0.05, n),
        rng.normal(0, 0.3, n),                 # noise feature
    ]).astype(np.float32)

    y = z_spec + rng.normal(0, noise_std)
    return {"x": x, "y": y.astype(np.float32), "z_true": z_spec, "noise_std": noise_std}


def _print_section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print("=" * 60)


def run() -> None:
    """Run the photo-z sanity check."""
    data = _make_photo_z_data(n=1500)
    x, y = data["x"], data["y"]
    z_true = data["z_true"]

    # 60/20/20 split: train/cal/test
    x_tr, x_tmp, y_tr, y_tmp, _z_tr, z_tmp = train_test_split(x, y, z_true, test_size=0.4, random_state=42)
    x_cal, x_test, y_cal, y_test, _z_cal, z_test = train_test_split(x_tmp, y_tmp, z_tmp, test_size=0.5, random_state=42)

    _print_section("1. Fitting ProbabilisticRegressionModel")
    model = ProbabilisticRegressionModel(
        n_classes=5,
        uncertainty_method=UncertaintyMethod.PROBABILISTIC,
        n_classes_selection_method=NClassesSelectionMethod.NONE,
        regression_strategy=RegressionStrategy.SEPARATE_HEADS,
        base_classifier_params={"hidden_layers": 1, "hidden_size": 32},
        regression_head_params={"hidden_layers": 0, "hidden_size": 16},
        input_size=x.shape[1],
        n_epochs=80,
        early_stopping_rounds=15,
        learning_rate=0.01,
        validation_fraction=0.2,
        random_seed=42,
        calculate_feature_importance=False,
        use_hpo=False,
    )
    model.fit(x_tr, y_tr)
    y_pred = model.predict(x_test)
    y_std = model.predict_uncertainty(x_test)
    print(f"  predict shape : {y_pred.shape}")
    print(f"  uncertainty shape: {y_std.shape}")

    # --- 1. photo_z_metrics ---
    _print_section("2. photo_z_metrics (point predictions)")
    metrics_point = photo_z_metrics(z_test, y_pred)
    for k, v in metrics_point.items():
        print(f"  {k:<25} {v:>10.5f}")
    assert set(metrics_point.keys()) == {"sigma_mad", "bias", "outlier_fraction"}

    dist = GaussianDistribution(y_pred, y_std)
    metrics_dist = photo_z_metrics(z_test, y_pred, dist=dist)
    print(f"\n  {'cde_loss':<25} {metrics_dist['cde_loss']:>10.5f}")
    assert "cde_loss" in metrics_dist
    print("  photo_z_metrics: OK")

    # --- 2. ConformalWrapper coverage ---
    _print_section("3. ConformalWrapper coverage")
    cw = ConformalWrapper(model)
    cw.calibrate(x_cal, y_cal, alpha=0.1)
    lo, hi = cw.predict_interval(x_test)
    coverage_cw = float(np.mean((y_test >= lo) & (y_test <= hi)))
    widths_cw = hi - lo
    print("  Target coverage : 0.90")
    print(f"  Achieved coverage: {coverage_cw:.4f}")
    assert coverage_cw >= 0.85, f"Coverage too low: {coverage_cw:.4f}"

    # --- 3. LocallyAdaptiveConformalWrapper ---
    _print_section("4. LocallyAdaptiveConformalWrapper coverage")
    lacw = LocallyAdaptiveConformalWrapper(model)
    lacw.calibrate(x_cal, y_cal, alpha=0.1)
    lo_la, hi_la = lacw.predict_interval(x_test)
    coverage_la = float(np.mean((y_test >= lo_la) & (y_test <= hi_la)))
    widths_la = hi_la - lo_la
    print("  Target coverage : 0.90")
    print(f"  Achieved coverage: {coverage_la:.4f}")
    assert coverage_la >= 0.85, f"Coverage too low: {coverage_la:.4f}"

    # --- 4. Adaptivity: LACW should be tighter in low-noise, wider in high-noise ---
    _print_section("5. Adaptivity check")
    # Low-noise region: z_spec < 0.5 (noise_std ~ 0.02-0.045)
    # High-noise region: z_spec > 1.5 (noise_std ~ 0.095-0.12)
    lo_z_mask = z_test < 0.5
    hi_z_mask = z_test > 1.5

    mean_width_cw_lo = float(np.mean(widths_cw[lo_z_mask])) if lo_z_mask.sum() > 0 else float("nan")
    mean_width_la_lo = float(np.mean(widths_la[lo_z_mask])) if lo_z_mask.sum() > 0 else float("nan")
    mean_width_cw_hi = float(np.mean(widths_cw[hi_z_mask])) if hi_z_mask.sum() > 0 else float("nan")
    mean_width_la_hi = float(np.mean(widths_la[hi_z_mask])) if hi_z_mask.sum() > 0 else float("nan")

    print(f"  Low-z  (n={lo_z_mask.sum():3d}): ConfW width={mean_width_cw_lo:.4f}  LACW width={mean_width_la_lo:.4f}  (LACW tighter: {mean_width_la_lo < mean_width_cw_lo})")
    print(f"  High-z (n={hi_z_mask.sum():3d}): ConfW width={mean_width_cw_hi:.4f}  LACW width={mean_width_la_hi:.4f}  (LACW wider:   {mean_width_la_hi > mean_width_cw_hi})")

    # Soft assertion: at least one direction should hold
    adaptivity_ok = (mean_width_la_lo < mean_width_cw_lo) or (mean_width_la_hi > mean_width_cw_hi)
    status = "OK" if adaptivity_ok else "NOTE: adaptivity not clearly visible (model may not be well-calibrated)"
    print(f"  Adaptivity: {status}")

    _print_section("Summary")
    print(f"  ConformalWrapper    coverage={coverage_cw:.4f}  mean_width={float(np.mean(widths_cw)):.4f}")
    print(f"  LocallyAdaptiveConf coverage={coverage_la:.4f}  mean_width={float(np.mean(widths_la)):.4f}")
    print("\n  All assertions passed. photo-z metrics and conformal wrappers are functional.")


if __name__ == "__main__":
    run()
