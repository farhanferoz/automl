"""I2: run head-structure diagnostics across a grid of ProbReg configs.

Flags configs where:
- mirror_ok is False (outer heads don't have opposite monotonicity)
- middle_flat_ok is False (middle head not approximately constant for odd k>=3)
- mean_sep_ok is False (outer-head means not well-separated)
- any head has zero output range (head never activates)

Writes a CSV that can be joined against ablation-sweep results to correlate
learned head structure with MSE/NLL outcomes.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from automl_package.enums import (
    NClassesSelectionMethod,
    RegressionStrategy,
    UncertaintyMethod,
)
from automl_package.models.probabilistic_regression import ProbabilisticRegressionModel
from automl_package.utils.head_diagnostics import analyse_head_structure

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

OUT_DIR = Path(__file__).parent / "head_structure_results"
OUT_DIR.mkdir(exist_ok=True)


def make_heteroscedastic(seed: int = 42) -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-5.0, 5.0, 600).reshape(-1, 1).astype(np.float32)
    y_true = np.sin(x.ravel()) * 2 + 0.5 * x.ravel()
    noise = rng.normal(0.0, 0.1 + 0.4 * np.abs(x.ravel()))
    y = (y_true + noise).astype(np.float32)
    return x, y, float(y.max() - y.min())


def make_bimodal(seed: int = 42) -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3.0, 3.0, 600).reshape(-1, 1).astype(np.float32)
    sign = rng.choice([-1.0, 1.0], size=600).reshape(-1, 1)
    y = (x + sign * 1.5 + rng.normal(0, 0.1, (600, 1))).ravel().astype(np.float32)
    return x, y, float(y.max() - y.min())


def run_config(
    name: str,
    x: np.ndarray,
    y: np.ndarray,
    y_scale: float,
    n_classes: int,
    strategy: RegressionStrategy,
    loss_type: str = "nll",
    beta: float = 0.5,
    n_epochs: int = 150,
) -> dict:
    model = ProbabilisticRegressionModel(
        input_size=x.shape[1], n_classes=n_classes, max_n_classes_for_probabilistic_path=n_classes + 2,
        uncertainty_method=UncertaintyMethod.PROBABILISTIC,
        n_classes_selection_method=NClassesSelectionMethod.NONE,
        regression_strategy=strategy, loss_type=loss_type, beta=beta,
        base_classifier_params={"hidden_layers": 1, "hidden_size": 64},
        regression_head_params={"hidden_layers": 0, "hidden_size": 32},
        n_epochs=n_epochs, learning_rate=0.01, early_stopping_rounds=25,
        validation_fraction=0.2, random_seed=42, calculate_feature_importance=False,
    )
    model.fit(x, y)
    rep = analyse_head_structure(model, y_scale=y_scale)
    y_pred = model.predict(x)
    mse = float(np.mean((y.ravel() - y_pred.ravel()) ** 2))
    row = {"config": name, "n_classes": n_classes, "strategy": strategy.value, "loss": loss_type, "beta": beta, "mse": mse}
    row.update({f"slope_{i}": s for i, s in enumerate(rep.head_slopes)})
    row.update({f"range_{i}": r for i, r in enumerate(rep.head_output_range)})
    row.update({
        "mirror_ok": rep.mirror_ok,
        "middle_flat_ok": rep.middle_flat_ok,
        "mean_sep_ok": rep.mean_sep_ok,
        "passed": rep.as_dict()["passed"],
    })
    return row


def main() -> None:
    datasets = [
        ("heteroscedastic", *make_heteroscedastic()),
        ("bimodal", *make_bimodal()),
    ]

    rows: list[dict] = []
    for ds_name, x, y, y_scale in datasets:
        for k in (2, 3, 5, 7):
            for strat in (RegressionStrategy.SEPARATE_HEADS, RegressionStrategy.SINGLE_HEAD_N_OUTPUTS):
                for loss in ("nll", "beta_nll"):
                    name = f"{ds_name}_k{k}_{strat.value}_{loss}"
                    try:
                        row = run_config(name, x, y, y_scale, k, strat, loss_type=loss, beta=0.5)
                        row["dataset"] = ds_name
                        rows.append(row)
                        status = "PASS" if row["passed"] else "FAIL"
                        logger.info(f"{status} {name:<50} mse={row['mse']:.4f}  mirror={row['mirror_ok']} sep={row['mean_sep_ok']}")
                    except Exception as e:  # noqa: BLE001
                        logger.exception(f"Config {name} failed: {e}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "head_structure.csv", index=False)

    # Failure summary
    failed = df[~df["passed"]]
    logger.info(f"\n{len(failed)}/{len(df)} configs failed structural checks")
    for _, r in failed.iterrows():
        logger.info(f"  FAIL: {r['config']} (mirror={r['mirror_ok']}, flat={r['middle_flat_ok']}, sep={r['mean_sep_ok']})")


if __name__ == "__main__":
    main()
