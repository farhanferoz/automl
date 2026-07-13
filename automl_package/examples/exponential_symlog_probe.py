"""One-shot probe: does target_transform='symlog' rescue C/D on exponential?

Runs the 8 identifiability cells + ClassReg on the exponential dataset with
target_transform in {None, 'symlog'}, k in {3, 5}, 3 seeds. Writes a single CSV
and prints a markdown comparison. ~10 min on XPU.

Hypothesis: on exponential, percentile-bin centroids are widely separated
(~0.2, 1.5, 12 at k=3), making the LTV inter-head variance term dominate σ²_total
and damp the NLL gradient. CE_STOP_GRAD cells can't compress classifier probs to
recover, so they fail. symlog compresses the range and should restore gradient.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd

from automl_package.enums import (
    ProbabilisticRegressionOptimizationStrategy,
    ProbRegLossType,
    RegressionStrategy,
    UncertaintyMethod,
)
from automl_package.models.probabilistic_regression import ProbabilisticRegressionModel

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

OUT_DIR = Path(__file__).parent / "exponential_symlog_probe_results"
OUT_DIR.mkdir(exist_ok=True)

CELLS = [
    ("A", ProbRegLossType.GAUSSIAN_LTV, ProbabilisticRegressionOptimizationStrategy.REGRESSION_ONLY, False),
    ("B", ProbRegLossType.GAUSSIAN_LTV, ProbabilisticRegressionOptimizationStrategy.REGRESSION_ONLY, True),
    ("C", ProbRegLossType.GAUSSIAN_LTV, ProbabilisticRegressionOptimizationStrategy.CE_STOP_GRAD, False),
    ("D", ProbRegLossType.GAUSSIAN_LTV, ProbabilisticRegressionOptimizationStrategy.CE_STOP_GRAD, True),
    ("E", ProbRegLossType.MDN, ProbabilisticRegressionOptimizationStrategy.REGRESSION_ONLY, False),
    ("F", ProbRegLossType.MDN, ProbabilisticRegressionOptimizationStrategy.REGRESSION_ONLY, True),
    ("G", ProbRegLossType.MDN, ProbabilisticRegressionOptimizationStrategy.CE_STOP_GRAD, False),
    ("H", ProbRegLossType.MDN, ProbabilisticRegressionOptimizationStrategy.CE_STOP_GRAD, True),
]

K_VALUES = [3, 5]
SEEDS = [42, 43, 44]
TRANSFORMS = [None, "symlog"]
N_EPOCHS = 80
LR = 0.01
EARLY_STOP = 15
VAL_FRAC = 0.2


def _make_exponential(seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3, 3, 600).reshape(-1, 1).astype(np.float32)
    y = (np.exp(x.ravel()) + rng.normal(0.0, 0.5, 600)).astype(np.float32)
    return x, y


def _train(x_tr, y_tr, k, seed, loss_type, opt_strategy, use_anchored, target_transform):
    model = ProbabilisticRegressionModel(
        input_size=1,
        n_classes=k,
        regression_strategy=RegressionStrategy.SEPARATE_HEADS,
        uncertainty_method=UncertaintyMethod.PROBABILISTIC,
        optimization_strategy=opt_strategy,
        prob_reg_loss_type=loss_type,
        use_anchored_heads=use_anchored,
        constrain_middle_class=not use_anchored,
        use_monotonic_constraints=False,
        target_transform=target_transform,
        n_epochs=N_EPOCHS,
        learning_rate=LR,
        early_stopping_rounds=EARLY_STOP,
        validation_fraction=VAL_FRAC,
        random_seed=seed,
        calculate_feature_importance=False,
    )
    model.fit(x_tr, y_tr)
    return model


def _gaussian_nll(y, mean, log_var):
    var = np.exp(log_var)
    return float(np.mean(0.5 * (math.log(2 * math.pi) + log_var + (y - mean) ** 2 / var)))


def main() -> None:
    x_all, y_all = _make_exponential(seed=0)
    split = int(0.8 * len(x_all))
    x_tr, x_te = x_all[:split], x_all[split:]
    y_tr, y_te = y_all[:split], y_all[split:]

    rows: list[dict] = []
    total = len(CELLS) * len(K_VALUES) * len(SEEDS) * len(TRANSFORMS)
    done = 0
    for k in K_VALUES:
        for transform in TRANSFORMS:
            for cell, loss_type, opt_strategy, use_anchored in CELLS:
                mses, nlls = [], []
                for seed in SEEDS:
                    done += 1
                    try:
                        m = _train(x_tr, y_tr, k, seed, loss_type, opt_strategy, use_anchored, transform)
                        preds = m.predict(x_te).ravel()
                        mse = float(np.mean((preds - y_te) ** 2))
                        try:
                            std = m.predict_uncertainty(x_te).ravel()
                            log_var = np.log(np.clip(std ** 2, 1e-8, None))
                            nll = _gaussian_nll(y_te, preds, log_var)
                        except Exception:  # noqa: BLE001
                            nll = float("nan")
                        mses.append(mse)
                        nlls.append(nll)
                        logger.info(f"[{done}/{total}] k={k} tf={transform} cell={cell} seed={seed} mse={mse:.4f} nll={nll:.4f}")
                    except Exception as e:  # noqa: BLE001
                        logger.exception(f"cell {cell} k={k} tf={transform} seed={seed} crashed")
                        rows.append(dict(k=k, transform=str(transform), cell=cell, seed=seed, mse=float("nan"), nll=float("nan"), error=str(e)))
                        continue
                rows.append(dict(
                    k=k, transform=str(transform), cell=cell,
                    mse_mean=float(np.mean(mses)) if mses else float("nan"),
                    mse_std=float(np.std(mses)) if mses else float("nan"),
                    nll_mean=float(np.mean(nlls)) if nlls else float("nan"),
                    n_seeds=len(mses),
                ))

    df = pd.DataFrame([r for r in rows if "mse_mean" in r])
    df.to_csv(OUT_DIR / "results.csv", index=False)
    logger.info("\n%s", df.to_string(index=False))


if __name__ == "__main__":
    main()
