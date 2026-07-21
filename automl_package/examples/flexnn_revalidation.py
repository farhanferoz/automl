r"""M0: FlexNN post-Phase-9-bug re-validation of the piecewise-dataset ELBO depth-selection claim.

Context (memory `project_phase9_bugs`; `docs/plans/capacity_programme/flexnn-moe.md` Task M0):
`FlexibleHiddenLayersNN`'s `n_predictor` sub-network was excluded from the optimizer (frozen) plus
had a tuple-unpack off-by-one until the Phase-9 fix (commit 07c0b09, "phase 9 autonomous: FlexNN
optimizer fix", 2026-04-18). Every FlexNN depth-regularization number that predates that commit —
including the headline claim in `docs/research_plan.md` Executive Summary ("FlexibleNN's ELBO
depth regularization works as advertised on the piecewise dataset") — is untrustworthy and must be
re-measured against CURRENT code, not cited.

Searched before writing (minimum-viable-code ladder rung 2): `grep -rln "FlexibleHiddenLayers\\|
DepthRegularization\\|n_predictor" automl_package/examples/` surfaced `flex_nn_depth_viz.py` (I3),
the exact piecewise-dataset ELBO depth-selection driver the claim traces to, and
`gumbel_elbo_retest.py` (I8), a post-fix NONE-vs-ELBO retest. Neither meets this task's spec (no
multi-seed loop, no convergence-gated trajectory / hit_cap flags per MASTER Decision 9, no
n_predictor grad-norm cross-check), so this script REUSES `flex_nn_depth_viz.piecewise_dataset`
and `.extract_depth` rather than redefining them, and reuses `convergence.ConvergenceTracker`
for the trajectory-rule bookkeeping rather than reimplementing patience logic — adding only the
seed loop, held-out test split, trajectory replay, and grad-norm assertion this task requires.

Per-run early stopping is `FlexibleHiddenLayersNN`'s own (strict-decrease, no min_delta) patience
gate over `N_EPOCHS_CAP` epochs; the returned per-epoch held-out trajectory is then replayed
through `ConvergenceTracker` (with a min-delta noise floor) to get the canonical
converged/trustworthy verdict. `hit_cap` is tracked directly as "ran the full epoch budget without
the model's own early stopping firing" (needs a larger budget, not a conclusion).

Run:
    AUTOML_DEVICE=cpu OMP_NUM_THREADS=2 ~/dev/.venv/bin/python automl_package/examples/flexnn_revalidation.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from automl_package.enums import DataSplitStrategy, DepthRegularization, LayerSelectionMethod, UncertaintyMethod
from automl_package.examples.convergence import ConvergenceTracker
from automl_package.examples.flex_nn_depth_viz import extract_depth, piecewise_dataset
from automl_package.models.flexible_neural_network import FlexibleHiddenLayersNN
from automl_package.utils.data_handler import create_train_val_split

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

OUT_DIR = Path(__file__).parent / "report_b_results"
OUT_DIR.mkdir(exist_ok=True)

SEEDS = (0, 1, 2, 3, 4)
DATASET_N = 1200
MAX_HIDDEN_LAYERS = 5
HIDDEN_SIZE = 64
LEARNING_RATE = 0.01
N_EPOCHS_CAP = 2000  # safety cap (MASTER Decision 9) — hitting it means "needs more budget", not a verdict
PATIENCE = 50  # epochs of no improvement before declaring convergence (both model's own gate and the replay)
MIN_DELTA = 1e-4  # held-out MSE decrease counted as real improvement in the trajectory replay (toy-scale loss)
VAL_FRACTION = 0.2
TEST_FRACTION = 0.2
REGIME_LINEAR = 0  # piecewise_dataset regime label: x < 0 (linear half)
REGIME_SIN = 1  # piecewise_dataset regime label: x >= 0 (linear + sinusoidal half)

REG_CONFIGS: tuple[tuple[str, DepthRegularization], ...] = (
    ("elbo", DepthRegularization.ELBO),
    ("none", DepthRegularization.NONE),
)


def assert_n_predictor_gets_gradient() -> dict:
    """Cross-check the Phase-9 fix is live: one real training step must give n_predictor a nonzero grad.

    Runs ONE epoch through `FlexibleHiddenLayersNN._fit_single` (the actual training path used by
    every run below, `forced_iterations=1`) and inspects `n_predictor`'s gradient after that epoch's
    last `loss.backward()`. `_setup_optimizers` (flexible_neural_network.py:458-476) includes
    `n_predictor` params in the main optimizer for every non-REINFORCE strategy — if that ever
    regresses, grads stay `None`/zero here and this assertion catches it before any depth-selection
    number below is trusted.
    """
    x, y, _, _ = piecewise_dataset(n=300, seed=0)
    train_idx, val_idx, _ = create_train_val_split(
        x, validation_fraction=VAL_FRACTION, test_fraction=TEST_FRACTION, split_strategy=DataSplitStrategy.RANDOM, timestamps=None, random_state=0
    )
    # SOFT_GATING is RETIRED under the nested ladder (MASTER Decision 29); this script's entire
    # subject IS SOFT_GATING's n_predictor and the ELBO depth-selection claim under it (the
    # Phase-9 re-validation this file is named for), so both constructions here stay labelled
    # comparison arms via the explicit opt-out.
    model = FlexibleHiddenLayersNN(
        max_hidden_layers=MAX_HIDDEN_LAYERS,
        hidden_size=HIDDEN_SIZE,
        layer_selection_method=LayerSelectionMethod.SOFT_GATING,
        depth_regularization=DepthRegularization.ELBO,
        uncertainty_method=UncertaintyMethod.CONSTANT,
        n_predictor_layers=1,
        learning_rate=LEARNING_RATE,
        early_stopping_rounds=PATIENCE,
        random_seed=0,
        calculate_feature_importance=False,
        allow_retired_capacity_selection=True,
    )
    model._fit_single(x[train_idx], y[train_idx], x_val=x[val_idx], y_val=y[val_idx], forced_iterations=1)

    grads = [p.grad for p in model.model.n_predictor.parameters()]
    grad_norm = float(sum(g.norm().item() for g in grads if g is not None))
    n_with_grad = sum(1 for g in grads if g is not None)
    n_total = len(grads)
    result = {
        "n_predictor_grad_norm": grad_norm,
        "n_params_with_nonnull_grad": n_with_grad,
        "n_params_total": n_total,
        "passed": grad_norm > 0.0 and n_with_grad == n_total,
    }
    logger.info(f"[grad-norm check] n_predictor grad_norm={grad_norm:.6g} params_with_grad={n_with_grad}/{n_total}  {'PASS' if result['passed'] else 'FAIL'}")
    return result


def run_one(reg_name: str, reg: DepthRegularization, seed: int) -> dict:
    """Trains one (regularization, seed) config on piecewise data and lands its result JSON immediately."""
    x, y, _y_true, regime = piecewise_dataset(n=DATASET_N, seed=seed)
    train_idx, val_idx, test_idx = create_train_val_split(
        x, validation_fraction=VAL_FRACTION, test_fraction=TEST_FRACTION, split_strategy=DataSplitStrategy.RANDOM, timestamps=None, random_state=seed
    )
    x_te, y_te, regime_te = x[test_idx], y[test_idx], regime[test_idx]

    model = FlexibleHiddenLayersNN(
        max_hidden_layers=MAX_HIDDEN_LAYERS,
        hidden_size=HIDDEN_SIZE,
        layer_selection_method=LayerSelectionMethod.SOFT_GATING,
        depth_regularization=reg,
        uncertainty_method=UncertaintyMethod.CONSTANT,
        n_predictor_layers=1,
        learning_rate=LEARNING_RATE,
        n_epochs=N_EPOCHS_CAP,
        early_stopping_rounds=PATIENCE,
        random_seed=seed,
        calculate_feature_importance=False,
        allow_retired_capacity_selection=True,
    )
    _best_epoch, val_loss_history = model._fit_single(x[train_idx], y[train_idx], x_val=x[val_idx], y_val=y[val_idx])

    hit_cap = len(val_loss_history) >= N_EPOCHS_CAP  # model's own early stop never fired
    tracker = ConvergenceTracker(patience=PATIENCE, min_delta=MIN_DELTA)
    for epoch, val in enumerate(val_loss_history, start=1):
        tracker.update(epoch, val)
    replay = tracker.result(final_epoch=len(val_loss_history))

    y_pred_te = model.predict(x_te, filter_data=False)
    test_mse = float(np.mean((y_te - y_pred_te) ** 2))

    argmax_depth, _soft_probs = extract_depth(model, x_te)
    rho_dist, p_dist = spearmanr(np.abs(x_te.ravel()), argmax_depth)
    rho_regime, p_regime = spearmanr(regime_te, argmax_depth)
    mean_depth_linear = float(argmax_depth[regime_te == REGIME_LINEAR].mean()) if np.any(regime_te == REGIME_LINEAR) else None
    mean_depth_sin = float(argmax_depth[regime_te == REGIME_SIN].mean()) if np.any(regime_te == REGIME_SIN) else None

    record = {
        "config": reg_name,
        "seed": seed,
        "dataset": "piecewise",
        "n_dataset": DATASET_N,
        "n_test": len(x_te),
        "layer_selection_method": "soft_gating",
        "depth_regularization": reg_name,
        "max_hidden_layers": MAX_HIDDEN_LAYERS,
        "n_epochs_cap": N_EPOCHS_CAP,
        "patience": PATIENCE,
        "epochs_trained": len(val_loss_history),
        "hit_cap": hit_cap,
        "model_early_stopped": not hit_cap,
        "trajectory_replay": replay.summary(),
        "trustworthy": replay.trustworthy and not hit_cap,
        "test_mse": test_mse,
        "mean_selected_depth_linear_regime": mean_depth_linear,
        "mean_selected_depth_sin_regime": mean_depth_sin,
        "spearman_depth_vs_abs_x": {"rho": float(rho_dist), "p": float(p_dist)},
        "spearman_depth_vs_regime": {"rho": float(rho_regime), "p": float(p_regime)},
    }
    out_path = OUT_DIR / f"flexnn_revalidation_piecewise_{reg_name}_seed{seed}.json"
    out_path.write_text(json.dumps(record, indent=2))
    logger.info(
        f"[{reg_name} seed={seed}] epochs={record['epochs_trained']} hit_cap={hit_cap} trustworthy={record['trustworthy']} "
        f"test_mse={test_mse:.4f} depth(linear/sin)={mean_depth_linear}/{mean_depth_sin} rho_regime={rho_regime:.3f} -> {out_path}"
    )
    return record


def main() -> None:
    """Runs the grad-norm cross-check then the full (regularization x seed) revalidation grid."""
    grad_check = assert_n_predictor_gets_gradient()
    (OUT_DIR / "flexnn_revalidation_grad_check.json").write_text(json.dumps(grad_check, indent=2))
    if not grad_check["passed"]:
        logger.warning("n_predictor grad-norm check FAILED — depth-selection results below cannot be trusted as evidence the Phase-9 fix is live.")

    records = [run_one(reg_name, reg, seed) for reg_name, reg in REG_CONFIGS for seed in SEEDS]

    summary = {"grad_check": grad_check, "n_configs": len(REG_CONFIGS), "n_seeds": len(SEEDS), "by_config": {}}
    for reg_name, _reg in REG_CONFIGS:
        rows = [r for r in records if r["config"] == reg_name]
        n_trustworthy = sum(1 for r in rows if r["trustworthy"])
        n_hit_cap = sum(1 for r in rows if r["hit_cap"])
        rhos_regime = [r["spearman_depth_vs_regime"]["rho"] for r in rows]
        depths_linear = [r["mean_selected_depth_linear_regime"] for r in rows if r["mean_selected_depth_linear_regime"] is not None]
        depths_sin = [r["mean_selected_depth_sin_regime"] for r in rows if r["mean_selected_depth_sin_regime"] is not None]
        summary["by_config"][reg_name] = {
            "n_seeds": len(rows),
            "n_trustworthy": n_trustworthy,
            "n_hit_cap": n_hit_cap,
            "mean_test_mse": float(np.mean([r["test_mse"] for r in rows])),
            "mean_spearman_depth_vs_regime": float(np.mean(rhos_regime)),
            "mean_selected_depth_linear_regime": float(np.mean(depths_linear)) if depths_linear else None,
            "mean_selected_depth_sin_regime": float(np.mean(depths_sin)) if depths_sin else None,
            "seeds": [r["seed"] for r in rows],
        }
    summary_path = OUT_DIR / "flexnn_revalidation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
