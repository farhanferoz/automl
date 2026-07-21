"""P8 — does explicit regularisation move the selected k? (`probreg.md` Task P8, MASTER Decision 21).

The programme's research training is entirely unregularised (no weight decay, dropout, norm
layers, or mini-batching), so cheapest-within-tolerance may partly select small k because small
OVERFITS LESS, not because small suffices -- a bias identical across all three capacity dials and
invisible to cross-strand agreement. This is the discriminating check: train M3's per-k ORDINARY
sweep models (`NClassesSelectionMethod.NONE`, one dedicated model per k) at AdamW weight_decay
lambda in {0, 1e-4, 1e-2}, 2 seeds, unchanged convergence gates, and see whether the strand's
global-arm selection rule (cheapest-within-tolerance at twice a bootstrap SE, MASTER Decision 18)
picks a different k as lambda grows.

Searched before writing (minimum-viable-code ladder rung 2 -- reuse, don't reinvent):
`automl_package/examples/report_a_benchmark.py` for the M3 per-k builder (`_probreg_fixed`,
`NClassesSelectionMethod.NONE`), the toy suite (`Toy`, `make_dataset`, `PRODUCTION_N`) and the
training config (`FULL_CONFIG`, `K_GRID` -- P1's pre-registered per-k grid, reused verbatim so this
check sweeps the SAME k values M3's battery does); `automl_package/utils/capacity_selection.py`
for the shared global-arm selector (`cheapest_within_tolerance`, FP-9.a -- twice-bootstrap-SE
tolerance over a held-out error curve, cheapest-first columns); `automl_package/examples/
flexnn_revalidation.py` for the pattern of calling `model._fit_single(x_train, y_train, x_val=,
y_val=)` directly (also used internally by `BaseModel`'s own CV loop, `models/base.py:443-444`) to
get an explicit `(best_epoch, val_loss_history)` trajectory and full control over the split, instead
of `model.fit()`'s built-in split. `automl_package/optimizers/standard_optimizers.py` confirmed
`AdamWrapper.create_optimizer` hardcodes `optim.Adam(model_params, lr=lr)` with NO weight_decay
kwarg on any path -- the library has no regularisation lever at all (Decision 21's audit). Adding
one is out of this task's write set (`probabilistic_regression.py` and `n_classes_strategies.py`
are being rewritten by another worker this session), so the lambda is injected via a scoped
monkeypatch of `AdamWrapper.create_optimizer` (see `_weight_decay` below) -- a driver-local
implementation of an experiment-specific protocol the library does not express, per the
package/experiment boundary rule (MASTER Decision 19). AdamW with weight_decay=0 is the identical
update rule to plain Adam (the decoupled-decay term vanishes), so the lambda=0 arm is a faithful
"no regularisation" control, not a different optimizer.

Data roles (disjoint, one draw per seed, same draw and split reused across all three lambdas for
that seed -- MASTER's confound doctrine, single-difference-only): each seed draws ONE
heteroscedastic toy sample, split 70/30 into train+val / holdout (`train_test_split`, matching
`report_a_benchmark.py::run_cell`'s convention); the 70% further splits 80/20 into the actual
training set and the early-stopping validation set (matching `_probreg_fixed`'s
`validation_fraction=0.2`); the 30% holdout splits 50/50 into a SELECTION split (scores the per-k
held-out curve `cheapest_within_tolerance` selects over) and a REPORT split (touched only for the
`report_mse`/`report_nll` context numbers) -- neither the selection split nor the report split is
used for early stopping, and the report split is not used for selection either, per the task's
split-discipline clause.

Usage:
    AUTOML_DEVICE=cpu OMP_NUM_THREADS=4 ~/dev/.venv/bin/python -m automl_package.examples.probreg_p8 --smoke
    AUTOML_DEVICE=cpu OMP_NUM_THREADS=4 ~/dev/.venv/bin/python -m automl_package.examples.probreg_p8
"""

from __future__ import annotations

import argparse
import json
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import torch.optim as optim
from sklearn.model_selection import train_test_split

from automl_package.examples.report_a_benchmark import (
    FULL_CONFIG,
    K_GRID,
    PRODUCTION_N,
    SMOKE_CONFIG,
    SMOKE_N,
    RunConfig,
    Toy,
    _probreg_fixed,
    make_dataset,
)
from automl_package.optimizers.standard_optimizers import AdamWrapper
from automl_package.utils.capacity_selection import cheapest_within_tolerance

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

RESULTS_DIR = Path(__file__).parent / "capacity_ladder_results" / "P8"

TOY = Toy.HETEROSCEDASTIC  # the strand's home-turf toy (report_a_benchmark.py::HOME_TURF_TOYS)
LAMBDA_GRID: tuple[float, ...] = (0.0, 1e-4, 1e-2)  # MASTER Decision 21's fixed grid
SEEDS: tuple[int, ...] = (100, 101)  # 2 seeds, disjoint from REPORT_SEEDS (0..4) and KSEL_SEED (999)
HOLDOUT_FRACTION = 0.3  # matches report_a_benchmark.py::run_cell's test_size=0.3
STOPPING_VAL_FRACTION = 0.2  # matches _probreg_fixed's validation_fraction=0.2
SELECTION_REPORT_SPLIT = 0.5  # 50/50 of the holdout: selection curve vs report numbers
BOOT_N = 1000  # capacity_selection.DEFAULT_N_BOOT
BOOT_SEED = 0  # capacity_selection.DEFAULT_SEED
K_MAX_RAISED = 20  # standing clause (g): re-run ceiling-bound cells here


@contextmanager
def _weight_decay(lam: float) -> Any:
    """Scoped monkeypatch: `AdamWrapper.create_optimizer` returns AdamW(weight_decay=lam) instead of Adam.

    The library exposes no weight_decay lever anywhere on the path from `ProbabilisticRegressionModel`
    to `optim.Adam(...)` (verified by reading `AdamWrapper.create_optimizer`,
    `automl_package/optimizers/standard_optimizers.py:17` -- see module docstring). This is scoped to
    the `with` block so it never leaks into any other driver sharing the process.
    """

    def _create_optimizer(self: AdamWrapper, model_params: list, lr: float) -> optim.Optimizer:  # noqa: ARG001 -- self unused, matches the wrapped signature
        return optim.AdamW(model_params, lr=lr, weight_decay=lam)

    with patch.object(AdamWrapper, "create_optimizer", _create_optimizer):
        yield


def _per_sample_nll(y_true: np.ndarray, y_pred_mean: np.ndarray, y_pred_std: np.ndarray) -> np.ndarray:
    """Elementwise Gaussian NLL per example -- the un-averaged term inside `utils/metrics.py::calculate_nll`.

    `calculate_nll` (`automl_package/utils/metrics.py:734-749`) returns only the mean; `cheapest_within_tolerance`
    needs a per-sample `(n_samples,)` column to build the paired held-out error table, so this mirrors that
    function's elementwise formula without its final `np.mean`, rather than editing the library.
    """
    variance = np.maximum(np.asarray(y_pred_std, dtype=np.float64) ** 2, 1e-9)
    diff_sq = (np.asarray(y_true, dtype=np.float64) - np.asarray(y_pred_mean, dtype=np.float64)) ** 2
    return 0.5 * (np.log(2 * np.pi * variance) + diff_sq / variance)


def _fit_one_k(
    k: int,
    lam: float,
    seed: int,
    cfg: RunConfig,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_stopval: np.ndarray,
    y_stopval: np.ndarray,
    x_sel: np.ndarray,
    y_sel: np.ndarray,
    x_rep: np.ndarray,
    y_rep: np.ndarray,
) -> dict[str, Any]:
    """Trains one dedicated per-k ORDINARY ProbReg model at weight_decay=lam; scores selection + report splits."""
    input_size = x_train.shape[1]
    model = _probreg_fixed(input_size, k, seed, cfg)
    with _weight_decay(lam):
        _best_epoch, val_loss_history = model._fit_single(x_train, y_train, x_val=x_stopval, y_val=y_stopval)
    hit_cap = len(val_loss_history) >= cfg.n_epochs

    y_pred_sel = model.predict(x_sel)
    y_std_sel = model.predict_uncertainty(x_sel)
    sel_nll_per_sample = _per_sample_nll(y_sel, y_pred_sel, y_std_sel)

    y_pred_rep = model.predict(x_rep)
    y_std_rep = model.predict_uncertainty(x_rep)
    report_mse = float(np.mean((np.asarray(y_rep) - y_pred_rep) ** 2))
    report_nll = float(np.mean(_per_sample_nll(y_rep, y_pred_rep, y_std_rep)))

    logger.info(
        f"[P8 lambda={lam} seed={seed} k={k}] epochs={len(val_loss_history)} hit_cap={hit_cap} "
        f"report_mse={report_mse:.4f} report_nll={report_nll:.4f}"
    )
    return {
        "k": k,
        "epochs_trained": len(val_loss_history),
        "hit_cap": hit_cap,
        "val_loss_history": val_loss_history,
        "selection_nll_per_sample": sel_nll_per_sample,  # kept in-memory only; not JSON-serialized (per-sample, large)
        "report_mse": report_mse,
        "report_nll": report_nll,
    }


def run_lambda_seed(lam: float, seed: int, cfg: RunConfig, n: int, k_grid: tuple[int, ...] = K_GRID) -> dict[str, Any]:
    """Fits every k in `k_grid` at (lambda, seed) on one shared data draw/split, then selects k from the curve."""
    x, y = make_dataset(TOY, seed, n)
    x_trainval, x_holdout, y_trainval, y_holdout = train_test_split(x, y, test_size=HOLDOUT_FRACTION, random_state=seed)
    x_train, x_stopval, y_train, y_stopval = train_test_split(x_trainval, y_trainval, test_size=STOPPING_VAL_FRACTION, random_state=seed)
    x_sel, x_rep, y_sel, y_rep = train_test_split(x_holdout, y_holdout, test_size=SELECTION_REPORT_SPLIT, random_state=seed)

    per_k: dict[int, dict[str, Any]] = {}
    for k in k_grid:
        per_k[k] = _fit_one_k(k, lam, seed, cfg, x_train, y_train, x_stopval, y_stopval, x_sel, y_sel, x_rep, y_rep)

    # error_table columns MUST be cheapest-first (capacity_selection.cheapest_within_tolerance contract);
    # k_grid is already ascending (K_GRID = (5, 8, 10, 12)).
    error_table = np.stack([per_k[k]["selection_nll_per_sample"] for k in k_grid], axis=1)
    selected_idx = cheapest_within_tolerance(error_table, n_boot=BOOT_N, seed=BOOT_SEED)
    selected_k = k_grid[selected_idx]
    ceiling_bound = selected_k == max(k_grid)
    any_hit_cap = any(per_k[k]["hit_cap"] for k in k_grid)

    logger.info(f"[P8 lambda={lam} seed={seed}] selected_k={selected_k} ceiling_bound={ceiling_bound} any_hit_cap={any_hit_cap}")

    return {
        "task": "P8",
        "toy": TOY.value,
        "lambda": lam,
        "seed": seed,
        "n_samples": n,
        "k_grid": list(k_grid),
        "n_epochs_cap": cfg.n_epochs,
        "n_train": len(x_train),
        "n_stopval": len(x_stopval),
        "n_selection": len(x_sel),
        "n_report": len(x_rep),
        "selection_rule": "cheapest_within_tolerance (automl_package/utils/capacity_selection.py), 2x bootstrap SE",
        "selected_k": selected_k,
        "ceiling_bound": ceiling_bound,
        "held_out_trajectory": {str(k): per_k[k]["val_loss_history"] for k in k_grid},
        "hit_cap": any_hit_cap,
        "per_k": {
            str(k): {
                "epochs_trained": per_k[k]["epochs_trained"],
                "hit_cap": per_k[k]["hit_cap"],
                "report_mse": per_k[k]["report_mse"],
                "report_nll": per_k[k]["report_nll"],
                "selection_nll_mean": float(np.mean(per_k[k]["selection_nll_per_sample"])),
            }
            for k in k_grid
        },
    }


def _save(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str))


def parse_args() -> argparse.Namespace:
    """Parses CLI args (`--smoke` for a tiny pipeline-proof config)."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke", action="store_true", help="Tiny config (few epochs, small N); files prefixed smoke__.")
    return parser.parse_args()


def main() -> None:
    """Runs the P8 discriminating check: {3 lambda} x {2 seeds} x {K_GRID}, then the selection-moved verdict."""
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        cfg = SMOKE_CONFIG
        n = SMOKE_N
        prefix = "smoke__"
    else:
        cfg = FULL_CONFIG
        n = PRODUCTION_N[TOY]
        prefix = ""

    raised_epochs_cfg = RunConfig(n_epochs=cfg.n_epochs * 4, n_estimators=cfg.n_estimators)  # Decision 9: "≥4x the self-terminated budget"

    selected_k_by_lambda_seed: dict[str, dict[str, int]] = {str(lam): {} for lam in LAMBDA_GRID}
    ceiling_raises: list[dict[str, Any]] = []
    epoch_raises: list[dict[str, Any]] = []
    any_hit_cap_overall = False

    for lam in LAMBDA_GRID:
        for seed in SEEDS:
            record = run_lambda_seed(lam, seed, cfg, n)
            out_path = RESULTS_DIR / f"{prefix}lambda{lam}_seed{seed}.json"
            _save(out_path, record)  # land immediately — standing clause (a)

            if record["hit_cap"]:
                # Decision 9 / standing clause (g): no conclusion from an endpoint — a cell that ran
                # out of its epoch budget without early stopping firing is not trajectory-verified.
                # Re-run at 4x the budget before trusting anything read off this cell (including
                # which k it feeds into the ceiling check below).
                logger.warning(
                    f"[P8 lambda={lam} seed={seed}] a k in the grid hit the {cfg.n_epochs}-epoch cap without "
                    f"early stopping — re-running at {raised_epochs_cfg.n_epochs} epochs"
                )
                raised_record = run_lambda_seed(lam, seed, raised_epochs_cfg, n)
                raised_path = RESULTS_DIR / f"{prefix}lambda{lam}_seed{seed}_epochs{raised_epochs_cfg.n_epochs}.json"
                _save(raised_path, raised_record)  # land immediately — standing clause (a)
                epoch_raises.append(
                    {
                        "lambda": lam,
                        "seed": seed,
                        "original_selected_k": record["selected_k"],
                        "raised_selected_k": raised_record["selected_k"],
                        "raised_hit_cap": raised_record["hit_cap"],
                    }
                )
                record = raised_record  # the raised run is authoritative for this cell

            if record["ceiling_bound"]:
                logger.warning(f"[P8 lambda={lam} seed={seed}] selected_k hit the K_GRID ceiling ({record['selected_k']}) — re-running at K_MAX_RAISED={K_MAX_RAISED}")
                raised_grid = (*K_GRID, K_MAX_RAISED) if K_MAX_RAISED not in K_GRID else K_GRID
                raised_record = run_lambda_seed(lam, seed, cfg, n, k_grid=raised_grid)
                raised_path = RESULTS_DIR / f"{prefix}lambda{lam}_seed{seed}_kmax{K_MAX_RAISED}.json"
                _save(raised_path, raised_record)  # land immediately — standing clause (a)
                ceiling_raises.append({"lambda": lam, "seed": seed, "original_selected_k": record["selected_k"], "raised_selected_k": raised_record["selected_k"]})
                record = raised_record  # the raised run is authoritative for this cell

            selected_k_by_lambda_seed[str(lam)][str(seed)] = record["selected_k"]
            any_hit_cap_overall = any_hit_cap_overall or record["hit_cap"]

    # selection_moved: for each seed, does the selected k differ across lambda? (per-seed curve, same
    # draw/split held fixed across lambda for that seed — the single-difference contrast).
    moved_per_seed = {}
    for seed in SEEDS:
        ks_across_lambda = {str(lam): selected_k_by_lambda_seed[str(lam)][str(seed)] for lam in LAMBDA_GRID}
        moved_per_seed[str(seed)] = len(set(ks_across_lambda.values())) > 1
    selection_moved = any(moved_per_seed.values())

    frozen = {
        "task": "P8",
        "toy": TOY.value,
        "lambda_grid": list(LAMBDA_GRID),
        "seeds": list(SEEDS),
        "k_grid": list(K_GRID),
        "selection_rule": "cheapest_within_tolerance (automl_package/utils/capacity_selection.py), 2x bootstrap SE, n_boot=1000",
        "selected_k_by_lambda_seed": selected_k_by_lambda_seed,
        "moved_per_seed": moved_per_seed,
        "selection_moved": selection_moved,
        "any_hit_cap": any_hit_cap_overall,
        "epoch_raises": epoch_raises,
        "ceiling_raises": ceiling_raises,
        "outcome": (
            "MOVES — strand-local BLOCK on ProbReg's battery reads (P4/P6); log prominently; do not "
            "proceed to P4/P6 until re-derived (MASTER Decision 21, block semantics 2026-07-21)."
            if selection_moved
            else "DOES NOT MOVE — robustness note; P6's report MUST cite this file."
        ),
    }
    frozen_path = RESULTS_DIR / f"{prefix}frozen.json"
    _save(frozen_path, frozen)  # land immediately — standing clause (a)
    logger.info(f"[P8] selection_moved={selection_moved} — {frozen['outcome']}")
    logger.info(f"[P8] wrote {frozen_path}")


if __name__ == "__main__":
    main()
