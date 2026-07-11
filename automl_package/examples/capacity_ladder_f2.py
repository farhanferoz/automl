"""F2 — nested-depth training + coherence bar on the input-varying-capacity toys.

(docs/plans/capacity_ladder_2026-07-09/EXECUTION_PLAN.md WS2, F2 section)

Trains the (already-implemented, already-tested) `NESTED` layer-selection strategy on toys
G / G-flat / H (`_capacity_ladder_toys.py`, validated by F1): per-sample depth
d ~ Uniform{1..max_hidden_layers} as a training schedule, shared trunk, BN off. Reads the
eval-time all-depth per-sample Gaussian log-likelihood table via
`model.strategy.all_depth_log_likelihood` (one forward pass, `test_nested_depth_strategy.py`
is the authoritative example this driver mirrors for construction).

Toy configs and hyperparameters (N_train/N_test, x-range, hidden_size, learning_rate,
activation) are the same as `capacity_ladder_f1_validation.py`'s fixed-capacity MLP sweep;
only max_hidden_layers/n_epochs/layer_selection_method differ (F1 swept depth as separate
nets at 300 epochs; F2 trains one shared-trunk NESTED net, plus a per-depth fixed-baseline
sweep for the coherence bar, at 800 epochs).

Three registered reads (EXECUTION_PLAN.md F2 section):

  B-coh (nesting-costless bar) — per depth d, the nested model's held-out per-sample
    Gaussian log-likelihood at depth d must be within 2*SE (paired bootstrap, G1 plain
    i.i.d. resample) of (a) a SEPARATELY-trained fixed-depth baseline (a NoneStrategy
    FlexibleHiddenLayersNN with max_hidden_layers=d) and (b) the independent-weights
    control (`IndependentWeightsFlexibleNN` trained with the same NESTED schedule — every
    depth is a genuinely separate network there, so its all-depth score table is built by
    hand from `model.model.independent_networks`, not `model.strategy.all_depth_log_
    likelihood` — that method exists on the shared-trunk `NestedStrategy` only, not on
    `IndependentWeightsNestedStrategy`).

  B-order — a cross-seed trunk-stability diagnostic: for each depth d, the Pearson
    correlation across seeds of the depth-d mean prediction, evaluated at a shared,
    seed-independent reference grid over x (not each seed's own noisy test draw, which
    would not align sample-for-sample across seeds). Reported only; instability is a
    registered fallback trigger, not something this driver acts on.

  B-knee — NOT a registered F2 bar (EXECUTION_PLAN.md scopes global knee/stacking to F3,
    "same battery as K1/K2 on the F2 ladder"). Included here anyway per explicit dispatch
    instructions, computed per-seed on the nested score table via the existing `cl.knee`
    reader; cross-seed reproducibility of r_star is reported, not gated.

Usage:
    AUTOML_DEVICE=cpu OMP_NUM_THREADS=3 ~/dev/.venv/bin/python -u automl_package/examples/capacity_ladder_f2.py --smoke
    AUTOML_DEVICE=cpu OMP_NUM_THREADS=3 ~/dev/.venv/bin/python -u automl_package/examples/capacity_ladder_f2.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import torch

_EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _EXAMPLES_DIR)
sys.path.insert(0, os.path.dirname(os.path.dirname(_EXAMPLES_DIR)))  # repo root, so `import automl_package` works

import _capacity_ladder as cl  # noqa: E402
import _capacity_ladder_toys as toys  # noqa: E402

from automl_package.enums import ActivationFunction, DepthRegularization, LayerSelectionMethod, TaskType, UncertaintyMethod  # noqa: E402
from automl_package.models.flexible_neural_network import FlexibleHiddenLayersNN  # noqa: E402
from automl_package.models.independent_weights_flexible_neural_network import IndependentWeightsFlexibleNN  # noqa: E402

RESULTS_DIR = os.path.join(_EXAMPLES_DIR, "capacity_ladder_results", "F2")

TOY_GENERATORS = {"G": toys.make_toy_g, "G_flat": toys.make_toy_g_flat, "H": toys.make_toy_h}
TOYS = list(TOY_GENERATORS)
SEEDS = [0, 1, 2]
MAX_DEPTH = 6
C_GRID = list(range(1, MAX_DEPTH + 1))

# F1's fixed hyperparameters (`capacity_ladder_f1_validation.py`), reused verbatim; only
# max_hidden_layers/n_epochs/layer_selection_method change for F2 (per dispatch instructions).
N_TRAIN = 1000
N_TEST = 500
HIDDEN_SIZE = 24
LEARNING_RATE = 5e-3
ACTIVATION = ActivationFunction.RELU
N_EPOCHS = 800

_BOOT_N = 1000
_BOOT_SEED = 0
_QUERY_GRID_N = 201  # shared, seed-independent reference grid for the B-order cross-seed read


@dataclass
class RunConfig:
    """One battery configuration (full or smoke)."""

    toy_names: list[str]
    seeds: list[int]
    max_depth: int
    n_train: int
    n_test: int
    n_epochs: int
    hidden_size: int
    learning_rate: float
    results_dir: str


# ---------------------------------------------------------------------------
# Model construction (shared builder for nested / control / fixed-depth baseline).
# ---------------------------------------------------------------------------


def _build_model(
    model_cls: type[FlexibleHiddenLayersNN] | type[IndependentWeightsFlexibleNN],
    max_hidden_layers: int,
    layer_selection_method: LayerSelectionMethod,
    seed: int,
    n_epochs: int,
    hidden_size: int,
    learning_rate: float,
) -> FlexibleHiddenLayersNN | IndependentWeightsFlexibleNN:
    """Builds one 1-D regression FlexNN-family model, BN off, strictly probabilistic (no depth penalty)."""
    return model_cls(
        task_type=TaskType.REGRESSION,
        max_hidden_layers=max_hidden_layers,
        hidden_size=hidden_size,
        activation=ACTIVATION,
        layer_selection_method=layer_selection_method,
        n_predictor_layers=0,
        use_batch_norm=False,
        uncertainty_method=UncertaintyMethod.PROBABILISTIC,
        depth_regularization=DepthRegularization.NONE,
        output_size=1,
        input_size=1,
        learning_rate=learning_rate,
        n_epochs=n_epochs,
        random_seed=seed,
        calculate_feature_importance=False,
    )


# ---------------------------------------------------------------------------
# Per-sample Gaussian log-likelihood extraction (nested's own reader for the shared-trunk
# model; hand-rolled duplicates of the SAME formula for the control and fixed baselines,
# since only `NestedStrategy` exposes `all_depth_log_likelihood`).
# ---------------------------------------------------------------------------


def _gaussian_log_likelihood(mean: torch.Tensor, log_var: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Per-example Gaussian log-likelihood — the same formula `NestedStrategy.all_depth_log_likelihood` uses."""
    variance = torch.exp(log_var)
    return -0.5 * (math.log(2.0 * math.pi) + log_var + (y - mean) ** 2 / variance)


def _nested_all_depth_log_likelihood(model: FlexibleHiddenLayersNN, x_t: torch.Tensor, y_t: torch.Tensor) -> np.ndarray:
    """(N, max_depth) held-out score table for the shared-trunk NESTED model, via its own reader."""
    model.model.eval()
    x_dev, y_dev = x_t.to(model.device), y_t.to(model.device)
    with torch.no_grad():
        table = model.strategy.all_depth_log_likelihood(x_dev, y_dev)
    return table.cpu().numpy()


def _independent_all_depth_log_likelihood(model: IndependentWeightsFlexibleNN, x_t: torch.Tensor, y_t: torch.Tensor) -> np.ndarray:
    """(N, max_depth) held-out score table for the independent-weights control.

    `IndependentWeightsNestedStrategy` has no `all_depth_log_likelihood` (every depth is a
    genuinely separate network, not a cached prefix) — mirrors `test_nested_depth_strategy.py`'s
    `model.model.independent_networks` convention instead of the shared-trunk reader.
    """
    model.model.eval()
    x_dev = x_t.to(model.device)
    y_dev = y_t.to(model.device).reshape(-1)
    cols = []
    with torch.no_grad():
        for net in model.model.independent_networks:
            out = net(x_dev)
            cols.append(_gaussian_log_likelihood(out[:, 0], out[:, 1], y_dev))
    return torch.stack(cols, dim=1).cpu().numpy()


def _fixed_depth_log_likelihood(model: FlexibleHiddenLayersNN, x_t: torch.Tensor, y_t: torch.Tensor) -> np.ndarray:
    """(N,) held-out score vector for one separately-trained fixed-depth (NoneStrategy) baseline."""
    model.model.eval()
    x_dev = x_t.to(model.device)
    y_dev = y_t.to(model.device).reshape(-1)
    with torch.no_grad():
        final_output, _, _, _, _ = model.model(x_dev)
    return _gaussian_log_likelihood(final_output[:, 0], final_output[:, 1], y_dev).cpu().numpy()


def _nested_all_depth_means(model: FlexibleHiddenLayersNN, query_x_t: torch.Tensor) -> np.ndarray:
    """(Q, max_depth) per-depth mean prediction at a shared query grid (the B-order read)."""
    model.model.eval()
    q_dev = query_x_t.to(model.device)
    with torch.no_grad():
        all_outputs = model.strategy.all_depth_outputs(q_dev)  # (Q, max_depth, 2)
    return all_outputs[:, :, 0].cpu().numpy()


# ---------------------------------------------------------------------------
# Bootstrap SE (G1 plain i.i.d. row bootstrap) — reuses `_capacity_ladder._bootstrap_col_means`,
# same convention as `capacity_ladder_k1k2k3.py`'s `_plain_boot_se`, not reimplemented.
# ---------------------------------------------------------------------------


def _plain_boot_se(vec: np.ndarray, n_boot: int = _BOOT_N, seed: int = _BOOT_SEED) -> float:
    rng = np.random.default_rng(seed)
    boot = cl._bootstrap_col_means(vec.reshape(-1, 1), n_boot, None, rng)
    return float(boot[:, 0].std(ddof=1))


def _paired_bootstrap_check(nested_col: np.ndarray, baseline_col: np.ndarray) -> dict:
    """B-coh's per-depth check: |mean(delta)| < 2*SE(delta), delta = nested - baseline (paired)."""
    delta = nested_col - baseline_col
    mean_delta = float(delta.mean())
    se = _plain_boot_se(delta)
    return {"mean_delta": mean_delta, "se": se, "pass": bool(abs(mean_delta) < 2.0 * se)}


def _jsonable(obj: object) -> object:
    """Recursively converts numpy/torch scalars and arrays to plain Python/JSON types (K1K2K3 convention)."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


# ---------------------------------------------------------------------------
# B-order: cross-seed trunk-stability diagnostic at a shared query grid.
# ---------------------------------------------------------------------------


def _b_order(nested_models: dict[int, FlexibleHiddenLayersNN], query_x: np.ndarray, max_depth: int) -> dict:
    query_x_t = torch.as_tensor(query_x, dtype=torch.float32)
    means_by_seed = {seed: _nested_all_depth_means(m, query_x_t) for seed, m in nested_models.items()}
    seeds_sorted = sorted(means_by_seed)
    per_depth_corr: dict[int, dict[str, float]] = {}
    for depth in range(1, max_depth + 1):
        pairwise = {}
        for i in range(len(seeds_sorted)):
            for j in range(i + 1, len(seeds_sorted)):
                si, sj = seeds_sorted[i], seeds_sorted[j]
                corr = float(np.corrcoef(means_by_seed[si][:, depth - 1], means_by_seed[sj][:, depth - 1])[0, 1])
                pairwise[f"{si}-{sj}"] = corr
        per_depth_corr[depth] = pairwise
    return {
        "n_query": int(query_x.shape[0]),
        "query_x_range": [float(query_x.min()), float(query_x.max())],
        "per_depth_pairwise_corr": per_depth_corr,
    }


# ---------------------------------------------------------------------------
# One (toy, seed) unit: nested fit + control fit + max_depth fixed-depth fits.
# ---------------------------------------------------------------------------


def run_one_case(toy: str, seed: int, cfg: RunConfig, timing: list[tuple[str, float]]) -> tuple[dict, FlexibleHiddenLayersNN]:
    """Trains all models for one (toy, seed), runs B-coh + B-knee, saves the per-case artifact.

    Returns the case's result dict and the fitted nested model (kept by the caller for the
    toy-level B-order cross-seed read).
    """
    make_fn = TOY_GENERATORS[toy]
    x_tr, y_tr = make_fn(n=cfg.n_train, seed=seed)
    x_te, y_te = make_fn(n=cfg.n_test, seed=seed + 500)
    x_te_t = torch.as_tensor(x_te, dtype=torch.float32)
    y_te_t = torch.as_tensor(y_te, dtype=torch.float32)

    t0 = time.time()
    nested_model = _build_model(FlexibleHiddenLayersNN, cfg.max_depth, LayerSelectionMethod.NESTED, seed, cfg.n_epochs, cfg.hidden_size, cfg.learning_rate)
    nested_model.fit(x_tr, y_tr)
    timing.append((f"nested toy={toy} seed={seed}", time.time() - t0))
    nested_score = _nested_all_depth_log_likelihood(nested_model, x_te_t, y_te_t)

    t0 = time.time()
    control_model = _build_model(IndependentWeightsFlexibleNN, cfg.max_depth, LayerSelectionMethod.NESTED, seed, cfg.n_epochs, cfg.hidden_size, cfg.learning_rate)
    control_model.fit(x_tr, y_tr)
    timing.append((f"control toy={toy} seed={seed}", time.time() - t0))
    control_score = _independent_all_depth_log_likelihood(control_model, x_te_t, y_te_t)

    if nested_score.shape != (cfg.n_test, cfg.max_depth):
        raise AssertionError(f"nested_score shape {nested_score.shape} != {(cfg.n_test, cfg.max_depth)}")
    if control_score.shape != (cfg.n_test, cfg.max_depth):
        raise AssertionError(f"control_score shape {control_score.shape} != {(cfg.n_test, cfg.max_depth)}")

    fixed_depth_ll: dict[int, np.ndarray] = {}
    for depth in range(1, cfg.max_depth + 1):
        t0 = time.time()
        fixed_model = _build_model(FlexibleHiddenLayersNN, depth, LayerSelectionMethod.NONE, seed, cfg.n_epochs, cfg.hidden_size, cfg.learning_rate)
        fixed_model.fit(x_tr, y_tr)
        timing.append((f"fixed_depth={depth} toy={toy} seed={seed}", time.time() - t0))
        fixed_depth_ll[depth] = _fixed_depth_log_likelihood(fixed_model, x_te_t, y_te_t)

    b_coh_vs_fixed = {depth: _paired_bootstrap_check(nested_score[:, depth - 1], fixed_depth_ll[depth]) for depth in range(1, cfg.max_depth + 1)}
    b_coh_vs_control = {depth: _paired_bootstrap_check(nested_score[:, depth - 1], control_score[:, depth - 1]) for depth in range(1, cfg.max_depth + 1)}

    r_star, delta_curve, se = cl.knee(nested_score, ref_c=1, n_boot=_BOOT_N, block=None, c_grid=C_GRID, seed=_BOOT_SEED)
    b_knee = {"r_star": int(r_star), "delta_curve": delta_curve, "se": se}

    case_result = {
        "toy": toy,
        "seed": seed,
        "n": int(nested_score.shape[0]),
        "b_coh_vs_fixed": b_coh_vs_fixed,
        "b_coh_vs_control": b_coh_vs_control,
        "b_knee": b_knee,
    }

    save_path = os.path.join(cfg.results_dir, f"nested_toy{toy}_seed{seed}.pt")
    torch.save(
        {
            "score": torch.as_tensor(nested_score, dtype=torch.float64),
            "x": torch.as_tensor(np.asarray(x_te, dtype=np.float64).ravel()),
            "y_te": torch.as_tensor(np.asarray(y_te, dtype=np.float64)),
            "c_grid": C_GRID,
            "control_score": torch.as_tensor(control_score, dtype=torch.float64),
            "fixed_depth_ll": {d: torch.as_tensor(v, dtype=torch.float64) for d, v in fixed_depth_ll.items()},
        },
        save_path,
    )
    print(f"  saved {save_path}")

    return case_result, nested_model


def main() -> None:
    """Runs the F2 battery (or `--smoke`'s tiny stand-in) and writes the per-case + summary artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Tiny config (1 toy, 1 seed, 3 epochs) to prove the script runs end-to-end.")
    parser.add_argument("--toys", default=None, help="Comma-separated subset of toys to run (default all); for sharded parallel runs.")
    parser.add_argument("--seeds", default=None, help="Comma-separated subset of seeds to run (default all); for sharded parallel runs.")
    args = parser.parse_args()

    toy_names = TOYS if args.toys is None else [t.strip() for t in args.toys.split(",")]
    seeds = SEEDS if args.seeds is None else [int(s) for s in args.seeds.split(",")]
    # Shard-specific summary name so concurrent (toy,seed)-sharded runs don't clobber f2_summary.json;
    # the per-case nested_toy{T}_seed{S}.pt artifacts (what F3 reads) are uniquely keyed regardless.
    summary_tag = "" if (args.toys is None and args.seeds is None) else "_" + "-".join(toy_names) + "_s" + "".join(str(s) for s in seeds)

    if args.smoke:
        cfg = RunConfig(
            toy_names=["G"], seeds=[0], max_depth=MAX_DEPTH, n_train=N_TRAIN, n_test=N_TEST,
            n_epochs=3, hidden_size=HIDDEN_SIZE, learning_rate=LEARNING_RATE, results_dir=RESULTS_DIR,
        )
        summary_tag = ""
    else:
        cfg = RunConfig(
            toy_names=toy_names, seeds=seeds, max_depth=MAX_DEPTH, n_train=N_TRAIN, n_test=N_TEST,
            n_epochs=N_EPOCHS, hidden_size=HIDDEN_SIZE, learning_rate=LEARNING_RATE, results_dir=RESULTS_DIR,
        )

    os.makedirs(cfg.results_dir, exist_ok=True)
    t_start = time.time()
    timing: list[tuple[str, float]] = []

    per_case = []
    b_knee_by_toy: dict[str, dict[int, int]] = {toy: {} for toy in cfg.toy_names}
    b_order_by_toy: dict[str, dict] = {}

    for toy in cfg.toy_names:
        nested_models_by_seed: dict[int, FlexibleHiddenLayersNN] = {}
        for seed in cfg.seeds:
            print(f"=== toy={toy} seed={seed} ===")
            case_result, nested_model = run_one_case(toy, seed, cfg, timing)
            per_case.append(case_result)
            b_knee_by_toy[toy][seed] = case_result["b_knee"]["r_star"]
            nested_models_by_seed[seed] = nested_model
            n_pass_fixed = sum(1 for v in case_result["b_coh_vs_fixed"].values() if v["pass"])
            n_pass_control = sum(1 for v in case_result["b_coh_vs_control"].values() if v["pass"])
            print(
                f"  b_knee r_star={case_result['b_knee']['r_star']}  "
                f"b_coh_vs_fixed passes={n_pass_fixed}/{cfg.max_depth}  b_coh_vs_control passes={n_pass_control}/{cfg.max_depth}"
            )

        query_x = np.linspace(-1.0, 1.0, _QUERY_GRID_N, dtype=np.float32).reshape(-1, 1)
        b_order_by_toy[toy] = _b_order(nested_models_by_seed, query_x, cfg.max_depth)
        del nested_models_by_seed

    wall_total = time.time() - t_start

    b_knee_coherence = {toy: {"r_star_by_seed": r, "coherent": len(set(r.values())) == 1} for toy, r in b_knee_by_toy.items()}
    b_coh_overall_pass = all(v["pass"] for case in per_case for v in list(case["b_coh_vs_fixed"].values()) + list(case["b_coh_vs_control"].values()))

    summary = {
        "config": {
            "toys": cfg.toy_names, "seeds": cfg.seeds, "max_depth": cfg.max_depth, "n_train": cfg.n_train,
            "n_test": cfg.n_test, "n_epochs": cfg.n_epochs, "hidden_size": cfg.hidden_size, "learning_rate": cfg.learning_rate,
        },
        "per_case": per_case,
        "b_knee_cross_seed": b_knee_coherence,
        "b_order": b_order_by_toy,
        "checks": {
            "B_coh_all_depths_all_toys_pass": b_coh_overall_pass,
        },
        "wall_time_sec": wall_total,
        "note": "B-knee is not a registered F2 bar (EXECUTION_PLAN.md scopes it to F3); included per explicit dispatch instructions, reported not gated.",
    }

    summary_path = os.path.join(cfg.results_dir, f"f2_summary{summary_tag}.json")
    with open(summary_path, "w") as f:
        json.dump(_jsonable(summary), f, indent=2)
    print(f"wrote {summary_path}")

    print("\n--- timing ---")
    for label, secs in timing:
        print(f"  {label}: {secs:.3f}s")
    total_fit_time = sum(s for _, s in timing)
    n_fits_per_case = cfg.max_depth + 2  # nested + control + max_depth fixed-depth baselines
    per_epoch = total_fit_time / (len(timing) * cfg.n_epochs)
    one_unit_at_800 = per_epoch * 800 * n_fits_per_case
    print(f"total fit wall-time this run: {total_fit_time:.2f}s over {len(timing)} fits at n_epochs={cfg.n_epochs}")
    print(f"extrapolated: one (toy, seed) unit ({n_fits_per_case} fits) at 800 epochs ~= {one_unit_at_800:.1f}s ({one_unit_at_800 / 60.0:.1f} min)")
    print(f"total wall-time: {wall_total:.1f}s")


if __name__ == "__main__":
    main()
