"""T1 — the provably-deep-required toy: the depth-lane discriminator.

(docs/plans/perinput_selector_2026-07-10/EXECUTION_PLAN.md WS-B, task T1; pre-registration in
capacity_ladder_results/T1/PREREGISTRATION.md)

F2's depth lane (`capacity_ladder_f2.py`, `capacity_ladder_results/F2/`) measured a NULL per-input
depth signal on toys G/G-flat/H — but toy G's compositional region (a sine modulated by a linear
envelope) turned out to carry an incidentally tiny per-input depth requirement, so the null could
mean either "the machinery can't see per-input depth structure" or "there was never much structure
to see." T1 builds a toy where the depth requirement is PROVABLE by construction (a Telgarsky-style
composed tent map, not merely "harder to fit smoothly"), so the depth-lane machinery is tested where
a large signal is guaranteed to exist. `toy T1` (`make_toy_t1`, `_capacity_ladder_toys.py`): region A
(x < 0.5) is linear (y = 1.5x, depth-1-sufficient); region B (x >= 0.5) is a 5-fold-iterated tent map
(32 linear pieces) — at trunk width 8 a depth-1 net realizes at most ~8 linear kinks, provably short
of 32, while depth >= 3 suffices.

Harness (reused verbatim from the certified capacity-ladder scripts, not reinvented):
  * `capacity_ladder_f2.py` — `RunConfig`, `_build_model`, `_nested_all_depth_log_likelihood`,
    `_independent_all_depth_log_likelihood`, `_fixed_depth_log_likelihood`, `_plain_boot_se`,
    `_jsonable`. Same nested-depth FlexibleHiddenLayersNN training loop + dedicated fixed-depth
    sweep, depths 1..6, BN off, 3 seeds. ONLY `hidden_size` is overridden (8, not F2's 24) — T1's
    provable-capacity argument (~hidden_size kinks per depth-1 net) is pinned to trunk width 8 by
    the EXECUTION_PLAN.md T1 spec.
  * `capacity_ladder_x3.run_repeated_crossfit` / `capacity_ladder_x1.run_repeated` (which itself
    calls `capacity_ladder_x1.hierarchical_perbin_stack`) — called UNMODIFIED at `n_bins=2` for
    bar (ii)'s pooled "A-vs-B contrast" (region A / region B is, for a toy uniform on [0, 1], the
    same 50/50 split `quantile_bins(x, 2)` already computes).
  * `_capacity_ladder.py` — `stack_em`, `perbin_stack`, `mixture_logscore`, `quantile_bins`,
    `_bootstrap_col_means` — the SAME primitives the two pooled wrappers above are built from. This
    script also re-assembles them into a PER-REGION-decomposed reader (`_split_perbin_advantage`,
    `_nb_aggregate`, `_repeated_perbin_by_region`) as a DIAGNOSTIC (not gated) — see "KNOWN READOUT
    PROPERTY" below — and for the "terciles within region B" secondary read.

Two pre-registered bars (full text + outcome semantics in PREREGISTRATION.md):
  (i) CONSTRUCTION — the dedicated fixed-depth sweep, read per region: region B's held-out NLL
      must improve > 2*SE at d1->d2 AND d2->d3; region A must stay flat (no > 2*SE gain past d1).
  (ii) PER-INPUT READ — the pooled "A-vs-B contrast" (`run_repeated_crossfit` / `run_repeated`'s
      `hier_vs_global`, `n_bins=2`) must be significant (corrected-t > 2) on BOTH the independent
      (crossfit-style) and hierarchical readers, on >= 2/3 seeds, AND T1's pass rate must exceed
      the G-flat negative-control band (F2's G-flat table, re-read through the SAME machinery —
      X3/X1's own G-flat summaries use a different bin scheme and are not reused here).

KNOWN READOUT PROPERTY (discovered while building this script's selftest; verified against the
real, unmodified `capacity_ladder_x3.run_repeated_crossfit` / `capacity_ladder_x1.run_repeated`,
not an artifact of this script): `stack_em`'s single shared "global" mixture is fit by maximizing
TOTAL log-likelihood pooled across every region, so it converges onto whichever region has the
LARGER nat-scale depth gaps — by T1's design, that is region B (the 32-piece tent map). Once the
global mixture already matches region B's own optimum, region B's OWN per-bin stack shows close to
ZERO advantage over it, REGARDLESS of how large region B's true depth requirement is; a genuinely
flat region A (no competing depth preference at all — the expected behaviour of a linear function,
which is fit equally well at any depth >= 1) cannot pull the global mixture away from region B
either. A minimal synthetic check (true-flat region A + a planted +0.8-nat region-B advantage) reads
`t_corrected` in [-0.6, +0.3] on BOTH readers despite the large planted signal — see `run_selftest`
part (c), printed every run as a non-gated diagnostic. Practical consequence: bar (ii) can plausibly
read NULL on the real T1 toy even when bar (i) shows a large, genuine region-B depth requirement;
`main()`'s `ab_decomposed_by_seed_DIAGNOSTIC_NOT_GATED` field (region-A vs region-B advantage read
SEPARATELY, not pooled) is what tells that case apart from an actual machinery failure — a near-zero
(not negative) region-B decomposed advantage alongside a real CONSTRUCTION gain is consistent with
this documented property, not evidence the depth-lane machinery itself is broken.

Usage:
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python automl_package/examples/capacity_ladder_t1.py --selftest
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python automl_package/examples/capacity_ladder_t1.py --smoke
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python automl_package/examples/capacity_ladder_t1.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

_EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _EXAMPLES_DIR)
sys.path.insert(0, os.path.dirname(os.path.dirname(_EXAMPLES_DIR)))  # repo root, so `import automl_package` works

import _capacity_ladder as cl  # noqa: E402
import _capacity_ladder_toys as toys  # noqa: E402
import capacity_ladder_f3 as f3mod  # noqa: E402  (load_f2_table, for the G-flat control band)
import capacity_ladder_x1 as x1mod  # noqa: E402  (hierarchical_perbin_stack, run_repeated)
import capacity_ladder_x3 as x3mod  # noqa: E402  (run_repeated_crossfit)
from capacity_ladder_f2 import (  # noqa: E402
    RunConfig,
    _build_model,
    _fixed_depth_log_likelihood,
    _independent_all_depth_log_likelihood,
    _jsonable,
    _nested_all_depth_log_likelihood,
    _plain_boot_se,
)

from automl_package.enums import LayerSelectionMethod  # noqa: E402
from automl_package.models.flexible_neural_network import FlexibleHiddenLayersNN  # noqa: E402
from automl_package.models.independent_weights_flexible_neural_network import IndependentWeightsFlexibleNN  # noqa: E402

RESULTS_DIR = os.path.join(_EXAMPLES_DIR, "capacity_ladder_results", "T1")

SEEDS = [0, 1, 2]
MAX_DEPTH = 6
C_GRID = list(range(1, MAX_DEPTH + 1))
N_TRAIN = 1600
N_TEST = 500
# NOT F2's HIDDEN_SIZE=24: the toy's provable-capacity claim (a depth-1 net realizes at most
# ~hidden_size linear kinks, provably short of region B's 32 pieces) is pinned to trunk width 8 by
# the EXECUTION_PLAN.md T1 spec. Every other hyperparameter below is F2's, reused verbatim.
HIDDEN_SIZE = 8
LEARNING_RATE = 5e-3
N_EPOCHS = 800

N_SPLITS = 50  # X3/X1 repeated-cross-fit convention, reused
TORCH_THREADS = 2  # x1.py's convention for the hierarchical EM post-hoc reads (not model training)
_BOOT_N = 1000


# ---------------------------------------------------------------------------
# CONSTRUCTION bar (i): per-region read of the dedicated fixed-depth sweep.
# ---------------------------------------------------------------------------


def _region_gain(fixed_depth_ll: dict[int, np.ndarray], mask: np.ndarray, d_lo: int, d_hi: int) -> dict:
    """One region's held-out-NLL gain from depth `d_lo` to `d_hi` (paired plain bootstrap SE, F2 convention)."""
    delta = fixed_depth_ll[d_hi][mask] - fixed_depth_ll[d_lo][mask]
    mean_delta = float(delta.mean())
    se = _plain_boot_se(delta)
    return {"mean_delta": mean_delta, "se": se, "beats_2se": bool(mean_delta > 2.0 * se)}


def _construction_bar(fixed_depth_ll: dict[int, np.ndarray], region: np.ndarray) -> dict:
    """CONSTRUCTION bar (i): dedicated fixed-depth sweep, per region.

    Region B (x >= 0.5, the composed tent map): d1->d2 AND d2->d3 must each improve held-out NLL by
    > 2*SE. Region A (x < 0.5, linear): no depth past d1 may improve by > 2*SE (flat).
    """
    mask_a = region == 0
    mask_b = region == 1

    region_b_gains = {"d1_to_d2": _region_gain(fixed_depth_ll, mask_b, 1, 2), "d2_to_d3": _region_gain(fixed_depth_ll, mask_b, 2, 3)}
    region_b_pass = region_b_gains["d1_to_d2"]["beats_2se"] and region_b_gains["d2_to_d3"]["beats_2se"]

    region_a_gains = {f"d1_to_d{d}": _region_gain(fixed_depth_ll, mask_a, 1, d) for d in range(2, MAX_DEPTH + 1)}
    region_a_flat = not any(g["beats_2se"] for g in region_a_gains.values())

    return {
        "n_region_a": int(mask_a.sum()),
        "n_region_b": int(mask_b.sum()),
        "region_b_gains": region_b_gains,
        "region_b_pass": bool(region_b_pass),
        "region_a_gains": region_a_gains,
        "region_a_flat": bool(region_a_flat),
        "construction_pass": bool(region_b_pass and region_a_flat),
    }


# ---------------------------------------------------------------------------
# PER-INPUT READ bar (ii) DIAGNOSTIC: repeated cross-fit decomposed PER BIN (not pooled), so
# region B's advantage can be told apart from region A's -- built from the same primitives
# `run_repeated_crossfit` (x3.py) / `run_repeated` (x1.py) use, since those two wrappers only
# expose a POOLED advantage over all held-out rows. The GATED bar (ii) statistic itself calls
# `run_repeated_crossfit` / `run_repeated` directly, unmodified (see `_per_input_bar` below);
# this decomposition is a reported-not-gated diagnostic (module docstring, "KNOWN READOUT
# PROPERTY") and also powers the "terciles within region B" secondary read.
# ---------------------------------------------------------------------------


def _boot_se(vec: np.ndarray, seed: int, n_boot: int = _BOOT_N) -> float:
    """Plain i.i.d. bootstrap SE of a 1-D vector's mean (`_capacity_ladder._bootstrap_col_means`, X3/X1 convention)."""
    rng = np.random.default_rng(seed)
    boot = cl._bootstrap_col_means(vec.reshape(-1, 1), n_boot, None, rng)
    return float(boot[:, 0].std(ddof=1))


def _split_perbin_advantage(score: np.ndarray, x: np.ndarray, n_bins: int, split_seed: int) -> tuple[dict, int, int]:
    """One fit/score split: per-bin (not pooled) advantage over global for the independent and hierarchical stackers.

    Independent arm: x3-style, `cl.perbin_stack`. Hierarchical arm: x1-style, `x1mod.
    hierarchical_perbin_stack`. Mirrors `capacity_ladder_f3._perbin_vs_global` /
    `capacity_ladder_x1.three_arm_split`'s split+fit+score structure, but reports the advantage
    PER BIN rather than pooled over all held-out rows.

    Args:
        score: `(N, C)` held-out log-likelihood table.
        x: `(N,)` inputs.
        n_bins: quantile bins (2 = the A/B region split; 3 = terciles within a region-B subset).
        split_seed: the 50/50 fit/score partition seed.

    Returns:
        `({bin_id: {"indep": {"mean", "se"}, "hier": {"mean", "se"}}}, n_fit, n_score)`.
    """
    n = score.shape[0]
    rng = np.random.default_rng(split_seed)
    perm = rng.permutation(n)
    half = n // 2
    fit_idx, score_idx = perm[:half], perm[half:]
    score_fit, x_fit = score[fit_idx], x[fit_idx]
    score_sc, x_sc = score[score_idx], x[score_idx]

    bins_fit = cl.quantile_bins(x_fit, n_bins)
    bins_sc = cl.quantile_bins(x_sc, n_bins)

    pi_global = cl.stack_em(score_fit)
    pi_indep = cl.perbin_stack(score_fit, bins_fit)
    pi_hier = x1mod.hierarchical_perbin_stack(score_fit, bins_fit)

    global_ls = cl.mixture_logscore(pi_global, score_sc)

    out: dict[int, dict] = {}
    for b in np.unique(bins_sc):
        mask = bins_sc == b
        indep_ls = cl.mixture_logscore(pi_indep.get(int(b), pi_global), score_sc[mask])
        hier_ls = cl.mixture_logscore(pi_hier.get(int(b), pi_global), score_sc[mask])
        diff_indep = indep_ls - global_ls[mask]
        diff_hier = hier_ls - global_ls[mask]
        out[int(b)] = {
            "indep": {"mean": float(diff_indep.mean()), "se": _boot_se(diff_indep, seed=1)},
            "hier": {"mean": float(diff_hier.mean()), "se": _boot_se(diff_hier, seed=2)},
        }
    return out, half, n - half


def _nb_aggregate(means: np.ndarray, n_fit: int, n_score: int, n_splits: int) -> dict:
    """Nadeau-Bengio (2003) corrected-resampled aggregate across splits.

    The SAME formula `capacity_ladder_x3.run_repeated_crossfit` / `capacity_ladder_x1._aggregate` use.
    """
    rho = n_score / n_fit
    var_s = float(np.var(means, ddof=1))
    mu_bar = float(means.mean())
    se_nb = float(np.sqrt(max(0.0, (1.0 / n_splits + rho) * var_s)))
    return {
        "mu_bar": mu_bar,
        "se_nadeau_bengio": se_nb,
        "t_corrected": (mu_bar / se_nb if se_nb > 0 else float("inf")),
        "sd_across_splits": float(np.sqrt(var_s)),
    }


def _repeated_perbin_by_region(score: np.ndarray, x: np.ndarray, n_bins: int, n_splits: int = N_SPLITS) -> dict[int, dict]:
    """Sweeps `_split_perbin_advantage` over `n_splits` splits and NB-aggregates, PER bin PER arm.

    The decomposed analogue of `run_repeated_crossfit`/`run_repeated` T1 needs to tell "significant
    in B" apart from "flat in A" as a diagnostic (see module docstring, "KNOWN READOUT PROPERTY").
    """
    split_rows = [_split_perbin_advantage(score, x, n_bins, s) for s in range(n_splits)]
    n_fit, n_score = split_rows[0][1], split_rows[0][2]
    per_bin = [r[0] for r in split_rows]
    bin_ids = sorted(per_bin[0].keys())

    out: dict[int, dict] = {}
    for b in bin_ids:
        for arm in ("indep", "hier"):
            means = np.array([r[b][arm]["mean"] for r in per_bin], dtype=np.float64)
            beats = np.array([r[b][arm]["mean"] > 2.0 * r[b][arm]["se"] for r in per_bin])
            agg = _nb_aggregate(means, n_fit, n_score, n_splits)
            agg["split_pass_fraction"] = float(beats.mean())
            out.setdefault(b, {})[arm] = agg
    return out


def _control_band(n_splits: int = N_SPLITS) -> dict | None:
    """Re-reads F2's G-flat tables through the SAME pooled `n_bins=2` machinery as the real bar (ii) read.

    Uses `x3mod.run_repeated_crossfit` / `x1mod.run_repeated`, called UNMODIFIED, giving an
    apples-to-apples false-positive band for "T1 pass rate must exceed the G-flat band". F2's
    G-flat tables (`capacity_ladder_results/F2/nested_toyG_flat_seed{0,1,2}.pt`) are the negative
    control per dispatch; X3/X1's own G-flat summaries use a DIFFERENT bin scheme (terciles across
    the whole domain) so they are not directly comparable and are not reused here.
    """
    rows = []
    for seed in SEEDS:
        t = f3mod.load_f2_table("G_flat", seed)
        if t is None:
            continue
        cf = x3mod.run_repeated_crossfit(t["score"], t["x"], n_splits=n_splits, n_bins=2)
        hi = x1mod.run_repeated(t["score"], t["x"], n_splits=n_splits, n_bins=2)
        rows.append({"seed": seed, "crossfit": cf, "hier_vs_global": hi["hier_vs_global"]})
    if not rows:
        return None
    return {
        "per_seed": rows,
        "mean_crossfit_pass_fraction": float(np.mean([r["crossfit"]["split_pass_fraction"] for r in rows])),
        "mean_hier_pass_fraction": float(np.mean([r["hier_vs_global"]["split_pass_fraction"] for r in rows])),
        "n_seeds_crossfit_t_gt_2": int(sum(r["crossfit"]["t_corrected"] > 2.0 for r in rows)),
        "n_seeds_hier_t_gt_2": int(sum(r["hier_vs_global"]["t_corrected"] > 2.0 for r in rows)),
    }


def _per_input_bar(per_input_by_seed: dict[int, dict], control_band: dict | None) -> dict:
    """PER-INPUT READ bar (ii): the pooled "A-vs-B contrast" significant on both readers, on >= 2/3 seeds.

    Uses `x3mod.run_repeated_crossfit` / `x1mod.run_repeated`'s `hier_vs_global`, called
    UNMODIFIED at `n_bins=2`, AND T1's pass rate must exceed the G-flat control band
    (`_control_band`). See the module docstring's "KNOWN READOUT PROPERTY" note: this pooled
    statistic can only be significant if region A ALSO has enough genuine structure to pull the
    global mixture away from region B's own optimum -- if region A is close to true noise, this
    bar can read null even when the CONSTRUCTION bar (i) shows a large, real region-B depth
    requirement. `_repeated_perbin_by_region`'s per-region decomposition (reported in `main()`,
    not gated here) is the diagnostic that tells the two cases apart.
    """
    per_seed_rows = []
    n_both_sig = 0
    for seed, r in per_input_by_seed.items():
        cf_t = r["crossfit"]["t_corrected"]
        hier_t = r["hier_vs_global"]["t_corrected"]
        both_sig = cf_t > 2.0 and hier_t > 2.0
        n_both_sig += int(both_sig)
        per_seed_rows.append({"seed": seed, "crossfit_t": cf_t, "hier_t": hier_t, "both_significant": both_sig})

    n_seeds = len(per_input_by_seed)
    sig_bar = n_both_sig >= 2  # ">= 2/3 seeds" (EXECUTION_PLAN.md T1 bar ii), SEEDS has 3 entries

    exceeds_control = None
    if control_band is not None and per_input_by_seed:
        t1_crossfit_pass = float(np.mean([r["crossfit"]["split_pass_fraction"] for r in per_input_by_seed.values()]))
        t1_hier_pass = float(np.mean([r["hier_vs_global"]["split_pass_fraction"] for r in per_input_by_seed.values()]))
        exceeds_control = bool(t1_crossfit_pass > control_band["mean_crossfit_pass_fraction"] and t1_hier_pass > control_band["mean_hier_pass_fraction"])

    return {
        "n_seeds": n_seeds,
        "n_seeds_both_significant": n_both_sig,
        "per_seed": per_seed_rows,
        "sig_bar_pass": bool(sig_bar),
        "exceeds_gflat_control_band": exceeds_control,
        "per_input_read_pass": bool(sig_bar and (exceeds_control is True)),
    }


# ---------------------------------------------------------------------------
# One (T1, seed) unit: nested fit + control fit + max_depth fixed-depth fits (mirrors
# `capacity_ladder_f2.run_one_case`, F2 table-key convention).
# ---------------------------------------------------------------------------


def run_one_case(seed: int, cfg: RunConfig, timing: list[tuple[str, float]]) -> dict:
    """Trains the nested model, the independent-weights control, and the fixed-depth sweep for one T1 seed.

    Runs the CONSTRUCTION bar and saves the per-case artifact.
    """
    x_tr, y_tr = toys.make_toy_t1(n=cfg.n_train, seed=seed)
    x_te, y_te = toys.make_toy_t1(n=cfg.n_test, seed=seed + 500)
    x_te_t = torch.as_tensor(x_te, dtype=torch.float32)
    y_te_t = torch.as_tensor(y_te, dtype=torch.float32)

    t0 = time.time()
    nested_model = _build_model(FlexibleHiddenLayersNN, cfg.max_depth, LayerSelectionMethod.NESTED, seed, cfg.n_epochs, cfg.hidden_size, cfg.learning_rate)
    nested_model.fit(x_tr, y_tr)
    timing.append((f"nested T1 seed={seed}", time.time() - t0))
    nested_score = _nested_all_depth_log_likelihood(nested_model, x_te_t, y_te_t)

    t0 = time.time()
    control_model = _build_model(IndependentWeightsFlexibleNN, cfg.max_depth, LayerSelectionMethod.NESTED, seed, cfg.n_epochs, cfg.hidden_size, cfg.learning_rate)
    control_model.fit(x_tr, y_tr)
    timing.append((f"control T1 seed={seed}", time.time() - t0))
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
        timing.append((f"fixed_depth={depth} T1 seed={seed}", time.time() - t0))
        fixed_depth_ll[depth] = _fixed_depth_log_likelihood(fixed_model, x_te_t, y_te_t)

    region = toys.toy_t1_region(x_te)
    construction = _construction_bar(fixed_depth_ll, region)

    save_path = os.path.join(cfg.results_dir, f"nested_toyT1_seed{seed}.pt")
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

    return {"seed": seed, "n": int(nested_score.shape[0]), "construction": construction}


def load_t1_table(seed: int) -> dict | None:
    """Loads one T1 nested-ladder score table (F2 table-key convention); `None` if not yet produced.

    The T1 analogue of `capacity_ladder_f3.load_f2_table`.
    """
    path = os.path.join(RESULTS_DIR, f"nested_toyT1_seed{seed}.pt")
    if not os.path.exists(path):
        return None
    d = torch.load(path, weights_only=False)
    score = np.asarray(d["score"], dtype=np.float64)
    x = np.asarray(d["x"], dtype=np.float64).ravel()
    c_grid = list(d["c_grid"])
    if c_grid != C_GRID:
        raise AssertionError(f"{path}: c_grid {c_grid} != {C_GRID}")
    return {"score": score, "x": x}


# ---------------------------------------------------------------------------
# Selftest -- synthetic known-answer table (no training). MUST pass before any real read.
# ---------------------------------------------------------------------------


def _synthetic_t1_table(n_per: int = 1500, noise_sd: float = 0.3, seed: int = 0, means_a: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Known-answer synthetic (N, 6) depth table: region A per `means_a`, region B a planted +0.8-nat step.

    Region B (x >= 0.5) is +0.4 nat at d1->d2 and +0.4 more (+0.8 total at d1->d3), flat past d3.
    Hand-built (not `cl._peaked_table`'s quadratic-peak shape) since the CONSTRUCTION bar needs an
    EXPLICIT step shape (two significant B increments, then flat). `means_a` defaults to true-flat
    noise (no per-depth preference at all); pass an alternative to test a competing region A.
    """
    rng = np.random.default_rng(seed)
    x_a = rng.uniform(0.0, 0.5, n_per)
    x_b = rng.uniform(0.5, 1.0, n_per)
    x = np.concatenate([x_a, x_b])

    if means_a is None:
        means_a = np.zeros(MAX_DEPTH)  # true flat: no per-depth preference at all
    means_b = np.array([0.0, 0.4, 0.8, 0.8, 0.8, 0.8])  # +0.4 at d1->d2, +0.4 more at d2->d3, flat after

    score_a = means_a[None, :] + rng.normal(0.0, noise_sd, size=(n_per, MAX_DEPTH))
    score_b = means_b[None, :] + rng.normal(0.0, noise_sd, size=(n_per, MAX_DEPTH))
    score = np.concatenate([score_a, score_b], axis=0)
    return score, x


def run_selftest() -> bool:
    """Known-answer synthetic checks (no training).

    (a) CONSTRUCTION bar (`_construction_bar`): on the true-flat-A / planted-+0.8-nat-B table,
    region B must show significant d1->d2 AND d2->d3 gains and region A must show no significant
    gain past d1 -- this is the DIRECT, unconfounded per-region NLL-by-depth read T1's bar (i) uses.

    (b) PER-INPUT READ (`x3mod.run_repeated_crossfit` / `x1mod.run_repeated`'s `hier_vs_global`,
    called UNMODIFIED, `n_bins=2`, mirroring X3/X1's OWN selftest philosophy -- a positive control
    needs GENUINELY CONFLICTING regions, not one flat and one dominant): with region A given an
    OPPOSING, comparable-magnitude depth preference (not merely flat), the pooled A-vs-B contrast
    must be clearly significant on BOTH readers -- this validates the read machinery is wired
    correctly (same estimator, same call sites the real read uses).

    (c) DIAGNOSTIC, NOT GATED: re-running (b)'s exact machinery on the (a)-style true-flat-A table
    (the literal region-A behaviour the real T1 toy is expected to show, since a linear function is
    fit equally well at every depth) demonstrates a KNOWN, load-bearing property of the `stack_em`
    "beats global" statistic -- verified here against the real, unmodified x3.py/x1.py functions,
    not a bug in this script: `stack_em`'s global mixture is fit by maximizing TOTAL log-likelihood
    across both regions, so it converges onto whichever region has the LARGER nat-scale depth gaps
    (region B, by T1's design). Once global already matches B's optimum, B's OWN per-bin stack adds
    ~zero advantage over it -- REGARDLESS of how large B's true gap is -- while a truly flat region
    A (no competing preference at all) cannot pull global away from B either, so it also reads null.
    Net effect: bar (ii)'s pooled statistic can plausibly read null on the real T1 toy even though
    bar (i) shows a large, genuine depth requirement in region B. This is reported prominently
    (printed every run) rather than silently gated, since it changes how a null bar (ii) should be
    interpreted -- see the module docstring's "KNOWN READOUT PROPERTY" note.
    """
    torch.set_num_threads(TORCH_THREADS)
    ok = True

    # (a) CONSTRUCTION bar.
    score_flat, x_flat = _synthetic_t1_table()
    fixed_depth_ll = {d: score_flat[:, d - 1] for d in range(1, MAX_DEPTH + 1)}
    region = toys.toy_t1_region(x_flat)
    construction = _construction_bar(fixed_depth_ll, region)
    print(f"[t1 selftest] (a) construction: region_b_pass={construction['region_b_pass']} "
          f"region_a_flat={construction['region_a_flat']} (want both True)")
    ok_a = construction["construction_pass"]
    ok = ok and ok_a

    # (b) PER-INPUT READ machinery check: genuinely conflicting regions (X3/X1 selftest style),
    # via the literal, unmodified wrappers the real bar (ii) read calls.
    means_a_conflict = np.array([0.8, 0.4, 0.0, 0.0, 0.0, 0.0])  # region A strongly prefers LOW depth
    score_conflict, x_conflict = _synthetic_t1_table(means_a=means_a_conflict)
    cf = x3mod.run_repeated_crossfit(score_conflict, x_conflict, n_splits=15, n_bins=2)
    hi = x1mod.run_repeated(score_conflict, x_conflict, n_splits=15, n_bins=2)["hier_vs_global"]
    print(f"[t1 selftest] (b) per-input machinery (conflicting regions, positive control): "
          f"crossfit_t={cf['t_corrected']:.2f} hier_t={hi['t_corrected']:.2f} (want both clearly > 2)")
    ok_b = cf["t_corrected"] > 5.0 and hi["t_corrected"] > 5.0
    ok = ok and ok_b

    # (c) DIAGNOSTIC (printed, not gated): the literal T1-style flat-A/dominant-B construction.
    cf_flat = x3mod.run_repeated_crossfit(score_flat, x_flat, n_splits=15, n_bins=2)
    hi_flat = x1mod.run_repeated(score_flat, x_flat, n_splits=15, n_bins=2)["hier_vs_global"]
    print(f"[t1 selftest] (c) DIAGNOSTIC (true-flat-A / dominant-B, T1's literal region shape): "
          f"crossfit_t={cf_flat['t_corrected']:.2f} hier_t={hi_flat['t_corrected']:.2f} "
          f"-- expected near-null despite the planted +0.8-nat B signal (see module docstring)")

    print(f"[t1 selftest] {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------------
# Real reader.
# ---------------------------------------------------------------------------


def main() -> None:
    """Runs the T1 battery (or `--smoke`'s tiny stand-in / `--selftest`'s synthetic check)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true", help="Synthetic known-answer check (no training); must PASS before any real read.")
    parser.add_argument("--smoke", action="store_true", help="Tiny config (1 seed, 3 epochs) to prove the script runs end-to-end.")
    args = parser.parse_args()

    if args.selftest:
        sys.exit(0 if run_selftest() else 1)

    if args.smoke:
        cfg = RunConfig(
            toy_names=["T1"], seeds=[0], max_depth=MAX_DEPTH, n_train=N_TRAIN, n_test=N_TEST,
            n_epochs=3, hidden_size=HIDDEN_SIZE, learning_rate=LEARNING_RATE, results_dir=RESULTS_DIR,
        )
    else:
        cfg = RunConfig(
            toy_names=["T1"], seeds=SEEDS, max_depth=MAX_DEPTH, n_train=N_TRAIN, n_test=N_TEST,
            n_epochs=N_EPOCHS, hidden_size=HIDDEN_SIZE, learning_rate=LEARNING_RATE, results_dir=RESULTS_DIR,
        )

    os.makedirs(cfg.results_dir, exist_ok=True)
    t_start = time.time()
    timing: list[tuple[str, float]] = []

    per_case = []
    for seed in cfg.seeds:
        print(f"=== T1 seed={seed} ===")
        case_result = run_one_case(seed, cfg, timing)
        per_case.append(case_result)
        c = case_result["construction"]
        print(f"  construction: region_b_pass={c['region_b_pass']} region_a_flat={c['region_a_flat']} overall={c['construction_pass']}")

    print("\n--- per-input read: pooled A-vs-B contrast (x3.run_repeated_crossfit / x1.run_repeated, n_bins=2, unmodified) ---")
    torch.set_num_threads(TORCH_THREADS)
    per_input_by_seed: dict[int, dict] = {}
    ab_decomposed_by_seed: dict[int, dict] = {}  # diagnostic only, not gated -- see module docstring
    b_terciles_by_seed: dict[int, dict] = {}
    for seed in cfg.seeds:
        t = load_t1_table(seed)
        if t is None:
            continue
        cf = x3mod.run_repeated_crossfit(t["score"], t["x"], n_splits=N_SPLITS, n_bins=2)
        hi = x1mod.run_repeated(t["score"], t["x"], n_splits=N_SPLITS, n_bins=2)
        per_input_by_seed[seed] = {"crossfit": cf, "hier_vs_global": hi["hier_vs_global"]}
        print(f"  seed={seed}: pooled crossfit_t={cf['t_corrected']:+.2f} hier_t={hi['hier_vs_global']['t_corrected']:+.2f}")

        ab_decomposed_by_seed[seed] = _repeated_perbin_by_region(t["score"], t["x"], n_bins=2, n_splits=N_SPLITS)
        lo_bin, hi_bin = sorted(ab_decomposed_by_seed[seed].keys())
        print(f"    [diagnostic, not gated] region_B indep_t={ab_decomposed_by_seed[seed][hi_bin]['indep']['t_corrected']:+.2f} "
              f"hier_t={ab_decomposed_by_seed[seed][hi_bin]['hier']['t_corrected']:+.2f}  "
              f"region_A indep_t={ab_decomposed_by_seed[seed][lo_bin]['indep']['t_corrected']:+.2f} "
              f"hier_t={ab_decomposed_by_seed[seed][lo_bin]['hier']['t_corrected']:+.2f}")

        mask_b = toys.toy_t1_region(t["x"]) == 1
        if mask_b.sum() >= 30:
            b_terciles_by_seed[seed] = _repeated_perbin_by_region(t["score"][mask_b], t["x"][mask_b], n_bins=3, n_splits=N_SPLITS)

    print("\ncontrol band: re-reading F2 G-flat tables through the same pooled n_bins=2 machinery...")
    control_band = _control_band(n_splits=N_SPLITS)

    per_input_bar = _per_input_bar(per_input_by_seed, control_band) if per_input_by_seed else None

    n_seeds_construction_pass = sum(1 for c in per_case if c["construction"]["construction_pass"])
    construction_bar = {
        "n_seeds_pass": n_seeds_construction_pass,
        "n_seeds": len(per_case),
        "pass": bool(n_seeds_construction_pass == len(per_case)),
    }

    if construction_bar["pass"] and per_input_bar and per_input_bar["per_input_read_pass"]:
        outcome = "PASS_BOTH: machinery VALIDATED; depth-lane null reframed toy-specific; H2 UNLOCKED."
    elif construction_bar["pass"] and per_input_bar and not per_input_bar["per_input_read_pass"]:
        outcome = (
            "PASS_i_FAIL_ii: bar (ii) null. Interpret via the REWRITTEN regime-contingent semantics in "
            "capacity_ladder_results/T1/PREREGISTRATION.md (adjudicated 2026-07-10, BAR_II_ADJUDICATION.md): "
            "read REGION A's construction NLL-by-depth curve (per_case[*].construction.region_a_gains). "
            "(1) region A INDIFFERENT (gains ~0 across depth) => DOCUMENTED STRUCTURAL ZERO, not a machinery "
            "failure (advantage-over-global cannot see an absolute-only depth requirement; positive control "
            "t~=76 confirms the reader is sound); per-input value is a COMPUTE question deferred to H2 -> STOP, "
            "adjudicator; H2-unlock is a G-FORK. (2) region A ACTIVELY HURT by depth (gains significantly "
            "NEGATIVE) => genuine heterogeneity exists yet readers did not fire = the true machinery-failure "
            "signature -> STOP, adjudicator, H2 stays locked. NOTE: ab_decomposed_by_seed is DIAGNOSTIC ONLY "
            "and is EQUALLY BLIND (shares the pooled global baseline; region-B decomposed t~=-0.84 on the "
            "planted table) -- do NOT use it to distinguish the two cases; use region A's construction curve."
        )
    elif not construction_bar["pass"]:
        outcome = "FAIL_i: redesign (iterate up to 7x, width unchanged), re-run; fails again -> adjudicator."
    else:
        outcome = "INCOMPLETE: per-input read not available (missing tables)."

    wall_total = time.time() - t_start
    summary = {
        "config": {
            "seeds": cfg.seeds, "max_depth": cfg.max_depth, "n_train": cfg.n_train, "n_test": cfg.n_test,
            "n_epochs": cfg.n_epochs, "hidden_size": cfg.hidden_size, "learning_rate": cfg.learning_rate,
        },
        "per_case": per_case,
        "construction_bar": construction_bar,
        "per_input_by_seed": _jsonable(per_input_by_seed),
        "ab_decomposed_by_seed_DIAGNOSTIC_NOT_GATED": _jsonable(ab_decomposed_by_seed),
        "b_terciles_by_seed": _jsonable(b_terciles_by_seed),
        "control_band_gflat": _jsonable(control_band),
        "per_input_bar": per_input_bar,
        "outcome": outcome,
        "wall_time_sec": wall_total,
    }

    if not args.smoke:
        summary_path = os.path.join(cfg.results_dir, "t1_summary.json")
        with open(summary_path, "w") as f:
            json.dump(_jsonable(summary), f, indent=2)
        print(f"\nwrote {summary_path}")

    print("\n--- timing ---")
    for label, secs in timing:
        print(f"  {label}: {secs:.3f}s")
    total_fit_time = sum(s for _, s in timing)
    n_fits_per_case = cfg.max_depth + 2  # nested + control + max_depth fixed-depth baselines
    if timing:
        per_epoch = total_fit_time / (len(timing) * cfg.n_epochs)
        one_unit_at_800 = per_epoch * N_EPOCHS * n_fits_per_case
        print(f"total fit wall-time this run: {total_fit_time:.2f}s over {len(timing)} fits at n_epochs={cfg.n_epochs}")
        print(f"extrapolated: one (T1, seed) unit ({n_fits_per_case} fits) at {N_EPOCHS} epochs ~= {one_unit_at_800:.1f}s ({one_unit_at_800 / 60.0:.1f} min)")
    print(f"total wall-time: {wall_total:.1f}s")
    print(f"\nOUTCOME: {outcome}")


if __name__ == "__main__":
    main()
