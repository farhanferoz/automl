"""WSEL-23 candidate 1 -- pinning, the gradient-attribution diagnostic, and generalization validation.

Spec (decision-complete, adversarially read and GO'd 2026-07-23):
`docs/plans/capacity_programme/shared/wsel23-candidate1-derived-weighting.md`. This script owns
everything that document's §6 driver contract assigns to "a NEW, SEPARATE small script" -- the
canonical training arms themselves (multi-head/single-head, `--loss weighted`, `--w-max 12/6`) run
through `kdropout_converged_width_experiment.py` (extended, not duplicated, per §3.9's "ONE home"
discipline); this script never trains the candidate's own reference cells.

Three things this script does, matching the spec's three lettered items:
  (a) **`--pin`** -- §3.3: fits the ONE free scalar `A` (exponent `p=2` fixed as a theory anchor) on
      the decay-regime widths `{4..7}` via closed-form OLS, from the 36 already-landed WSEL8
      dedicated-net cells (zero new compute), and freezes `(A, p, sigma2)` into `a2_law.json` --
      `kdropout_converged_width_experiment.py`'s `--loss weighted` reads this file to train.
  (b) **`--validate`** -- §3.5: the zero-cost held-out-widths (`{8..12}`, factor-of-3 tolerance,
      >=4/5 required) and out-of-regime (`{1,2,3}`, under-prediction expected/acceptable) profile
      response, from the SAME 36 cells, no new training.
  (c) **`--diagnostic --arch ARCH --seed SEED`** -- §2: ONE (architecture, seed) gradient-attribution
      cell -- init (genuinely zero-training) and mid-training (a bounded PARTIAL run, stopped at the
      FIRST epoch any one width's `ConvergenceTracker` converges, per §2.3's root-corrected finding
      that no existing checkpoint sources a mid-training state of the canonical unweighted joint
      nets) gradient L2-norm share on the architecture's SHARED parameters (§2.1's table). The
      mid-training state dict is SAVED (`.pt`, already git-ignored repo-wide) so the diagnostic is
      re-runnable without retraining (§1's root requirement). One JSON per cell, landed to disk the
      moment it is produced -- the standing clause every dispatch in this programme carries.
  **`--summarize`** aggregates every already-landed per-cell JSON (diagnostic cells here, plus the
  training-arm summary JSONs `kdropout_converged_width_experiment.py --loss weighted` writes) into
  the §2.4 confirmed/mixed/refuted verdict, the §5.1 per-width matched-width closure bar, and the
  §3.5 required `w_max'=6` generalization probe (weighted vs its unweighted control) -- WITHOUT
  retraining anything. `--selftest` proves the three wiring guarantees the driver contract names.

**Tag discipline the ROOT must follow when running the real grid (a landmine `_summary_filename`
does NOT resolve on its own):** `kdropout_converged_width_experiment.py`'s summary filename is a
function of `(arch, loss, toy, n_train, sigma, tag)` -- NOT `w_max`. A `--w-max 12` and a `--w-max 6`
run at the same `(arch=shared_trunk, loss=weighted)` would therefore silently CLOBBER each other's
file unless given DIFFERENT `--tag` values. This module's defaults (`DEFAULT_W12_TAG`/
`DEFAULT_W6_TAG` below) exist for exactly this reason -- pass them through to every real training
invocation, do not reuse one tag across both scales.

**Never trains the candidate's own reference cells (non-goal, standing clause):** `--diagnostic`'s
partial runs are the diagnostic's OWN evidence, discarded after producing a snapshot (§2.3); this
script never runs the 15 unconditional training cells the spec charges to the root.

Usage:
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python -m automl_package.examples.width_wsel23_candidate1_pin --selftest
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python -m automl_package.examples.width_wsel23_candidate1_pin --pin
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python -m automl_package.examples.width_wsel23_candidate1_pin --validate
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python -m automl_package.examples.width_wsel23_candidate1_pin --diagnostic --arch shared_trunk --seed 0
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python -m automl_package.examples.width_wsel23_candidate1_pin --summarize
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import tempfile

import numpy as np
import torch

_EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _EXAMPLES_DIR)
sys.path.insert(0, os.path.dirname(os.path.dirname(_EXAMPLES_DIR)))  # repo root

import converged_width_experiment as cwe  # noqa: E402
import convergence as cvg  # noqa: E402
import kdropout_converged_width_experiment as kce  # noqa: E402
import nested_width_net as nwn  # noqa: E402
import sinc_width_experiment as sw  # noqa: E402  (only for `_jsonable`, this module's JSON convention)

from automl_package.utils.run_provenance import run_provenance  # noqa: E402

RESULTS_DIR = os.path.join(_EXAMPLES_DIR, "capacity_ladder_results", "WSEL23")
WSEL8_DIR = os.path.join(_EXAMPLES_DIR, "capacity_ladder_results", "WSEL8")
A2_LAW_FILENAME = "a2_law.json"  # bare filename, per §3.4 -- written under RESULTS_DIR.

SEEDS = kce.SEEDS  # (0, 1, 2), §3.8's fixed set.
DIAGNOSTIC_ARCHES = (kce.Arch.SHARED_TRUNK, kce.Arch.NESTED)

# §3.3 fit/held-out/out-of-regime domains, root-amended (fit and held-out roles SWAPPED from the draft).
FIT_DOMAIN = (4, 5, 6, 7)  # the verified smooth-decay regime -- fit A here.
HELD_OUT_DOMAIN = (8, 9, 10, 11, 12)  # the convergence FLOOR -- held out, load-bearing empirical check (§3.5).
OUT_OF_REGIME_DOMAIN = (1, 2, 3)  # representational floor, not smooth decay -- miss expected/acceptable, never a falsifier.
HELD_OUT_FACTOR_TOLERANCE = 3.0  # declared factor-of-3 tolerance (§3.5), not a 2xSE rule -- reasoned there.
HELD_OUT_MIN_PASS = 4  # of 5 held-out widths, mirroring G-WIDTH's own ">=3/4 corners" bar shape.

# §2.2/§2.4: the narrow widths whose gradient share is under test, and the checkable uniform-share
# reference point at the diagnostic's fixed w_max=kce.W_MAX=12 (the diagnostic is never sliced to a
# smaller w_max -- unlike the weighted TRAINING arms, which do run at w_max=6 too, §3.5).
GRAD_SHARE_NARROW_WIDTHS = (1, 2, 3)
UNIFORM_BASELINE_SHARE = len(GRAD_SHARE_NARROW_WIDTHS) / kce.W_MAX  # 3/12 = 0.25

# §5.1 root-amended affected band (widths 4..11; widths 1-3 already at parity, width 12 excluded --
# smallest premium): PER-WIDTH closure, never a pooled aggregate.
AFFECTED_BAND = tuple(range(4, 12))
# §3.5's REQUIRED w_max'=6 probe has no width band of its own in the spec; this is this script's own
# instantiation, mirroring §5.1's "largest premium" reasoning bounded by w_max=6's own widest head.
WMAX6_BAND = (4, 5, 6)

# Tag discipline (see module docstring) -- DIFFERENT tags for the two ladder scales, since
# `kce._summary_filename` does not encode `w_max`.
DEFAULT_W12_TAG = "wsel23c1"
DEFAULT_W6_TAG = "wsel23c1-wmax6"

# §2.1's table, verified at the constructor line (`architectures.py:41-143,214-356`): the exact
# dotted `named_parameters()` names each architecture SHARES across every width. An explicit
# include-set, not a substring-exclude: `SharedTrunkPerWidthHeadNet`'s per-width `mean_heads.*` carry
# no "logvar" substring at all, so excluding `LOGVAR_HEAD_PATH_SUBSTRING` would (wrongly) keep them --
# the two architectures share genuinely different sets, which only an explicit per-architecture list
# captures correctly.
_SHARED_PARAM_NAMES: dict[kce.Arch, frozenset[str]] = {
    kce.Arch.SHARED_TRUNK: frozenset({"trunk.weight", "trunk.bias"}),
    kce.Arch.NESTED: frozenset({"trunk.weight", "trunk.bias", "mean_head.weight", "mean_head.bias"}),
}


# ---------------------------------------------------------------------------
# §3.3 -- pinning, from the 36 already-landed WSEL8 dedicated-net cells.
# ---------------------------------------------------------------------------


def _wsel8_cell_path(wsel8_dir: str, seed: int, width: int) -> str:
    """Path of one dedicated-net cell (`--arm w_sweep`), already on disk (verified present, §0)."""
    return os.path.join(wsel8_dir, f"hetero_{seed}_{width}_w_sweep.json")


def _wsel8_held_out_mse(wsel8_dir: str, seed: int, width: int) -> float:
    """Reads one dedicated-net cell's `held_out_mse` -- the leaf both pinning and §5.1 closure read."""
    with open(_wsel8_cell_path(wsel8_dir, seed, width)) as f:
        return float(json.load(f)["held_out_mse"])


def load_wsel8_mean_mse(wsel8_dir: str = WSEL8_DIR, seeds: tuple[int, ...] = SEEDS, w_max: int = kce.W_MAX) -> dict[int, float]:
    """§3.3 step 1: `mean_mse(w)` over `seeds` for every width `1..w_max`, from the 36 landed cells."""
    return {w: sum(_wsel8_held_out_mse(wsel8_dir, s, w) for s in seeds) / len(seeds) for w in range(1, w_max + 1)}


def pin_a2_law(wsel8_dir: str = WSEL8_DIR) -> dict:
    """Fits `A` (OLS, `p=2` fixed) on `FIT_DOMAIN`, freezes `(A, p, sigma2)` (§3.3).

    Also freezes every leaf a downstream `--validate`/`--summarize` call needs
    (`mean_mse_by_width`/`raw_a2_by_width`, all 12 widths) -- never refit per run
    (§3.4's "instantiated once and FROZEN").
    """
    mean_mse = load_wsel8_mean_mse(wsel8_dir)
    sigma2 = nwn.HETERO_NOISE_SIGMA**2  # read from HETERO_NOISE_SIGMA, never re-estimated (§3.3 step 4).
    raw_a2 = {w: max(mean_mse[w] - sigma2, 0.0) for w in mean_mse}  # step 2: floor at 0, a seed-noise safety clamp.
    numerator = sum(raw_a2[w] / w**2 for w in FIT_DOMAIN)
    denominator = sum(1.0 / w**4 for w in FIT_DOMAIN)
    a_star = numerator / denominator  # step 3's closed-form OLS with p fixed at 2.
    return {
        "A": a_star,
        "p": 2.0,
        "sigma2": sigma2,
        "fit_domain": list(FIT_DOMAIN),
        "held_out_domain": list(HELD_OUT_DOMAIN),
        "out_of_regime_domain": list(OUT_OF_REGIME_DOMAIN),
        "mean_mse_by_width": mean_mse,
        "raw_a2_by_width": raw_a2,
        "wsel8_dir": wsel8_dir,
        "provenance": run_provenance(),
    }


def _raw_a2(a2_law: dict, w: int) -> float:
    """Robust lookup into `a2_law["raw_a2_by_width"]`, whose keys are `int` in memory / `str` post-JSON."""
    tbl = a2_law["raw_a2_by_width"]
    return float(tbl[str(w)]) if str(w) in tbl else float(tbl[w])


# ---------------------------------------------------------------------------
# §3.5 -- generalization validation (zero new compute; reads only `a2_law.json`'s own frozen leaves).
# ---------------------------------------------------------------------------


def validate_a2_law(a2_law: dict) -> dict:
    """§3.5: shape sanity, the held-out-widths factor-of-3 check, and the out-of-regime report.

    The held-out check is load-bearing; the out-of-regime report is an expected miss, never a
    falsifier. Pure function of `a2_law` -- no new WSEL8 reads.
    """
    shape_sane = all(kce.a2_of_w(a2_law, w) > kce.a2_of_w(a2_law, w + 1) for w in range(1, kce.W_MAX))

    held_out_rows = []
    for w in HELD_OUT_DOMAIN:
        pred = kce.a2_of_w(a2_law, w)
        obs = _raw_a2(a2_law, w)
        ratio = obs / pred if pred > 0 else float("inf")
        within = (1.0 / HELD_OUT_FACTOR_TOLERANCE) <= ratio <= HELD_OUT_FACTOR_TOLERANCE
        held_out_rows.append({"w": w, "pred_a2": pred, "obs_a2": obs, "ratio_obs_over_pred": ratio, "within_factor_3": within})
    n_pass = sum(row["within_factor_3"] for row in held_out_rows)
    held_out_pass = n_pass >= HELD_OUT_MIN_PASS

    out_of_regime_rows = []
    for w in OUT_OF_REGIME_DOMAIN:
        pred = kce.a2_of_w(a2_law, w)
        obs = _raw_a2(a2_law, w)
        out_of_regime_rows.append({"w": w, "pred_a2": pred, "obs_a2": obs, "under_predicts": obs > pred})

    return {
        "shape_sane_monotone_decreasing": shape_sane,
        "held_out": {"rows": held_out_rows, "n_pass": n_pass, "n_total": len(HELD_OUT_DOMAIN), "min_pass_required": HELD_OUT_MIN_PASS, "pass": held_out_pass},
        "out_of_regime": {
            "rows": out_of_regime_rows,
            "note": "under-prediction here is EXPECTED and ACCEPTABLE (spec §3.3/§3.5) -- the conservative, non-disruptive direction, never a falsifier",
        },
        "falsified": (not held_out_pass) or (not shape_sane),
    }


# ---------------------------------------------------------------------------
# §2 -- the gradient-attribution diagnostic.
# ---------------------------------------------------------------------------


def _canonical_data(seed: int, n_train: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Phase-1 train/val split for the canonical tier-1 `hetero` cell.

    Mirrors `kce.run_case`'s own construction exactly (`kdropout_converged_width_experiment.py` lines
    ~511-529), train/val only: the diagnostic never scores held-out TEST, so phase-2 and the test
    split are skipped entirely.
    """
    x_tr, y_tr, _region = nwn.make_hetero(n_train, seed, sigma=nwn.HETERO_NOISE_SIGMA)
    p1_idx = np.arange(0, n_train, 2)
    x_p1, y_p1 = x_tr[p1_idx], y_tr[p1_idx]
    val_mask = (np.arange(len(x_p1)) % cwe.VAL_EVERY) == 0
    norm = cwe._standardize_fit(x_p1[~val_mask], y_p1[~val_mask])
    x_tr_t, y_tr_t = cwe._to_std_tensors(x_p1[~val_mask], y_p1[~val_mask], norm)
    x_val_t, y_val_t = cwe._to_std_tensors(x_p1[val_mask], y_p1[val_mask], norm)
    return x_tr_t, y_tr_t, x_val_t, y_val_t


def _grad_share(net: torch.nn.Module, arch: kce.Arch, x_tr: torch.Tensor, y_tr: torch.Tensor, w_max: int) -> dict[int, float]:
    """§2.2: per-width L2-norm share of gradient magnitude on `arch`'s SHARED parameters.

    `w_max` ISOLATED forward+backward passes via `kce._width_loss` (which calls `net.forward_width`,
    NOT `sampled_widths_forward`'s shared-autograd-graph trick), `zero_grad()` between each -- the
    ratified reading: "this measurement wants each width's contribution ISOLATED, not summed." Always
    the CURRENT unweighted MSE objective (`kce.LossType.MSE`), regardless of what the candidate's own
    training arms use -- this measurement is about the PROBLEM, not the candidate fix (§2.3).
    """
    wanted = _SHARED_PARAM_NAMES[arch]
    named = dict(net.named_parameters())
    shared_names = [name for name in named if name in wanted]
    if len(shared_names) != len(wanted):
        raise AssertionError(f"expected shared params {sorted(wanted)} on {type(net).__name__}, found {shared_names}")

    norms: dict[int, float] = {}
    for k in range(1, w_max + 1):
        net.zero_grad(set_to_none=True)
        loss = kce._width_loss(kce.LossType.MSE, net, k, x_tr, y_tr)
        loss.backward()
        g = torch.cat([named[name].grad.detach().flatten() for name in shared_names])
        norms[k] = float(g.norm(p=2).item())
    net.zero_grad(set_to_none=True)

    total = sum(norms.values())
    return {k: (v / total if total > 0 else float("nan")) for k, v in norms.items()}


def diagnostic_cell(
    arch: kce.Arch,
    seed: int,
    *,
    w_max: int = kce.W_MAX,
    n_train: int = kce.N_TRAIN,
    max_epochs: int = kce.DEFAULT_MAX_EPOCHS,
    check_every: int = cvg.DEFAULT_CHECK_EVERY,
    patience: int = cvg.DEFAULT_PATIENCE,
    min_delta: float = cvg.DEFAULT_MIN_DELTA,
    snapshot_dir: str | None = None,
) -> dict:
    """§2: one (arch, seed) diagnostic cell -- init (zero-training) + mid-training gradient share.

    Mid-training: a bounded PARTIAL run under `WidthSchedule.ALL` (every width, every step,
    deterministic -- §1's binding `--schedule all`), the SAME per-width `ConvergenceTracker` machinery
    `_train_kdropout_to_convergence` uses, but stopped at the FIRST epoch any ONE width's tracker
    converges (not ALL, per §2.3's "first genuinely mid state") -- a genuinely different stop rule
    from the canonical driver's, so this loop is NOT a call to `_train_kdropout_to_convergence`
    (which cannot express it), only a reuse of its tracker/step primitives. The state dict at that
    epoch is SAVED (`snapshot_dir`, `.pt`, already git-ignored repo-wide) so the diagnostic never
    needs to retrain to re-measure `mid_share` (§1's root requirement); the run is then DISCARDED
    otherwise -- it is not the candidate's own reference number (Part 3 trains its own arms).
    """
    x_tr_t, y_tr_t, x_val_t, y_val_t = _canonical_data(seed, n_train)

    torch.manual_seed(seed)
    net: nwn.SharedTrunkPerWidthHeadNet | nwn.NestedWidthNet
    net = nwn.SharedTrunkPerWidthHeadNet(w_max=w_max) if arch is kce.Arch.SHARED_TRUNK else nwn.NestedWidthNet(w_max=w_max)

    init_share = _grad_share(net, arch, x_tr_t, y_tr_t, w_max)

    opt = torch.optim.Adam(net.parameters(), lr=cwe.LR)
    trackers = {k: cvg.ConvergenceTracker(patience=patience, min_delta=min_delta) for k in range(1, w_max + 1)}
    all_widths = list(range(1, w_max + 1))  # WidthSchedule.ALL -- deterministic, no RNG consumed.
    net.train()
    stop_epoch = max_epochs
    any_converged = False
    for epoch in range(1, max_epochs + 1):
        opt.zero_grad()
        total_loss = kce._sampled_widths_total_loss(kce.LossType.MSE, net, all_widths, x_tr_t, y_tr_t)
        total_loss.backward()
        opt.step()
        if epoch % check_every == 0:
            net.eval()
            with torch.no_grad():
                for k in range(1, w_max + 1):
                    trackers[k].update(epoch, float(kce._width_loss(kce.LossType.MSE, net, k, x_val_t, y_val_t).item()))
            net.train()
            if any(t.done for t in trackers.values()):
                stop_epoch = epoch
                any_converged = True
                break
    net.eval()

    mid_share = _grad_share(net, arch, x_tr_t, y_tr_t, w_max)

    snapshot_path = None
    if snapshot_dir is not None:
        os.makedirs(snapshot_dir, exist_ok=True)
        snapshot_path = os.path.join(snapshot_dir, f"diagnostic_mid_{arch.value}_seed{seed}.pt")
        torch.save(net.state_dict(), snapshot_path)

    # Honest edge-case flag: if every width happened to converge at the SAME checkpoint (essentially
    # impossible under ALL with heterogeneous per-width difficulty, but never silently assumed away),
    # this was not a genuine "some done, some not" mid state -- recorded, not swallowed (§2.3).
    all_converged_simultaneously = any_converged and all(t.done for t in trackers.values())

    return {
        "arch": arch.value,
        "seed": seed,
        "w_max": w_max,
        "n_train": n_train,
        "init_share": init_share,
        "mid_share": mid_share,
        "mid_stop_epoch": stop_epoch,
        "mid_any_converged_at_stop": any_converged,
        "mid_all_converged_simultaneously": all_converged_simultaneously,
        "mid_hit_cap": not any_converged,
        "mid_convergence_by_width": {k: t.result(final_epoch=stop_epoch).summary() for k, t in trackers.items()},
        "snapshot_path": snapshot_path,
        "provenance": run_provenance(),
    }


def _load_diagnostic_cells(results_dir: str) -> list[dict]:
    """Every already-landed `diagnostic_{arch}_seed{seed}.json` cell under `results_dir`."""
    cells = []
    for path in sorted(glob.glob(os.path.join(results_dir, "diagnostic_*_seed*.json"))):
        with open(path) as f:
            cells.append(json.load(f))
    return cells


def _share_at(shares: dict, k: int) -> float:
    """Robust lookup into a `{width: share}` dict whose keys may be `int` (in memory) or `str` (post-JSON)."""
    return float(shares[str(k)]) if str(k) in shares else float(shares[k])


def _mean_se(values: list[float]) -> dict[str, float]:
    """Mean and standard error of `values` (SE is `0.0` for fewer than 2 values, not NaN).

    Same convention as `width_wsel19.py`'s `_mean_se` (ddof=1 sample SE of the mean), reused rather
    than re-derived.
    """
    arr = np.asarray(values, dtype=np.float64)
    se = float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return {"mean": float(arr.mean()), "se": se, "n": len(arr)}


def classify_diagnostic(cells: list[dict]) -> dict:
    """§2.4: per-arch, per-checkpoint confirmed/mixed/refuted verdict.

    WSEL-20's noise-aware rule, reused for a different decision rather than re-litigated: `Sigma_{k=1..3}
    share_k` beyond the `UNIFORM_BASELINE_SHARE=0.25` baseline by more than 2xSE of that sum across
    present seeds. **Confirmed** iff BOTH checkpoints clear the bar; **refuted** iff NEITHER does;
    otherwise **mixed** -- and mixed does NOT authorize the single-head companion (root ruling, §2.4).
    """
    by_arch: dict[str, dict] = {}
    for arch in DIAGNOSTIC_ARCHES:
        arch_cells = [c for c in cells if c["arch"] == arch.value]
        if not arch_cells:
            by_arch[arch.value] = {"status": "no_cells_present"}
            continue
        entry: dict = {"n_seeds_present": len(arch_cells), "seeds_present": sorted(c["seed"] for c in arch_cells)}
        beyond_by_checkpoint = {}
        for ckpt in ("init_share", "mid_share"):
            sums = [sum(_share_at(c[ckpt], k) for k in GRAD_SHARE_NARROW_WIDTHS) for c in arch_cells]
            stats = _mean_se(sums)
            beyond = (stats["mean"] - UNIFORM_BASELINE_SHARE) > 2.0 * stats["se"]
            entry[ckpt] = {**stats, "beyond_2se_of_uniform_baseline": beyond}
            beyond_by_checkpoint[ckpt] = beyond
        if len(arch_cells) < len(SEEDS):
            entry["verdict"] = "insufficient_seeds"
        elif beyond_by_checkpoint["init_share"] and beyond_by_checkpoint["mid_share"]:
            entry["verdict"] = "confirmed"
        elif not beyond_by_checkpoint["init_share"] and not beyond_by_checkpoint["mid_share"]:
            entry["verdict"] = "refuted"
        else:
            entry["verdict"] = "mixed"
        by_arch[arch.value] = entry
    return by_arch


# ---------------------------------------------------------------------------
# §5.1 / §3.5 -- reading already-landed TRAINING-arm cells (written by
# `kdropout_converged_width_experiment.py --loss weighted`, never by this script).
# ---------------------------------------------------------------------------


def read_training_arm(results_dir: str, arch: kce.Arch, loss: kce.LossType, tag: str) -> dict | None:
    """The training-arm summary JSON, or `None` if it has not landed yet.

    This is whatever `kce.main()` would have written for `(arch, loss, tag)` at the canonical tier-1
    `(n_train, sigma)`.
    """
    path = os.path.join(results_dir, kce._summary_filename(arch, loss, nwn.Toy.HETERO, kce.N_TRAIN, nwn.HETERO_NOISE_SIGMA, tag))
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def matched_width_closure(training_arm_summary: dict, wsel8_dir: str, affected_band: tuple[int, ...] = AFFECTED_BAND) -> dict:
    """§5.1: PER-WIDTH matched-width-ratio closure (joint held-out MSE / WSEL8 dedicated held-out MSE).

    Root-amended closure rule: EACH band width's own 3-seed mean ratio must independently close (gap
    to `1.0` within 2xSE of that width's own 3 seeds) -- never a pooled 24-cell aggregate, which would
    hide one badly-open width under seven closed ones. The pooled mean+-SE is reported too, but
    descriptively only, never as the gate (§5.1 root ruling).
    """
    per_case = training_arm_summary["per_case"]
    w_max = training_arm_summary["config"]["w_max"]
    ratios_by_width: dict[int, list[float]] = {k: [] for k in range(1, w_max + 1)}
    for case in per_case:
        seed = case["seed"]
        held = case["held_out_mse_by_width"]
        for k in range(1, w_max + 1):
            joint_mse = _share_at(held, k)
            sweep_mse = _wsel8_held_out_mse(wsel8_dir, seed, k)
            ratios_by_width[k].append(joint_mse / sweep_mse)

    band = tuple(w for w in affected_band if w <= w_max)
    rows = []
    for k in band:
        stats = _mean_se(ratios_by_width[k])
        gap = abs(stats["mean"] - 1.0)
        closes = gap <= 2.0 * stats["se"]
        rows.append({"width": k, **stats, "gap_to_1": gap, "closes": closes})
    all_close = bool(rows) and all(row["closes"] for row in rows)
    pooled = _mean_se([r for k in band for r in ratios_by_width[k]])
    return {
        "affected_band": list(band),
        "per_width": rows,
        "all_widths_close": all_close,
        "pooled_mean_se_descriptive_only": pooled,
        "per_width_ratios_by_width": ratios_by_width,
    }


def wmax6_probe_report(results_dir: str, wsel8_dir: str, tag: str) -> dict:
    """§3.5's REQUIRED `w_max'=6` generalization probe: weighted vs its unweighted control.

    Same per-width noise-aware readout as §5.1, reporting whether the mid-width premium SHRINKS at
    every width -- the pre-registered profile response ("moves in the SAME direction... the premium
    shrinks, or the candidate's generalization clause fails and that failure is recorded").
    """
    weighted = read_training_arm(results_dir, kce.Arch.SHARED_TRUNK, kce.LossType.WEIGHTED, tag)
    control = read_training_arm(results_dir, kce.Arch.SHARED_TRUNK, kce.LossType.MSE, tag)
    if weighted is None or control is None:
        return {"status": "not_yet_run", "weighted_present": weighted is not None, "control_present": control is not None}

    weighted_closure = matched_width_closure(weighted, wsel8_dir, affected_band=WMAX6_BAND)
    control_closure = matched_width_closure(control, wsel8_dir, affected_band=WMAX6_BAND)
    per_width = []
    all_shrink = True
    for k in WMAX6_BAND:
        w_mean = next(row["mean"] for row in weighted_closure["per_width"] if row["width"] == k)
        c_mean = next(row["mean"] for row in control_closure["per_width"] if row["width"] == k)
        shrinks = w_mean < c_mean
        all_shrink = all_shrink and shrinks
        per_width.append({"width": k, "weighted_mean_ratio": w_mean, "control_mean_ratio": c_mean, "premium_shrinks": shrinks})
    return {
        "status": "landed",
        "affected_band": list(WMAX6_BAND),
        "per_width": per_width,
        "premium_shrinks_at_every_width": all_shrink,
        "weighted_closure": weighted_closure,
        "control_closure": control_closure,
    }


def summarize(results_dir: str, wsel8_dir: str, w12_tag: str, w6_tag: str) -> dict:
    """Aggregates every already-landed per-cell JSON into the §2.4/§3.5/§5.1 report -- no retraining."""
    diagnostic_cells = _load_diagnostic_cells(results_dir)
    diagnostic_report = {
        "n_cells_present": len(diagnostic_cells),
        "by_arch": classify_diagnostic(diagnostic_cells) if diagnostic_cells else {"status": "no_cells_present"},
    }

    a2_law_path = os.path.join(results_dir, A2_LAW_FILENAME)
    validation = validate_a2_law(kce.load_a2_law(a2_law_path)) if os.path.isfile(a2_law_path) else {"status": "not_pinned"}

    primary = {}
    for arch in DIAGNOSTIC_ARCHES:
        arm = read_training_arm(results_dir, arch, kce.LossType.WEIGHTED, w12_tag)
        primary[arch.value] = {"status": "not_yet_run"} if arm is None else matched_width_closure(arm, wsel8_dir)

    return {
        "diagnostic": diagnostic_report,
        "a2_law_validation": validation,
        "primary_matched_width_closure": primary,
        "wmax6_generalization_probe": wmax6_probe_report(results_dir, wsel8_dir, w6_tag),
        "provenance": run_provenance(),
    }


def _print_summary(report: dict) -> None:
    """Readable stdout digest of `summarize`'s report."""
    diag = report["diagnostic"]
    print(f"[wsel23c1-summarize] diagnostic cells present: {diag['n_cells_present']}")
    for arch_val, entry in diag["by_arch"].items() if isinstance(diag["by_arch"], dict) and "status" not in diag["by_arch"] else []:
        print(f"  arch={arch_val} verdict={entry.get('verdict')} seeds={entry.get('seeds_present')}")

    val = report["a2_law_validation"]
    if "status" in val:
        print(f"[wsel23c1-summarize] a2_law validation: {val['status']}")
    else:
        print(
            f"[wsel23c1-summarize] a2_law validation: shape_sane={val['shape_sane_monotone_decreasing']} "
            f"held_out_pass={val['held_out']['pass']} ({val['held_out']['n_pass']}/{val['held_out']['n_total']}) falsified={val['falsified']}"
        )

    for arch_val, closure in report["primary_matched_width_closure"].items():
        if closure.get("status") == "not_yet_run":
            print(f"[wsel23c1-summarize] primary matched-width closure ({arch_val}): not yet run")
        else:
            print(f"[wsel23c1-summarize] primary matched-width closure ({arch_val}): all_widths_close={closure['all_widths_close']}")

    probe = report["wmax6_generalization_probe"]
    if probe["status"] == "not_yet_run":
        print(f"[wsel23c1-summarize] w_max'=6 probe: not yet run (weighted_present={probe['weighted_present']} control_present={probe['control_present']})")
    else:
        print(f"[wsel23c1-summarize] w_max'=6 probe: premium_shrinks_at_every_width={probe['premium_shrinks_at_every_width']}")


# ---------------------------------------------------------------------------
# Selftest.
# ---------------------------------------------------------------------------


_SELFTEST_HALF_WEIGHT_TOL = 1e-6  # float32 slop for "weighted@0.5 == 0.5x plain MSE"
_SELFTEST_A_ROUNDTRIP_TOL = 1e-12  # float round-trip slop for "A survives a JSON write+read"
_SELFTEST_SMOKE_MAX_EPOCHS = 3000  # tiny synthetic cap for check (3) -- seconds, not minutes


def run_selftest() -> bool:
    """Three wiring checks the driver contract names, each seconds-scale synthetic (no real training grid)."""
    ok = True

    # (1) weighted loss == plain MSE when every weight is 1.0 (byte-identical regression guard, §6);
    #     also proves the weight is NOT a vacuous no-op by checking a non-trivial table actually scales.
    torch.manual_seed(0)
    net = nwn.SharedTrunkPerWidthHeadNet(w_max=4)
    x, y = torch.randn(32, 1), torch.randn(32, 1)
    widths = [1, 2, 3, 4]
    mse_loss = kce._sampled_widths_total_loss(kce.LossType.MSE, net, widths, x, y)
    weighted_ones = kce._sampled_widths_total_loss(kce.LossType.WEIGHTED, net, widths, x, y, weight_table=dict.fromkeys(range(1, 5), 1.0))
    diff_ones = (mse_loss - weighted_ones).abs().item()
    ok_1a = diff_ones == 0.0
    weighted_half = kce._sampled_widths_total_loss(kce.LossType.WEIGHTED, net, widths, x, y, weight_table=dict.fromkeys(range(1, 5), 0.5))
    ok_1b = abs(weighted_half.item() - 0.5 * mse_loss.item()) < _SELFTEST_HALF_WEIGHT_TOL
    ok_1 = ok_1a and ok_1b
    print(f"[wsel23c1-pin selftest] (1) weighted@1.0==plain MSE: diff={diff_ones:.3e}; weighted@0.5==0.5x MSE: {ok_1b}  {'PASS' if ok_1 else 'FAIL'}")
    ok = ok and ok_1

    # (2) a2_law.json schema round-trip (real WSEL8 cells -- a 36-file read, not a training cost).
    law = pin_a2_law(WSEL8_DIR)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, A2_LAW_FILENAME)
        with open(path, "w") as f:
            json.dump(sw._jsonable(law), f)
        law_back = kce.load_a2_law(path)
    required_keys = ("A", "p", "sigma2", "fit_domain", "held_out_domain", "mean_mse_by_width", "raw_a2_by_width")
    ok_2 = all(key in law_back for key in required_keys) and abs(law_back["A"] - law["A"]) < _SELFTEST_A_ROUNDTRIP_TOL and law_back["p"] == law["p"]
    print(f"[wsel23c1-pin selftest] (2) a2_law.json schema round-trip: A={law_back.get('A')!r} p={law_back.get('p')!r}  {'PASS' if ok_2 else 'FAIL'}")
    ok = ok and ok_2

    # (3) early-stop-on-first-convergence stops BEFORE max_epochs on a tiny synthetic smoke config,
    #     and the mid-training snapshot is actually written to disk (§1 root requirement, wired).
    with tempfile.TemporaryDirectory() as tmp:
        cell = diagnostic_cell(
            kce.Arch.SHARED_TRUNK, seed=0, w_max=3, n_train=80, max_epochs=_SELFTEST_SMOKE_MAX_EPOCHS,
            check_every=20, patience=2, min_delta=1e-4, snapshot_dir=tmp,
        )
        snapshot_ok = cell["snapshot_path"] is not None and os.path.isfile(cell["snapshot_path"])
    ok_3 = cell["mid_any_converged_at_stop"] and cell["mid_stop_epoch"] < _SELFTEST_SMOKE_MAX_EPOCHS and snapshot_ok
    print(
        f"[wsel23c1-pin selftest] (3) early-stop-on-first-convergence: stop_epoch={cell['mid_stop_epoch']} (<3000) "
        f"any_converged={cell['mid_any_converged_at_stop']} snapshot_saved={snapshot_ok}  {'PASS' if ok_3 else 'FAIL'}"
    )
    ok = ok and ok_3

    print(f"[wsel23c1-pin selftest] {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> None:
    """Runs whichever of `--pin`/`--validate`/`--diagnostic`/`--summarize`/`--selftest` was asked for."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--selftest", action="store_true", help="Seconds-scale synthetic wiring checks, then exit.")
    parser.add_argument("--pin", action="store_true", help="§3.3: fit A on w=4..7, write a2_law.json.")
    parser.add_argument("--validate", action="store_true", help="§3.5: zero-cost held-out/out-of-regime report from a2_law.json.")
    parser.add_argument("--diagnostic", action="store_true", help="§2: ONE (--arch, --seed) gradient-attribution cell (requires both flags).")
    parser.add_argument("--summarize", action="store_true", help="Aggregate every already-landed per-cell JSON into the §2.4/§3.5/§5.1 report.")
    parser.add_argument("--arch", choices=[a.value for a in DIAGNOSTIC_ARCHES], default=None, help="--diagnostic only: which architecture cell to run.")
    parser.add_argument("--seed", type=int, default=None, help="--diagnostic only: which seed cell to run.")
    parser.add_argument("--results-dir", type=str, default=None, help=f"Override the output directory (default: {RESULTS_DIR}).")
    parser.add_argument("--wsel8-dir", type=str, default=None, help=f"Override the WSEL8 source directory (default: {WSEL8_DIR}).")
    parser.add_argument("--a2-law-path", type=str, default=None, help="Override a2_law.json's path (default: <results-dir>/a2_law.json).")
    parser.add_argument("--w12-tag", type=str, default=DEFAULT_W12_TAG, help=f"--summarize only: the --tag the w_max=12 training arms were run with (default: {DEFAULT_W12_TAG}).")
    parser.add_argument("--w6-tag", type=str, default=DEFAULT_W6_TAG, help=f"--summarize only: the --tag the w_max=6 training arms were run with (default: {DEFAULT_W6_TAG}).")
    parser.add_argument("--max-epochs", type=int, default=kce.DEFAULT_MAX_EPOCHS, help="--diagnostic only: partial-run safety cap.")
    parser.add_argument("--check-every", type=int, default=cvg.DEFAULT_CHECK_EVERY, help="--diagnostic only: epochs between per-width held-out checkpoints.")
    parser.add_argument("--patience", type=int, default=cvg.DEFAULT_PATIENCE, help="--diagnostic only: flat checkpoints before a width converges.")
    parser.add_argument("--min-delta", type=float, default=cvg.DEFAULT_MIN_DELTA, help="--diagnostic only: held-out-loss decrease counted as improvement.")
    parser.add_argument("--n-train", type=int, default=kce.N_TRAIN, help="--diagnostic only: training-set size (default: the canonical tier-1 N_TRAIN).")
    args = parser.parse_args()

    if args.selftest:
        sys.exit(0 if run_selftest() else 1)

    results_dir = args.results_dir if args.results_dir is not None else RESULTS_DIR
    wsel8_dir = args.wsel8_dir if args.wsel8_dir is not None else WSEL8_DIR
    a2_law_path = args.a2_law_path if args.a2_law_path is not None else os.path.join(results_dir, A2_LAW_FILENAME)
    os.makedirs(results_dir, exist_ok=True)

    ran_something = False

    if args.pin:
        ran_something = True
        law = pin_a2_law(wsel8_dir)
        with open(a2_law_path, "w") as f:
            json.dump(sw._jsonable(law), f, indent=2)
        print(f"[wsel23c1-pin] wrote {a2_law_path}  A={law['A']:.6f} p={law['p']:g} sigma2={law['sigma2']:.6f}")

    if args.validate:
        ran_something = True
        report = validate_a2_law(kce.load_a2_law(a2_law_path))
        out_path = os.path.join(results_dir, "validation_report.json")
        with open(out_path, "w") as f:
            json.dump(sw._jsonable(report), f, indent=2)
        print(f"[wsel23c1-validate] wrote {out_path}  held_out_pass={report['held_out']['pass']} ({report['held_out']['n_pass']}/{report['held_out']['n_total']})")

    if args.diagnostic:
        ran_something = True
        if args.arch is None or args.seed is None:
            raise SystemExit("--diagnostic requires --arch and --seed (one cell at a time -- the root backgrounds each cell separately)")
        arch = kce.Arch(args.arch)
        snapshot_dir = os.path.join(results_dir, "_diagnostic_snapshots")  # git-ignored (*.pt, repo-wide)
        cell = diagnostic_cell(
            arch, args.seed, n_train=args.n_train, max_epochs=args.max_epochs, check_every=args.check_every,
            patience=args.patience, min_delta=args.min_delta, snapshot_dir=snapshot_dir,
        )
        out_path = os.path.join(results_dir, f"diagnostic_{arch.value}_seed{args.seed}.json")
        with open(out_path, "w") as f:
            json.dump(sw._jsonable(cell), f, indent=2)
        print(f"[wsel23c1-diagnostic] wrote {out_path}  mid_stop_epoch={cell['mid_stop_epoch']} mid_any_converged={cell['mid_any_converged_at_stop']}")

    if args.summarize:
        ran_something = True
        report = summarize(results_dir, wsel8_dir, args.w12_tag, args.w6_tag)
        out_path = os.path.join(results_dir, "wsel23c1_summary.json")
        with open(out_path, "w") as f:
            json.dump(sw._jsonable(report), f, indent=2)
        print(f"[wsel23c1-summarize] wrote {out_path}")
        _print_summary(report)

    if not ran_something:
        parser.print_help()


if __name__ == "__main__":
    main()
