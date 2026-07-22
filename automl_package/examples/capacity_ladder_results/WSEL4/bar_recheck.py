"""Wave-A recompute of WSEL-4's reproduction gaps in the ratified per-point-LL bar.

`docs/plans/capacity_programme/width.md` WSEL-4, sign-off item 4 (2026-07-22, user): the landed
2%-relative-NLL reproduction bar (`reproduction.json`'s `relative_error`/`within_bar`) was ruled
arbitrary and replaced with an absolute, consumer-anchored standard --

  (a) UNITS: per-cell agreement as an absolute PER-POINT log-likelihood difference.
  (b) TOLERANCE: 0.1x the per-point LL difference that a 10% change in that cell's own held-out
      error (MSE) would produce, under a fixed-sigma Gaussian idealization.

REFINED 2026-07-22 (root follow-up): the first cut of this recompute used each cell's OWN achieved
held-out MSE as the fixed-sigma proxy, which makes the bar artificially strict exactly where the
net underfits (achieved MSE >> true noise variance there). The hetero toy's noise is GENERATIVE
TRUTH, readable from source with no retraining -- `nested_width_net.make_hetero` adds Gaussian
noise, COMMON-MODE across both regions, at a fixed `HETERO_NOISE_SIGMA=0.05` (nested_width_net.py
:93,144,161-164,176) -- so this version uses that true noise variance as sigma, and keeps the old
MSE-based construction alongside under `proxy_*` fields so both are on record.

This script is pure arithmetic (plus a deterministic, seeded NumPy data regeneration -- no model
training, no model runs) over JSONs already on disk. It joins `reproduction.json`'s `cells`
(control_nll/reference_nll, the control-arm-vs-`W_CONVERGED` reproduction check) with its
`ported_vs_control` (control_held_out_mse, from the SAME trained control-arm net), regenerates each
seed's exact TEST inputs via `nested_width_net.make_hetero` to compute the true noise variance those
inputs carry, and recomputes each of the 36 (toy, seed, width) gaps in the new units. Full
derivation and the assumptions it rests on are written into `bar_recheck.json` itself (`formula`,
`assumptions`) so a reader never needs this docstring.

Determinism note: `provenance.timestamp` is derived from `reproduction.json`'s own mtime, not
wall-clock time, so two runs against the same landed input reproduce `bar_recheck.json`
byte-identically (the driver contract's verify step).

Usage:
    ~/dev/.venv/bin/python automl_package/examples/capacity_ladder_results/WSEL4/bar_recheck.py
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
_EXAMPLES_DIR = _HERE.parent.parent
sys.path.insert(0, str(_EXAMPLES_DIR))

import nested_width_net as nwn  # noqa: E402 -- reused verbatim: make_hetero, HETERO_NOISE_SIGMA

_REPRODUCTION_PATH = _HERE / "reproduction.json"
_W_CONVERGED_PATH = _EXAMPLES_DIR / "capacity_ladder_results" / "W_CONVERGED" / "w_converged_summary.json"
_OUTPUT_PATH = _HERE / "bar_recheck.json"

_TEN_PCT_CHANGE = 0.10  # the consumer-anchor's own "10% change in that cell's error" fraction
_BAR_FRACTION = 0.10  # ratified tolerance: bar = 0.1x the 10%-equivalent LL difference

_FORMULA = (
    "achieved_per_point_ll_diff = |control_nll - reference_nll|, both already per-point-mean "
    "Gaussian NLL over the same N_TEST held-out points (n_test read per-cell from that cell's own "
    "landed JSON) computed by the identical gaussian_log_likelihood()/.mean() construction "
    "(converged_width_experiment.py:120, width_wsel4.py:220-221 via sinc_width_experiment.py's "
    "_score_all_widths) -- directly comparable with no further per-point normalization. "
    "PRIMARY construction: sigma_sq_true = the hetero toy's GENERATIVE noise variance "
    "(HETERO_NOISE_SIGMA**2), averaged over this cell's own regenerated TEST inputs "
    "(`nested_width_net.make_hetero(n_test, seed + 500)`, the exact deterministic draw both the "
    "control and reference arms scored) -- read from source, not fitted. "
    "ten_pct_equivalent_ll_diff = (0.10 * control_held_out_mse) / (2 * sigma_sq_true). "
    "bar = 0.10 * ten_pct_equivalent_ll_diff. ratio = achieved_per_point_ll_diff / bar. "
    "pass = achieved_per_point_ll_diff <= bar. "
    "PROXY construction (retained for the record, prefixed proxy_*): identical formula with "
    "sigma_sq_true replaced by control_held_out_mse itself (the cell's own achieved error standing "
    "in for its noise floor) -- this was the first-cut construction; it over-tightens the bar "
    "wherever the achieved MSE exceeds the true noise floor (i.e. wherever the net underfits)."
)

_ASSUMPTIONS = [
    "control_nll and reference_nll are both per-point MEAN Gaussian NLL over the same N_TEST "
    "held-out points -- converged_width_experiment.py:120 "
    "`per_k_nll = {k: float(-fixed_width_ll[k].mean()) ...}` where fixed_width_ll comes from "
    "sinc_width_experiment.py:154-170 `_score_all_widths`; width_wsel4.py:220-221 "
    "`nll_te = float(-ll_te.mean())` reuses the identical `sw._score_all_widths`. No rescaling "
    "needed to compare them as per-point quantities. n_test is read per cell from that cell's own "
    "landed `<toy>_<seed>_<width>_control.json` (`config.n_test`) rather than assumed; verified "
    "uniform at 500 across all 36 landed control cells.",
    "The hetero toy's noise is GENERATIVE TRUTH, not fitted: `make_hetero` adds "
    "`rng.normal(0.0, sigma, n)` with `sigma=HETERO_NOISE_SIGMA=0.05` "
    "(nested_width_net.py:93,144,176), and its own docstring states this is 'COMMON-MODE across "
    "both regions' (nested_width_net.py:161-164) -- i.e. sigma(t)**2 is the SAME constant "
    "(0.05**2 = 0.0025) at every input t, not x-/region-dependent (contrast `make_hetero3`, which "
    "DOES vary noise by region at nested_width_net.py:218 -- out of WSEL-4's hetero-only scope). "
    "Both the control and reference arms score the SAME deterministic test draw, "
    "`make_hetero(n_test, seed + 500)` (width_wsel4.py:155, converged_width_experiment.py:94) -- "
    "this script regenerates that exact draw (a pure seeded NumPy call, no model training/scoring) "
    "and averages the true per-point noise variance over it, rather than shortcutting straight to "
    "the constant, so the on-disk number reflects a real per-cell computation. Because the noise is "
    "input-independent, that average is exactly 0.0025 for every cell regardless of seed/width -- a "
    "verified consequence of the toy's construction, not an assumption.",
    "The per-width net predicts a LEARNED, per-point-varying (heteroscedastic) log-variance -- "
    "nested_width_net.py:126-129 `gaussian_log_likelihood(mean, log_var, y) = -0.5*(log(2*pi) + "
    "log_var + (y-mean)**2/exp(log_var))` -- which is a MODEL of the noise, distinct from the toy's "
    "true generative noise used here. The historical reference "
    "(W_CONVERGED/w_converged_summary.json) records neither the model's per-point variances nor a "
    "raw MSE (verified: its `per_case` entries carry only "
    "{seed, convergence, n_widths_trustworthy, all_widths_trustworthy, hard_curve, easy_curve, "
    "construction, recovery, deploy, marginal_p, per_k_nll}), so an exact per-point MODEL-sigma- "
    "weighted ΔLL/ΔMSE relationship still cannot be reconstructed without re-scoring the trained "
    "nets (prohibited). Using the toy's true GENERATIVE sigma instead sidesteps that gap entirely "
    "for this toy, since it needs no model internals at all.",
    "`reference_seed_untrustworthy` is read directly from "
    "`W_CONVERGED/w_converged_summary.json`'s top-level `untrustworthy_seeds` list (verified content "
    "below, not assumed from the dispatching message) -- per width.md's WSEL-4 carry-forward "
    "caveat, those seeds' historical numbers are not trajectory-certified, so a bar failure on one "
    "of them reflects on the historical reference's own certification, not solely on the port.",
]


def _load(path: Path) -> dict[str, Any]:
    """Loads a JSON file already landed on disk."""
    with path.open() as f:
        return json.load(f)


def _index_mse(rows: list[dict[str, Any]]) -> dict[tuple[str, int, int], dict[str, Any]]:
    """Keys `ported_vs_control` rows by (toy, seed, width) for the join against `cells`."""
    return {(r["toy"], r["seed"], r["width"]): r for r in rows}


def _control_cell_n_test(toy: str, seed: int, width: int) -> int:
    """Reads this cell's own recorded `n_test` from its landed per-cell control JSON (no hardcode)."""
    path = _HERE / f"{toy}_{seed}_{width}_control.json"
    return int(_load(path)["config"]["n_test"])


def _mean_true_sigma_sq(toy: str, seed: int, n_test: int) -> float:
    """Mean GENERATIVE noise variance over the exact TEST inputs this (toy, seed) cell scored.

    Regenerates the deterministic draw both arms scored (`make_hetero(n_test, seed + 500)` --
    width_wsel4.py:155, converged_width_experiment.py:94) and averages the toy's true per-point
    noise variance over it. For `hetero`, that variance is a fixed constant (COMMON-MODE noise,
    nested_width_net.py:161-164), so the average is invariant to seed/width/n_test -- computed
    explicitly here rather than shortcut, so the number on disk is a real per-cell computation.
    """
    if toy != nwn.Toy.HETERO.value:
        raise ValueError(f"no true-noise generator wired for toy={toy!r} -- WSEL-4's scope is hetero only")
    x_te, _y_te, _region_te = nwn.make_hetero(n_test, seed + 500)
    sigma_sq = np.full(x_te.shape, nwn.HETERO_NOISE_SIGMA**2, dtype=np.float64)
    return float(sigma_sq.mean())


def _ten_pct_bar(achieved: float, mse: float | None, sigma_sq: float | None, sigma_label: str) -> dict[str, Any]:
    """Shared 10%-consumer-anchor construction: `ten_pct_equivalent_ll_diff = 0.10*mse / (2*sigma_sq)`, `bar = 0.10*that`."""
    if mse is None or not math.isfinite(mse) or mse <= 0:
        reason = f"control_held_out_mse={mse!r} missing/non-positive -- conversion ill-defined"
        return {"ten_pct_equivalent_ll_diff": None, "bar": None, "ratio": None, "pass": None, "reason": reason}
    if sigma_sq is None or not math.isfinite(sigma_sq) or sigma_sq <= 0:
        reason = f"{sigma_label} sigma_sq={sigma_sq!r} missing/non-positive -- conversion ill-defined"
        return {"ten_pct_equivalent_ll_diff": None, "bar": None, "ratio": None, "pass": None, "reason": reason}
    ten_pct_equivalent = (_TEN_PCT_CHANGE * mse) / (2.0 * sigma_sq)
    bar = _BAR_FRACTION * ten_pct_equivalent
    return {
        "ten_pct_equivalent_ll_diff": ten_pct_equivalent,
        "bar": bar,
        "ratio": (achieved / bar) if bar > 0 else None,
        "pass": bool(achieved <= bar),
        "reason": None,
    }


def _recompute_cell(cell: dict[str, Any], mse_row: dict[str, Any] | None, reference_untrustworthy_seeds: set[int]) -> dict[str, Any]:
    """Recomputes one (toy, seed, width) cell's reproduction gap under both the refined and proxy constructions."""
    achieved = abs(cell["control_nll"] - cell["reference_nll"])
    mse = mse_row["control_held_out_mse"] if mse_row is not None else None

    n_test = _control_cell_n_test(cell["toy"], cell["seed"], cell["width"])
    sigma_sq_true = _mean_true_sigma_sq(cell["toy"], cell["seed"], n_test)

    primary = _ten_pct_bar(achieved, mse, sigma_sq_true, "true-noise")
    proxy = _ten_pct_bar(achieved, mse, mse, "proxy-MSE")  # the first-cut construction: sigma_sq := mse itself

    row: dict[str, Any] = {
        "toy": cell["toy"],
        "seed": cell["seed"],
        "width": cell["width"],
        "control_nll": cell["control_nll"],
        "reference_nll": cell["reference_nll"],
        "achieved_per_point_ll_diff": achieved,
        "control_held_out_mse": mse,
        "reference_seed_untrustworthy": cell["seed"] in reference_untrustworthy_seeds,
        "sigma_sq_true": sigma_sq_true,
        "ten_pct_equivalent_ll_diff": primary["ten_pct_equivalent_ll_diff"],
        "bar": primary["bar"],
        "ratio": primary["ratio"],
        "pass": primary["pass"],
        "proxy_ten_pct_equivalent_ll_diff": proxy["ten_pct_equivalent_ll_diff"],
        "proxy_bar": proxy["bar"],
        "proxy_ratio": proxy["ratio"],
        "proxy_pass": proxy["pass"],
    }
    if primary["reason"] is not None:
        row["reason"] = primary["reason"]
    if proxy["reason"] is not None:
        row["proxy_reason"] = proxy["reason"]
    return row


def _provenance() -> dict[str, Any]:
    """Provenance for `bar_recheck.json` -- deterministic across reruns against the same landed input."""
    git_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_HERE, capture_output=True, check=True, text=True).stdout.strip()
    input_mtime = _REPRODUCTION_PATH.stat().st_mtime
    return {
        "git_commit": git_commit,
        "timestamp": datetime.fromtimestamp(input_mtime, tz=UTC).isoformat(),
        "timestamp_source": "reproduction.json mtime (not wall-clock) -- keeps reruns byte-identical",
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "source_reproduction_json": str(_REPRODUCTION_PATH),
        "source_w_converged_json": str(_W_CONVERGED_PATH),
    }


def build_bar_recheck() -> dict[str, Any]:
    """Builds the full `bar_recheck.json` payload from `reproduction.json` and `w_converged_summary.json`."""
    reproduction = _load(_REPRODUCTION_PATH)
    reference = _load(_W_CONVERGED_PATH)
    reference_untrustworthy_seeds = set(reference["untrustworthy_seeds"])
    mse_by_key = _index_mse(reproduction["ported_vs_control"])

    cells = [
        _recompute_cell(cell, mse_by_key.get((cell["toy"], cell["seed"], cell["width"])), reference_untrustworthy_seeds) for cell in reproduction["cells"]
    ]

    n_ill_defined = sum(1 for c in cells if c["pass"] is None)
    all_pass = n_ill_defined == 0 and all(c["pass"] for c in cells)
    n_fail_on_trustworthy_seed = sum(1 for c in cells if c["pass"] is False and not c["reference_seed_untrustworthy"])

    return {
        "all_pass": all_pass,
        "n_cells": len(cells),
        "n_ill_defined": n_ill_defined,
        "n_fail_on_trustworthy_seed": n_fail_on_trustworthy_seed,
        "reference_untrustworthy_seeds": sorted(reference_untrustworthy_seeds),
        "formula": _FORMULA,
        "assumptions": _ASSUMPTIONS,
        "cells": cells,
        "provenance": _provenance(),
    }


def main() -> None:
    """Recomputes `bar_recheck.json` from the landed `reproduction.json` and writes it to disk."""
    result = build_bar_recheck()
    with _OUTPUT_PATH.open("w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    worst = max(result["cells"], key=lambda c: (c["ratio"] is not None, c["ratio"] or 0.0))
    print(
        f"[bar_recheck] wrote {_OUTPUT_PATH} -- all_pass={result['all_pass']} "
        f"n_cells={result['n_cells']} n_ill_defined={result['n_ill_defined']} "
        f"n_fail_on_trustworthy_seed={result['n_fail_on_trustworthy_seed']} "
        f"worst_ratio={worst['ratio']} at (toy={worst['toy']}, seed={worst['seed']}, width={worst['width']})"
    )


if __name__ == "__main__":
    main()
