"""Wave-A recompute of WSEL-4's reproduction gaps in the ratified per-point-LL bar.

`docs/plans/capacity_programme/width.md` WSEL-4, sign-off item 4 (2026-07-22, user): the landed
2%-relative-NLL reproduction bar (`reproduction.json`'s `relative_error`/`within_bar`) was ruled
arbitrary and replaced with an absolute, consumer-anchored standard --

  (a) UNITS: per-cell agreement as an absolute PER-POINT log-likelihood difference.
  (b) TOLERANCE: 0.1x the per-point LL difference that a 10% change in that cell's own held-out
      error (MSE) would produce, under a fixed-sigma Gaussian idealization.

This script is pure arithmetic over `reproduction.json` (already on disk, landed 2026-07-22) --
NO retraining, NO model runs. It joins that file's `cells` (control_nll/reference_nll, the
control-arm-vs-`W_CONVERGED` reproduction check) with its `ported_vs_control`
(control_held_out_mse, from the SAME trained control-arm net) and recomputes each of the 36
(toy, seed, width) gaps in the new units. Full derivation and the assumptions it rests on are
written into `bar_recheck.json` itself (`formula`, `assumptions`) so a reader never needs this
docstring.

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
_REPRODUCTION_PATH = _HERE / "reproduction.json"
_OUTPUT_PATH = _HERE / "bar_recheck.json"

_TEN_PCT_CHANGE = 0.10  # the consumer-anchor's own "10% change in that cell's error" fraction
_BAR_FRACTION = 0.10  # ratified tolerance: bar = 0.1x the 10%-equivalent LL difference

_FORMULA = (
    "achieved_per_point_ll_diff = |control_nll - reference_nll|, both already per-point-mean "
    "Gaussian NLL over the same N_TEST=500 held-out points computed by the identical "
    "gaussian_log_likelihood()/.mean() construction (converged_width_experiment.py:120, "
    "width_wsel4.py:220-221 via sinc_width_experiment.py's _score_all_widths) -- directly "
    "comparable with no further per-point normalization. "
    "sigma_sq_proxy = control_held_out_mse, this SAME cell's own control-arm held-out MSE from "
    "the SAME trained net that produced control_nll (width_wsel4.py:220-223), used as the "
    "fixed-sigma idealization's noise-floor proxy (docs/plans/capacity_programme/width.md:878-880 "
    "states the formula for 'a Gaussian with fixed per-point sigma'; no raw per-point variance or "
    "historical MSE anchor is recorded on disk to do otherwise -- see `assumptions`). "
    "ten_pct_equivalent_ll_diff = (0.10 * sigma_sq_proxy) / (2 * sigma_sq_proxy). "
    "bar = 0.10 * ten_pct_equivalent_ll_diff. "
    "ratio = achieved_per_point_ll_diff / bar. "
    "pass = achieved_per_point_ll_diff <= bar."
)

_ASSUMPTIONS = [
    "control_nll and reference_nll are both per-point MEAN Gaussian NLL over the same N_TEST=500 "
    "held-out points (converged_width_experiment.py:47-48 N_TEST=500, used unmodified at :188 for "
    "the landed full run; width_wsel4.py's --n-test defaults to cwe.N_TEST). Same scoring function "
    "on both sides: converged_width_experiment.py:120 "
    "`per_k_nll = {k: float(-fixed_width_ll[k].mean()) ...}` where fixed_width_ll comes from "
    "sinc_width_experiment.py:154-170 `_score_all_widths`; width_wsel4.py:220-221 "
    "`nll_te = float(-ll_te.mean())` reuses the identical `sw._score_all_widths`. No rescaling is "
    "needed to compare them as per-point quantities.",
    "The per-width net predicts a LEARNED, per-point-varying (heteroscedastic) log-variance -- "
    "nested_width_net.py:126-129 `gaussian_log_likelihood(mean, log_var, y) = -0.5*(log(2*pi) + "
    "log_var + (y-mean)**2/exp(log_var))`. The historical reference "
    "(W_CONVERGED/w_converged_summary.json) records no per-point variances and no raw MSE -- "
    "verified: its `per_case` entries carry only "
    "{seed, convergence, n_widths_trustworthy, all_widths_trustworthy, hard_curve, easy_curve, "
    "construction, recovery, deploy, marginal_p, per_k_nll}, and width_wsel4.py:27-35 documents the "
    "same finding ('W_CONVERGED's summary JSON never recorded a raw MSE'). An exact "
    "per-point-sigma-weighted ΔLL/ΔMSE relationship therefore cannot be reconstructed from disk "
    "without re-scoring the trained nets, which this task's non-goals prohibit (no retraining, no "
    "model runs).",
    "Given that, sigma_sq_proxy is taken as each cell's OWN control-arm held-out MSE "
    "(`control_held_out_mse` in reproduction.json's `ported_vs_control`, produced by the SAME "
    "trained net as that cell's `control_nll` -- width_wsel4.py:220-223 computes both "
    "`nll_te`/`mse_te` from one `net`/`split[\"x_test\"]`/`split[\"y_test\"]`) -- i.e. the cell's own "
    "achieved noise floor stands in for the fixed sigma under which 'a 10% change in that cell's "
    "error' is evaluated. This is an assumption, not a value read from the source.",
    "Algebraic consequence of that choice: ten_pct_equivalent_ll_diff = (0.10*MSE)/(2*MSE) = 0.05 "
    "nats for every cell (MSE cancels exactly), so bar = 0.10*0.05 = 0.005 nats is a constant "
    "across all 36 cells -- a discovered identity of the chosen proxy, not a hardcoded threshold; "
    "computed per cell below rather than assumed.",
]


def _load(path: Path) -> dict[str, Any]:
    """Loads a JSON file already landed on disk."""
    with path.open() as f:
        return json.load(f)


def _index_mse(rows: list[dict[str, Any]]) -> dict[tuple[str, int, int], dict[str, Any]]:
    """Keys `ported_vs_control` rows by (toy, seed, width) for the join against `cells`."""
    return {(r["toy"], r["seed"], r["width"]): r for r in rows}


def _recompute_cell(cell: dict[str, Any], mse_row: dict[str, Any] | None) -> dict[str, Any]:
    """Recomputes one (toy, seed, width) cell's reproduction gap in the ratified per-point-LL units."""
    achieved = abs(cell["control_nll"] - cell["reference_nll"])
    row: dict[str, Any] = {
        "toy": cell["toy"],
        "seed": cell["seed"],
        "width": cell["width"],
        "control_nll": cell["control_nll"],
        "reference_nll": cell["reference_nll"],
        "achieved_per_point_ll_diff": achieved,
    }

    mse = mse_row["control_held_out_mse"] if mse_row is not None else None
    if mse_row is None:
        reason = "no matching ported_vs_control row for this (toy, seed, width)"
        row.update(control_held_out_mse=None, ten_pct_equivalent_ll_diff=None, bar=None, ratio=None, pass_=None, reason=reason)
    elif mse is None or not math.isfinite(mse) or mse <= 0:
        reason = f"control_held_out_mse={mse!r} is missing/non-positive -- sigma-proxy undefined, conversion ill-defined"
        row.update(control_held_out_mse=mse, ten_pct_equivalent_ll_diff=None, bar=None, ratio=None, pass_=None, reason=reason)
    else:
        ten_pct_equivalent = (_TEN_PCT_CHANGE * mse) / (2.0 * mse)
        bar = _BAR_FRACTION * ten_pct_equivalent
        row.update(
            control_held_out_mse=mse,
            ten_pct_equivalent_ll_diff=ten_pct_equivalent,
            bar=bar,
            ratio=(achieved / bar) if bar > 0 else None,
            pass_=bool(achieved <= bar),
        )
    # JSON output key is "pass" (a reserved word in Python, hence "pass_" internally).
    row["pass"] = row.pop("pass_")
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
    }


def build_bar_recheck() -> dict[str, Any]:
    """Builds the full `bar_recheck.json` payload from `reproduction.json`."""
    reproduction = _load(_REPRODUCTION_PATH)
    mse_by_key = _index_mse(reproduction["ported_vs_control"])

    cells = [_recompute_cell(cell, mse_by_key.get((cell["toy"], cell["seed"], cell["width"]))) for cell in reproduction["cells"]]

    n_ill_defined = sum(1 for c in cells if c["pass"] is None)
    all_pass = n_ill_defined == 0 and all(c["pass"] for c in cells)

    return {
        "all_pass": all_pass,
        "n_cells": len(cells),
        "n_ill_defined": n_ill_defined,
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
        f"worst_ratio={worst['ratio']} at (toy={worst['toy']}, seed={worst['seed']}, width={worst['width']})"
    )


if __name__ == "__main__":
    main()
