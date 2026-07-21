"""WSEL-15 Step 1 -- does the prefix-normalisation trick actually work? (`docs/plans/capacity_programme/
width.md` ~1466-1471.)

Arm B (`PrefixNormMode.RUNNING_TOTALS`) computes every width's RMS normaliser `r_k` from ONE
`cumsum(h**2)` pass over the shared hidden vector, dividing by the ACTIVE count `k` -- never `w_max`.
Arm D (`PrefixNormMode.NAIVE`) computes the IDENTICAL formula the textbook way: loop over `k`, slice
`h[:, :k]`, take its RMS directly. Both live in `automl_package/examples/width_candidates.py`'s
`PrefixNormWidthNet` -- `mode` is the only thing that differs between them, so if the two ever
disagree, the vectorised trick (repair 2 of `shared/width_transformer_port.md` SS5) is wrong. This is
the load-bearing check WSEL-15 stops and reports on if it fails -- nothing else in that task runs
before this passes.

`check_prefix_norm_exact` (imported from `automl_package/examples/width_wsel15.py`, not re-derived
here) does the actual comparison, at initialisation and after 10 training steps, on a fixed seed and a
fixed `(64, 1)` input -- exactly the spec's tolerance (`1e-5`, `torch.allclose`) and toy size. The
driver's `frozen.json`-writing `summarize()` calls the SAME function, so the pytest gate below and the
`prefix_norm_exact` field written to disk can never silently disagree with each other.

**Prove-it-fails** (`test_full_vector_totals_disagrees_with_prefix`): swaps the correct per-k prefix
total for a normaliser computed over the FULL `w_max`-wide vector instead -- the literal bug the exact
spec (`r_k = sqrt(cumsum(h**2)[:k]/k + eps)`, divide by the active count, never by `w_max`) exists to
prevent -- and shows it disagrees with the naive oracle (arm D) at every width `k < w_max` (at `k =
w_max` the two coincide: the full-vector mean IS the width-`w_max` prefix). If this test does NOT fail,
arm D is not a discriminating oracle and Step 1's pass above would be meaningless.

Usage:
    AUTOML_DEVICE=cpu OMP_NUM_THREADS=4 ~/dev/.venv/bin/python -m pytest tests/test_prefix_norm_equivalence.py -q
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "automl_package", "examples"))
import width_candidates as wc
import width_wsel15 as w15

_TOL = 1e-5  # brief's exact bar
_W_MAX = 12
_N = 64


def test_running_totals_matches_naive_at_init_and_after_training() -> None:
    """Step 1's load-bearing check: arm B (running totals) vs arm D (naive), init AND post-training."""
    exact, max_err = w15.check_prefix_norm_exact(w_max=_W_MAX, n=_N, seed=0, n_train_steps=10)
    assert exact, f"arm B (running-totals cumsum) disagrees with arm D (naive per-k slice): max_abs_err={max_err:.3e} (tol={_TOL:.0e})"


def test_full_vector_totals_disagrees_with_prefix() -> None:
    """Prove-it-fails: a normaliser computed over the FULL vector (dividing by `w_max`, not the active
    count `k`) must disagree with the true prefix RMS (arm D) at every width `k < w_max` -- the exact
    bug the spec's `r_k` formula exists to prevent. If it does NOT disagree anywhere, arm D cannot
    actually catch this bug and Step 1's PASS above proves nothing.
    """
    torch.manual_seed(0)
    net = wc.PrefixNormWidthNet(w_max=_W_MAX, mode=wc.PrefixNormMode.NAIVE)
    net.eval()
    x = torch.randn(_N, 1)

    with torch.no_grad():
        h = net.hidden(x)
        full_vector_r = torch.sqrt((h * h).mean(dim=1, keepdim=True) + wc.PREFIX_NORM_EPS)  # BUGGY: /w_max at every k, not /k.

        mismatches = []
        for k in range(1, _W_MAX):  # k = w_max deliberately excluded: full-vector mean IS the w_max prefix there.
            mean_d, _ = net.forward_width(x, k)
            mask = torch.zeros_like(h)
            mask[:, :k] = 1.0
            h_buggy = (h / full_vector_r) * mask
            mean_buggy = net.base.mean_heads[k - 1](h_buggy)
            if not torch.allclose(mean_buggy, mean_d, atol=_TOL, rtol=0):
                mismatches.append(k)

    assert mismatches, (
        "a full-vector (divide-by-w_max) normaliser must disagree with the true prefix RMS (arm D) at "
        "some width k < w_max -- it did not, so this oracle would not catch the /w_max bug the spec "
        "exists to prevent"
    )
