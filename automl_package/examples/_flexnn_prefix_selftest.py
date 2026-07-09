"""F0 — FlexibleHiddenLayersNN prefix-property + native-scoring selftest (read-only audit).

This is a numerical AUDIT of the EXISTING `FlexibleHiddenLayersNN`, not a new
mechanism. It asserts two facts of record from the 2026-07-09 capacity-ladder
survey (see `docs/plans/capacity_ladder_2026-07-09/EXECUTION_PLAN.md` §1/§3):

  (i) Prefix property: depth d reuses blocks 1..d. The soft layer-selection
      strategies (`GumbelSoftmaxStrategy`, `SoftGatingStrategy`, `SteStrategy`;
      `automl_package/models/selection_strategies/layer_selection_strategies.py`)
      already cache every intermediate block representation inside one forward
      pass and apply the SHARED output layer to each depth in turn — the
      all-depth score table is a free by-product of one forward pass, not
      something that needs `max_hidden_layers` separate forwards. Nothing in
      the strategy's public return value exposes this per-depth table (only
      the probability-weighted aggregate is returned), so this selftest
      extracts it with a `torch.nn.Module.register_forward_hook` on the
      shared `output_layer`: during ONE call to `FlexibleNNModule.forward`,
      that hook fires once per depth, in depth order (see
      `layer_selection_strategies.py:79-90` for the loop that produces this).
      That hook-captured sequence is asserted equal, to 1e-6, to an
      INDEPENDENT truncated forward that runs only blocks 1..d and then the
      shared output layer — i.e. the cached value really is what a from-scratch
      prefix computation would give, for every d.

  (ii) Native NLL scoring: under `UncertaintyMethod.PROBABILISTIC` the shared
       output layer emits `(mean, log_var)` (`flexible_neural_network.py:82-84`),
       so per-depth Gaussian NLL scoring falls out of the SAME forward pass —
       no WS3 shim is needed for the primary ladder arm.

Design decision recorded here (NOT a code change): capacity-ladder arms that
build on `FlexibleHiddenLayersNN` (F2 onward) run with BatchNorm OFF
(`use_batch_norm=False`). Blocks MAY contain BatchNorm
(`flexible_neural_network.py:101-111`), but a shared BatchNorm layer's running
statistics are corrupted by mixed-depth batches during nested-depth training —
the same lesson the slimmable-networks line of work (Yu & Huang 2019,
arXiv:1903.05134) hit with switchable BN. Per-depth BatchNorm statistics are
the REGISTERED FALLBACK if BN-off measurably costs accuracy in F2; that
fallback is not implemented here.

This script is read-only with respect to `automl_package/models/` — it
constructs a small `FlexibleHiddenLayersNN`, calls its existing
`build_model()`, and runs forward passes. No training, no library edits.

Usage:
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python automl_package/examples/_flexnn_prefix_selftest.py --selftest
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

_EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _EXAMPLES_DIR)
sys.path.insert(0, os.path.dirname(os.path.dirname(_EXAMPLES_DIR)))  # repo root, so `import automl_package` works

from automl_package.enums import ActivationFunction, LayerSelectionMethod, TaskType, UncertaintyMethod  # noqa: E402
from automl_package.models.flexible_neural_network import FlexibleHiddenLayersNN  # noqa: E402

TOL = 1e-6


def _make_model(uncertainty_method: UncertaintyMethod = UncertaintyMethod.CONSTANT) -> FlexibleHiddenLayersNN:
    """Builds a small FlexibleHiddenLayersNN, weights initialized only (no training).

    BN off throughout, per the F0 design decision recorded in this file's docstring.
    """
    torch.manual_seed(0)
    model = FlexibleHiddenLayersNN(
        task_type=TaskType.REGRESSION,
        max_hidden_layers=4,
        hidden_size=8,
        activation=ActivationFunction.RELU,
        layer_selection_method=LayerSelectionMethod.SOFT_GATING,  # deterministic (plain softmax, no sampling/noise)
        n_predictor_layers=1,
        use_batch_norm=False,
        uncertainty_method=uncertainty_method,
        output_size=1,
        input_size=3,
        random_seed=0,
    )
    model.build_model()
    model.model.eval()
    return model


def _truncated_forward(module: torch.nn.Module, x: torch.Tensor, depth: int) -> torch.Tensor:
    """Independent truncated forward: runs blocks 1..depth then the shared output layer.

    Deliberately does NOT go through any selection strategy — it is the ground truth
    that assertion (i) checks the strategy's cached per-depth value against.
    """
    current = x
    for i in range(depth):
        current = module.hidden_layers_blocks[i](current)
    return module.output_layer(current)


def assert_prefix_property() -> tuple[bool, float]:
    """Asserts fact (i): cached per-depth output == independent truncated forward, to TOL.

    Extracts the cached per-depth outputs via a forward hook on the shared `output_layer`
    during ONE call to `model.model(x)` — the exact API the next task (F2) should reuse.
    """
    model = _make_model()
    x = torch.randn(37, 3, device=model.device)

    captured: list[torch.Tensor] = []
    handle = model.model.output_layer.register_forward_hook(lambda _mod, _inp, out: captured.append(out.detach().clone()))
    try:
        with torch.no_grad():
            model.model(x)  # one forward pass; SoftGatingStrategy.forward() calls output_layer once per depth internally
    finally:
        handle.remove()

    max_depth = model.max_hidden_layers
    if len(captured) != max_depth:
        print(f"FAIL prefix-property: expected {max_depth} cached per-depth outputs from the forward hook, got {len(captured)}")
        return False, float("nan")

    max_abs_err = 0.0
    all_ok = True
    with torch.no_grad():
        for depth_idx in range(max_depth):
            depth = depth_idx + 1
            cached = captured[depth_idx]
            independent = _truncated_forward(model.model, x, depth)
            err = (cached - independent).abs().max().item()
            max_abs_err = max(max_abs_err, err)
            ok = err < TOL
            all_ok = all_ok and ok
            status = "OK" if ok else "FAIL"
            print(f"  depth={depth}: max_abs_err={err:.3e} [{status}]")

    print(f"{'PASS' if all_ok else 'FAIL'} prefix-property (i): cached per-depth output == independent truncated forward, max_abs_err={max_abs_err:.3e} (tol={TOL:.0e})")
    return all_ok, max_abs_err


def assert_probabilistic_output() -> bool:
    """Asserts fact (ii): PROBABILISTIC uncertainty gives native (mean, log_var) heads."""
    model = _make_model(uncertainty_method=UncertaintyMethod.PROBABILISTIC)
    n = 11
    x = torch.randn(n, 3, device=model.device)

    ok = True
    if model.model.output_layer.out_features != 2:
        print(f"FAIL probabilistic-output (ii): output_layer.out_features={model.model.output_layer.out_features}, expected 2 (mean, log_var)")
        ok = False

    with torch.no_grad():
        final_output, _n_actual, _unused, _n_probs, _log_prob = model.model(x)
    if tuple(final_output.shape) != (n, 2):
        print(f"FAIL probabilistic-output (ii): final_output.shape={tuple(final_output.shape)}, expected {(n, 2)}")
        ok = False

    # End-to-end: predict_uncertainty() reads column 1 as log_var and exponentiates it
    # (flexible_neural_network.py:382-384) — exercise that path, not just the shape.
    x_np = x.cpu().numpy()
    uncertainty = model.predict_uncertainty(x_np, filter_data=False)
    if uncertainty.shape != (n,) or not (uncertainty > 0).all():
        print(f"FAIL probabilistic-output (ii): predict_uncertainty() output shape/sign wrong: shape={uncertainty.shape}, min={uncertainty.min() if uncertainty.size else 'n/a'}")
        ok = False

    if ok:
        print(f"PASS probabilistic-output (ii): native (mean, log_var) heads, final_output.shape={tuple(final_output.shape)}, predict_uncertainty() shape={uncertainty.shape}")
    return ok


def main() -> int:
    """Runs both F0 assertions and prints PASS/FAIL per assertion plus a final summary line."""
    t0 = time.time()
    ok_i, max_err = assert_prefix_property()
    ok_ii = assert_probabilistic_output()
    elapsed = time.time() - t0

    print(f"wall-time: {elapsed:.2f}s")
    if ok_i and ok_ii:
        print("ALL PASS")
        return 0
    print(f"SOME FAILED (prefix-property max_abs_err={max_err:.3e})")
    return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true", help="Run the F0 prefix-property + native-scoring selftest.")
    args = parser.parse_args()
    if args.selftest:
        raise SystemExit(main())
    parser.print_help()
