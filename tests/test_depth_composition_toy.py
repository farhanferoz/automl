"""Tests for DSEL-1b's nested (all-rungs) feed-forward training scheme (`depth_composition_toy.py`,
docs/plans/capacity_programme/depth-selection.md task DSEL-1b).

The substrate here is deliberately small (short word length, narrow width, short ladder) -- this
file exists to prove the NESTING MECHANISM works (every depth in the ladder gets a usable gradient
and reads non-degenerate on held-out data), not to certify a depth-separation result. That claim,
at the full group-word substrate, is DSEL-2's job.

Run DIRECTLY, e.g. ``python3 tests/test_depth_composition_toy.py`` (repo-wide pytest collection is
broken by an omegaconf conflict; this file also has a ``__main__`` runner, matching
``tests/test_capacity_ladder_toys.py``/``tests/test_variational_em.py``).
"""

import json
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "automl_package", "examples"))
import depth_composition_toy as dct  # noqa: E402

torch.set_num_threads(dct.TORCH_THREADS)  # repo convention on this shared, many-core sandbox (see module comment)

# Small, fast substrate -- enough to prove the nesting mechanism, not a certified result.
_GROUP = dct.Group.A5
_SEQ_LEN = 5
_SEED = 0
_LADDER = (1, 2, 3, 4)
_WIDTH = 16
_MAX_EPOCHS = 3000
_CHECK_EVERY = 50
_PATIENCE = 4
_MIN_DELTA = 1e-3
_NONDEGENERATE_MARGIN = 2.0  # every depth's held-out accuracy must clear this multiple of chance


def _train_shared(ladder: tuple[int, ...] = _LADDER) -> tuple[float, dict[int, float], dict[int, float]]:
    """Trains `NestedFeedForwardClf` (SHARED readout) under DSEL-1b's scheme (b); returns (chance, train_acc, val_acc)."""
    data = dct.make_word_data(_GROUP, _SEQ_LEN, _SEED)
    in_dim = data["seq_len"] * data["n_gen"]
    torch.manual_seed(_SEED)
    net = dct.NestedFeedForwardClf(max(ladder), _WIDTH, in_dim, data["n_classes"], readout=dct.ReadoutMode.SHARED)
    _result, train_acc, val_acc = dct.train_nested_clf(
        net, data, ladder, max_epochs=_MAX_EPOCHS, check_every=_CHECK_EVERY, patience=_PATIENCE, min_delta=_MIN_DELTA,
    )
    return 1.0 / data["n_classes"], train_acc, val_acc


def test_every_depth_nondegenerate_on_held_out_data():
    """The property `train_clf` (the failed DSEL-1 arm) never had: EVERY depth in the ladder, not
    just the deepest, reads meaningfully above chance on HELD-OUT data once every depth carries a
    loss term (DSEL-1b scheme (b)). This is the test the DSEL-1b verify contract requires; the
    "truncate the loss to the final depth only, show this test fail, then restore" cycle is run by
    hand against this exact test (not shipped as code -- it is a one-time demonstration, per the
    task brief), landing a checksum-matched file either way.
    """
    chance, _train_acc, val_acc = _train_shared()
    for d in _LADDER:
        assert val_acc[d] > _NONDEGENERATE_MARGIN * chance, f"depth {d} val_acc={val_acc[d]:.4f} not above {_NONDEGENERATE_MARGIN}x chance ({chance:.4f})"


def test_forward_all_depths_matches_truncated_forward():
    """`forward_all_depths`'s depth-`d` entry must equal `forward(x, depth=d)` called directly --
    the prefix-shared-compute path must not change the read at any depth, for either readout arm.
    """
    data = dct.make_word_data(_GROUP, _SEQ_LEN, _SEED)
    in_dim = data["seq_len"] * data["n_gen"]
    x = torch.as_tensor(data["x_val"][:8], dtype=torch.float32)
    for readout in (dct.ReadoutMode.SHARED, dct.ReadoutMode.PER_DEPTH):
        torch.manual_seed(_SEED)
        net = dct.NestedFeedForwardClf(max(_LADDER), _WIDTH, in_dim, data["n_classes"], readout=readout)
        net.eval()
        with torch.no_grad():
            all_outs = net.forward_all_depths(x)
            for d in range(1, max(_LADDER) + 1):
                direct = net.forward(x, depth=d)
                assert torch.allclose(all_outs[d - 1], direct, atol=1e-6), f"readout={readout.value} depth={d} mismatch"


def test_per_depth_readout_gives_each_depth_its_own_head():
    net = dct.NestedFeedForwardClf(4, 8, 16, 5, readout=dct.ReadoutMode.PER_DEPTH)
    assert net.output_layer is None
    assert set(net.heads.keys()) == {"1", "2", "3", "4"}


def test_shared_readout_reuses_one_head_at_every_depth():
    net = dct.NestedFeedForwardClf(4, 8, 16, 5, readout=dct.ReadoutMode.SHARED)
    assert net.heads is None
    assert isinstance(net.output_layer, torch.nn.Linear)


def test_depth_must_be_at_least_one():
    try:
        dct.NestedFeedForwardClf(0, 8, 16, 5, readout=dct.ReadoutMode.SHARED)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for max_depth=0")


def test_write_frozen_ladder_round_trips():
    with tempfile.TemporaryDirectory() as d:
        path = dct.write_frozen_ladder(out_dir=d, ladder=(1, 2, 3))
        assert os.path.basename(path) == "frozen.json"
        with open(path) as f:
            payload = json.load(f)
        assert payload["depth_ladder"] == [1, 2, 3]


if __name__ == "__main__":
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print(f"PASS {_name}")
    print("all tests passed")
