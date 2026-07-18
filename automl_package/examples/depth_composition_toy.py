"""Depth toy via GROUP WORD-PROBLEM composition — the depth analog of the width toy's flat+sine.

**Why this replaces the 1-D regression candidates** (`docs/plans/capacity_programme/depth.md` D1;
`docs/depth_capacity/depth_toy_negative_note.md`; this session's grounding). The four 1-D
smooth-regression candidates all failed because depth-hunger and GD-learnability are MUTUALLY
EXCLUSIVE for smooth 1-D targets — now known to be a proven theorem, not an artifact: Malach &
Shalev-Shwartz 2019 (arXiv:1903.03488) and Malach, Yehudai, Shalev-Shwartz, Shamir 2021
(arXiv:2102.00434) show GD-learnability of a deep net REQUIRES weak approximability by SHALLOW nets,
so a genuinely shallow-unapproximable (depth-hungry) target categorically blocks GD. The only
depth-hungry 1-D targets (Telgarsky sawtooth compositions) are high-frequency and hit the
spectral-bias wall — exactly candidates 1-2.

**The escape**: make the hardness COMBINATORIAL (number of sequential steps), not spectral. The
word problem over a finite group does this. Composing a sequence of elements of a NON-SOLVABLE group
(S5) is the canonical primitive where WIDTH PROVABLY CANNOT SUBSTITUTE FOR DEPTH (Barrington 1989:
width-5 branching programs over S5 capture NC1 exactly — more width buys nothing, only sequential
program length ≈ depth helps). It IS GD-learnable (Liu, Ash, Goel, Krishnamurthy, Zhang 2022,
"Transformers Learn Shortcuts to Automata", arXiv:2210.10749, train it with plain GD), and the
difficulty knob is the SEQUENCE LENGTH n = number of composition steps — a natural per-input graded
DEPTH dial. The incremental structure (each step adds one composition) is the staircase/leap
property (Abbe et al. 2021-2023) that lets SGD climb it despite the depth-hunger.

**Built-in negative control** (the depth analog of the width toy's noisy-easy region): a SOLVABLE
group of the SAME order (abelian Z_|G|) is width-substitutable — Liu et al. show solvable-group
automata get "shortcut" to O(1)-depth shallow nets — so a WIDE-SHALLOW net FITS it. Same #classes,
same sequence structure, only the group's solvability differs.

**Depth-separation SIGNATURE** (bars INVERTED from the width toy: there the control had to FIT to
prove learnability; here the wide-shallow net must STALL to prove the hardness is DEPTH not width):
  - deep-narrow net FITS the length-n S5 product (high held-out accuracy)      [learnable]
  - wide-shallow net STALLS on it (width cannot buy the sequential steps)      [genuinely DEPTH]
  - graded: held-out accuracy rises with depth toward the length-n ceiling
  - control: wide-shallow FITS the solvable-group (Z_|G|) product              [width CAN substitute]

**Open point this module's pilot settles**: whether a plain FEEDFORWARD MLP on the flattened
one-hot word learns S5-composition by GD (the cited learnability results are for TRANSFORMERS), or
whether the RECURRENT-UNROLLED (weight-shared per step) form is needed. `--net {mlp,recurrent}`
runs either; the pilot reports both so the winning architecture is a measured fact, not a guess.

**Metric**: held-out (generalization) accuracy on UNSEEN words of the given length — a net that
learned the composition ALGORITHM generalizes; one that only memorized the train words fails. Chance
is 1/|G|. No regression / NLL here (this is classification over |G| group elements); the
convergence gate (`convergence.py`, full-trajectory rule, MASTER Decision 9) still governs every
read, on val cross-entropy.

Usage:
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python automl_package/examples/depth_composition_toy.py --selftest
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python automl_package/examples/depth_composition_toy.py \
        --pilot --group s5 --seq-len 5 --net mlp --seed 0
"""

from __future__ import annotations

import argparse
import enum
import itertools
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

_EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _EXAMPLES_DIR)
sys.path.insert(0, os.path.dirname(os.path.dirname(_EXAMPLES_DIR)))  # repo root, so `import automl_package` works

import convergence as cvg  # noqa: E402 — full-trajectory convergence gate, shared with the width/depth toys

# ---------------------------------------------------------------------------
# Groups (closed set) + their generator sets. S5 = non-solvable (depth-hungry, Barrington);
# Z120 = abelian/solvable of the SAME order 120 (width-substitutable control).
# ---------------------------------------------------------------------------

S5_N = 5  # symmetric group on 5 points; |S5| = 120 and it is the smallest non-solvable symmetric group
Z_ORDER = 120  # cyclic control group order, matched to |S5| so #classes and word structure agree
Z_GENERATORS = (1, 7, 41, 83)  # 4 fixed generators of Z120 (match S5's 4-generator alphabet; abelian => solvable)

A5_N = 5  # A5 acts on 5 points; |A5| = 60, the alternating (even-permutation) subgroup of S5 -- still
# non-solvable (Barrington 1989 applies unchanged). Alphabet for the depth-SELECTION toy
# (`depth_selection_toy.py`, D8b): four ORDER-2 involutions (double-transpositions), verified below to
# generate all 60 elements. Rationale (2026-07-17): the earlier five-cycle alphabet had NO identity word
# shorter than length 5 -- a CONCEALMENT property that is now dropped (design §6 Option A: detection
# difficulty is not the charter). At the GD-trainable length L=10, that no-short-relation property makes
# every early-commitment stratum unconstructible (a t=6 commitment needs a length-4 identity suffix, of
# which five-cycles have none). Involutions restore short identity relations (g*g = identity at length 2;
# further identity words at lengths 3,4,...), so the L=10 ladder {6,8,10} is constructible while the
# group stays non-solvable and all 60 classes are reachable at every rung. Set found by exhaustive search
# over A5's 15 involutions for a 4-subset whose BFS-closure is all 60 (see selftest `generators_span`).
A5_GENERATORS = ((0, 2, 1, 4, 3), (0, 3, 4, 1, 2), (0, 4, 3, 2, 1), (1, 0, 2, 4, 3))

LR = 1e-2
MAX_EPOCHS = 40000
CHECK_EVERY = 250
PATIENCE = 10
MIN_DELTA = 1e-4  # on val cross-entropy (nats); below any genuine early-training improvement, above full-batch jitter

NARROW_WIDTH = 16  # fixed small width for the depth ladder (classification needs more units than the 1-D toy's 8)
WIDE_WIDTH = 512  # wide-shallow control width (>> NARROW_WIDTH); tests whether width can substitute for depth
REC_STATE_WIDTH = 64  # recurrent net's state width — must encode |G| group elements, so wider than NARROW_WIDTH
PROBE_DEPTHS = (1, 2, 3, 4, 5, 6)  # depth ladder; P1 reads the deepest, "stall" bar reads depth 1
DEFAULT_TRAIN_FRAC = 0.5  # held-out fraction tests generalization (compositional), not memorization
MAX_ENUM = 20000  # enumerate all n_gen**seq_len words below this; sample above it

FIT_ACC = 0.90  # deep-narrow "fits" the length-n product at >= this held-out accuracy
STALL_ACC = 0.60  # wide-shallow "stalls" at <= this held-out accuracy (well below FIT_ACC)

DEFAULT_OUT_DIR = os.path.join(_EXAMPLES_DIR, "capacity_ladder_results", "D_TOY_PROBES")


class Group(enum.StrEnum):
    """Which group the word-problem toy composes over (closed set)."""

    S5 = "s5"  # non-solvable: width provably cannot substitute for depth (Barrington 1989)
    Z120 = "z120"  # abelian/solvable control of the same order: width CAN substitute
    A5 = "a5"  # non-solvable, order 60, involution (order-2 generator) alphabet: the depth-SELECTION toy's group (D8b)


class NetKind(enum.StrEnum):
    """Which architecture the probe trains (closed set)."""

    MLP = "mlp"  # plain feedforward on the flattened one-hot word (the direct depth-vs-width test)
    RECURRENT = "recurrent"  # weight-shared block applied once per word position (the natural sequential form)


# ---------------------------------------------------------------------------
# Group construction — element list, index map, multiplication table, generator element indices.
# ---------------------------------------------------------------------------


def _compose_perm(a: tuple[int, ...], b: tuple[int, ...]) -> tuple[int, ...]:
    """Permutation composition `(a∘b)[i] = a[b[i]]` — apply `b` first, then `a`."""
    return tuple(a[b[i]] for i in range(len(a)))


def _adjacent_transposition(n: int, i: int) -> tuple[int, ...]:
    """The permutation of `range(n)` that swaps positions `i` and `i+1`, identity elsewhere."""
    p = list(range(n))
    p[i], p[i + 1] = p[i + 1], p[i]
    return tuple(p)


def _is_even_permutation(perm: tuple[int, ...]) -> bool:
    """True if `perm` (a permutation of `range(len(perm))`) is EVEN (a product of an even number of transpositions).

    Standard cycle-decomposition parity: a cycle of length `c` is `c-1` transpositions; sum those over
    all cycles and check the total is even. Used to build A5 (the even-permutation subgroup of S5).
    """
    n = len(perm)
    seen = [False] * n
    parity = 0
    for i in range(n):
        if seen[i]:
            continue
        j, clen = i, 0
        while not seen[j]:
            seen[j] = True
            j = perm[j]
            clen += 1
        parity += clen - 1
    return parity % 2 == 0


def build_group(group: Group) -> dict:
    """Return `{elements, index, mult, identity, generators}` for `group`.

    `elements`: list of the group elements (perm tuples for S5, ints for Z120). `index`: element ->
    int id. `mult`: `(|G|, |G|)` int array, `mult[i, j]` = id of `elements[i] * elements[j]`
    (composition `a∘b` for S5; `(a+b) mod order` for Z120). `identity`: id of the identity element.
    `generators`: tuple of element ids used as the word alphabet (4 for both groups).
    """
    if group is Group.S5:
        elements = list(itertools.permutations(range(S5_N)))
        index = {e: i for i, e in enumerate(elements)}
        n = len(elements)
        mult = np.empty((n, n), dtype=np.int64)
        for i, a in enumerate(elements):
            for j, b in enumerate(elements):
                mult[i, j] = index[_compose_perm(a, b)]
        identity = index[tuple(range(S5_N))]
        generators = tuple(index[_adjacent_transposition(S5_N, i)] for i in range(S5_N - 1))
        return {"elements": elements, "index": index, "mult": mult, "identity": identity, "generators": generators}

    if group is Group.A5:
        elements = [e for e in itertools.permutations(range(A5_N)) if _is_even_permutation(e)]
        index = {e: i for i, e in enumerate(elements)}
        n = len(elements)
        mult = np.empty((n, n), dtype=np.int64)
        for i, a in enumerate(elements):
            for j, b in enumerate(elements):
                mult[i, j] = index[_compose_perm(a, b)]
        identity = index[tuple(range(A5_N))]
        generators = tuple(index[g] for g in A5_GENERATORS)
        return {"elements": elements, "index": index, "mult": mult, "identity": identity, "generators": generators}

    # Z120: elements are 0..order-1; composition is addition mod order (abelian => solvable).
    order = Z_ORDER
    elements = list(range(order))
    index = {e: e for e in elements}
    mult = np.array([[(a + b) % order for b in elements] for a in elements], dtype=np.int64)
    identity = 0
    generators = tuple(g % order for g in Z_GENERATORS)
    return {"elements": elements, "index": index, "mult": mult, "identity": identity, "generators": generators}


def word_product(word_gen_ids: np.ndarray, grp: dict) -> np.ndarray:
    """Fold each length-`L` word (row of generator element-ids) into its product id via `grp['mult']`.

    Left-to-right application: `P_0 = identity`, `P_t = elements[g_t] ∘ P_{t-1}` — matches reading the
    word `g_1 g_2 … g_L` as "apply `g_1` first". Vectorized over all rows at once.
    """
    mult = grp["mult"]
    n_rows, seq_len = word_gen_ids.shape
    prod = np.full(n_rows, grp["identity"], dtype=np.int64)
    for t in range(seq_len):
        prod = mult[word_gen_ids[:, t], prod]  # mult[g_t, P_{t-1}] = g_t ∘ P_{t-1}
    return prod


# ---------------------------------------------------------------------------
# Data — words of a fixed length over the group's generator alphabet, one-hot encoded, with a
# generalization (held-out) split. Metric is accuracy on UNSEEN words.
# ---------------------------------------------------------------------------


def _all_or_sampled_words(n_gen: int, seq_len: int, seed: int) -> np.ndarray:
    """All `n_gen**seq_len` words if that is `<= MAX_ENUM`, else `MAX_ENUM` uniformly-sampled words.

    Returns an `(n_words, seq_len)` int array of GENERATOR INDICES (0..n_gen-1), shuffled by `seed`.
    """
    total = n_gen**seq_len
    rng = np.random.default_rng(seed)
    if total <= MAX_ENUM:
        grid = np.array(list(itertools.product(range(n_gen), repeat=seq_len)), dtype=np.int64)
        rng.shuffle(grid)
        return grid
    return rng.integers(0, n_gen, size=(MAX_ENUM, seq_len), dtype=np.int64)


def make_word_data(group: Group, seq_len: int, seed: int, train_frac: float = DEFAULT_TRAIN_FRAC) -> dict:
    """Build the (train, val) one-hot word dataset + product labels for `group` at length `seq_len`.

    Returns a dict with `x_tr/x_val` float32 `(n, seq_len*n_gen)` flattened one-hot words,
    `y_tr/y_val` int64 product ids, plus `n_classes`, `n_gen`, `seq_len`, `grp` (the group dict).
    Train/val are a `train_frac` split of DISTINCT words, so val measures generalization to unseen
    words (a net that learned the composition algorithm generalizes; a memorizer does not).
    """
    grp = build_group(group)
    n_gen = len(grp["generators"])
    gen_arr = np.array(grp["generators"], dtype=np.int64)

    words_gen = _all_or_sampled_words(n_gen, seq_len, seed)  # generator INDICES
    word_elem_ids = gen_arr[words_gen]  # -> generator ELEMENT ids for the product fold
    labels = word_product(word_elem_ids, grp)

    n = words_gen.shape[0]
    n_tr = round(train_frac * n)
    onehot = np.eye(n_gen, dtype=np.float32)[words_gen].reshape(n, seq_len * n_gen)

    return {
        "x_tr": onehot[:n_tr],
        "y_tr": labels[:n_tr],
        "x_val": onehot[n_tr:],
        "y_val": labels[n_tr:],
        "n_classes": len(grp["elements"]),
        "n_gen": n_gen,
        "seq_len": seq_len,
        "grp": grp,
    }


# ---------------------------------------------------------------------------
# Nets — a plain feedforward classifier family (narrow depth-d and wide-shallow) plus a
# recurrent-unrolled classifier (weight-shared block applied once per word position).
# ---------------------------------------------------------------------------


def build_narrow_clf(depth: int, width: int, in_dim: int, n_classes: int) -> nn.Sequential:
    """`Linear(in,width)->Tanh->[Linear(width,width)->Tanh]*(depth-1)->Linear(width,n_classes)`."""
    if depth < 1:
        raise ValueError(f"depth must be >= 1, got {depth}")
    layers: list[nn.Module] = [nn.Linear(in_dim, width), nn.Tanh()]
    for _ in range(depth - 1):
        layers += [nn.Linear(width, width), nn.Tanh()]
    layers += [nn.Linear(width, n_classes)]
    return nn.Sequential(*layers)


def build_wide_shallow_clf(width: int, in_dim: int, n_classes: int) -> nn.Sequential:
    """`Linear(in,width)->Tanh->Linear(width,n_classes)` — the depth-1 width-substitution control."""
    return nn.Sequential(nn.Linear(in_dim, width), nn.Tanh(), nn.Linear(width, n_classes))


class RecurrentComposer(nn.Module):
    """Weight-shared sequential classifier: one shared block folds the word left-to-right.

    `state_0 = 0`; for each position `t`, `state = tanh(block([state, onehot(g_t)]))`; then
    `readout(state)`. Unrolling depth = `seq_len` with SHARED weights — the natural "apply one more
    generator per step" inductive bias, and the per-input depth dial itself (unroll d times = depth-d
    for a length-d word). The transition block is a 2-layer MLP (`[state,gen] -> block_hidden ->
    state`): a single Linear was too weak to represent the group multiplication (only 0.46 val at
    width 16), so the block gets its own hidden layer.
    """

    def __init__(self, width: int, n_gen: int, n_classes: int, block_hidden: int | None = None) -> None:
        """Shared 2-layer transition block over `[state(width), onehot_gen(n_gen)] -> width`, then readout."""
        super().__init__()
        self.width = int(width)
        self.n_gen = int(n_gen)
        h = block_hidden if block_hidden is not None else width
        self.block = nn.Sequential(nn.Linear(width + n_gen, h), nn.Tanh(), nn.Linear(h, width))
        self.readout = nn.Linear(width, n_classes)

    def forward(self, x_flat: torch.Tensor) -> torch.Tensor:
        """`x_flat` is `(N, seq_len*n_gen)` flattened one-hot; fold with the shared block, then read out."""
        n = x_flat.shape[0]
        seq = x_flat.view(n, -1, self.n_gen)  # (N, seq_len, n_gen)
        state = torch.zeros(n, self.width, device=x_flat.device)
        for t in range(seq.shape[1]):
            state = torch.tanh(self.block(torch.cat([state, seq[:, t, :]], dim=1)))
        return self.readout(state)


def count_params(module: nn.Module) -> int:
    """Total learnable scalar parameter count."""
    return sum(p.numel() for p in module.parameters())


# ---------------------------------------------------------------------------
# Cross-entropy training, convergence-gated on val CE (full-trajectory rule), + held-out accuracy.
# ---------------------------------------------------------------------------


def train_clf(net: nn.Module, data: dict, device: str = "cpu", max_epochs: int = MAX_EPOCHS) -> tuple[cvg.ConvergenceResult, float, float]:
    """Full-batch Adam + cross-entropy, gated by `convergence.fit_to_convergence` on val CE.

    Returns `(convergence_result, train_acc, val_acc)`; accuracies are top-1 on the (unseen) val words
    and on the train words, both in `[0,1]`.
    """
    net.to(device)
    x_tr = torch.as_tensor(data["x_tr"], dtype=torch.float32, device=device)
    y_tr = torch.as_tensor(data["y_tr"], dtype=torch.long, device=device)
    x_val = torch.as_tensor(data["x_val"], dtype=torch.float32, device=device)
    y_val = torch.as_tensor(data["y_val"], dtype=torch.long, device=device)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    ce = nn.CrossEntropyLoss()

    def step_fn() -> None:
        opt.zero_grad()
        loss = ce(net(x_tr), y_tr)
        loss.backward()
        opt.step()

    def val_fn() -> float:
        net.eval()
        with torch.no_grad():
            v = ce(net(x_val), y_val).item()
        net.train()
        return v

    result = cvg.fit_to_convergence(net, step_fn, val_fn, max_epochs=max_epochs, check_every=CHECK_EVERY, patience=PATIENCE, min_delta=MIN_DELTA)
    net.eval()
    with torch.no_grad():
        train_acc = float((net(x_tr).argmax(1) == y_tr).float().mean().item())
        val_acc = float((net(x_val).argmax(1) == y_val).float().mean().item())
    return result, train_acc, val_acc


# ---------------------------------------------------------------------------
# Pilot — for a fixed (group, seq_len, net kind): train the narrow depth ladder + the wide-shallow
# control, report held-out accuracy per depth. The depth-separation signature is: deep-narrow FITS,
# wide-shallow STALLS (for S5); for the Z120 control, wide-shallow FITS.
# ---------------------------------------------------------------------------


def run_pilot(
    group: Group,
    seq_len: int,
    seed: int,
    net_kind: NetKind = NetKind.MLP,
    narrow_width: int = NARROW_WIDTH,
    wide_width: int = WIDE_WIDTH,
    rec_state_width: int = REC_STATE_WIDTH,
    depths: tuple[int, ...] = PROBE_DEPTHS,
    device: str = "cpu",
) -> dict:
    """Train the narrow depth ladder + wide-shallow control on `group` length-`seq_len` words; report accuracy."""
    data = make_word_data(group, seq_len, seed)
    n_classes = data["n_classes"]
    in_dim = data["seq_len"] * data["n_gen"]
    chance = 1.0 / n_classes

    def _build_narrow(d: int) -> nn.Module:
        if net_kind is NetKind.RECURRENT:
            return RecurrentComposer(rec_state_width, data["n_gen"], n_classes)  # depth is intrinsic (=seq_len, shared)
        return build_narrow_clf(d, narrow_width, in_dim, n_classes)

    narrow_rows: dict[int, dict] = {}
    ladder_depths = (data["seq_len"],) if net_kind is NetKind.RECURRENT else depths
    for d in ladder_depths:
        torch.manual_seed(1000 * seed + d)
        net = _build_narrow(d)
        result, tr_acc, val_acc = train_clf(net, data, device=device)
        narrow_rows[d] = {
            "depth": d, "width": narrow_width, "params": count_params(net),
            "train_acc": tr_acc, "val_acc": val_acc,
            "trustworthy": bool(result.trustworthy), "convergence": result.summary(),
        }

    torch.manual_seed(1000 * seed + 777)
    wide_net = build_wide_shallow_clf(wide_width, in_dim, n_classes)  # control is always a plain wide-shallow MLP
    w_result, w_tr_acc, w_val_acc = train_clf(wide_net, data, device=device)
    wide_row = {
        "width": wide_width, "params": count_params(wide_net),
        "train_acc": w_tr_acc, "val_acc": w_val_acc,
        "trustworthy": bool(w_result.trustworthy), "convergence": w_result.summary(),
    }

    return {
        "group": group.value,
        "net_kind": net_kind.value,
        "seq_len": seq_len,
        "seed": seed,
        "n_classes": n_classes,
        "chance": chance,
        "n_train": int(data["x_tr"].shape[0]),
        "n_val": int(data["x_val"].shape[0]),
        "narrow_width": narrow_width,
        "wide_width": wide_width,
        "fit_acc": FIT_ACC,
        "stall_acc": STALL_ACC,
        "narrow": {str(d): v for d, v in narrow_rows.items()},
        "wide_shallow": wide_row,
    }


def _pilot_path(out_dir: str, group: Group, net_kind: NetKind, seq_len: int, seed: int) -> str:
    return os.path.join(out_dir, f"depth_comp_pilot_{group.value}_{net_kind.value}_n{seq_len}_seed{seed}.json")


def run_and_save_pilot(group: Group, seq_len: int, seed: int, net_kind: NetKind = NetKind.MLP, out_dir: str = DEFAULT_OUT_DIR, device: str = "cpu") -> dict:
    """`run_pilot` + immediate JSON land (standing clause: land results the moment they exist)."""
    os.makedirs(out_dir, exist_ok=True)
    pilot = run_pilot(group, seq_len, seed, net_kind=net_kind, device=device)
    path = _pilot_path(out_dir, group, net_kind, seq_len, seed)
    with open(path, "w") as f:
        json.dump(pilot, f, indent=2)
    return pilot


# ---------------------------------------------------------------------------
# Selftest — group axioms (identity/associativity/closure), generator validity, product correctness,
# data shapes. No training.
# ---------------------------------------------------------------------------


def _check_group_axioms(group: Group) -> bool:
    """Identity acts trivially; composition is associative; every sampled element has a two-sided inverse; generators generate the whole group."""
    grp = build_group(group)
    mult, ident, n = grp["mult"], grp["identity"], len(grp["elements"])
    rng = np.random.default_rng(0)
    ids = rng.integers(0, n, size=30)

    id_ok = all(mult[i, ident] == i and mult[ident, i] == i for i in ids)
    trips = rng.integers(0, n, size=(30, 3))
    assoc_ok = all(mult[mult[a, b], c] == mult[a, mult[b, c]] for a, b, c in trips)

    # inverses: for each sampled element, exactly one j solves mult[i,j]==identity, and it is two-sided.
    inv_ok = True
    for i in ids:
        row_hits = np.flatnonzero(mult[i] == ident)
        if row_hits.shape[0] != 1:
            inv_ok = False
            break
        j = int(row_hits[0])
        if mult[j, i] != ident:
            inv_ok = False
            break

    # closure/generation: BFS from identity under right-multiplication by generators must reach all |G|.
    seen = {ident}
    frontier = [ident]
    while frontier:
        nxt = []
        for x in frontier:
            for g in grp["generators"]:
                y = int(mult[x, g])
                if y not in seen:
                    seen.add(y)
                    nxt.append(y)
        frontier = nxt
    gen_ok = len(seen) == n

    ok = bool(id_ok and assoc_ok and inv_ok and gen_ok)
    print(
        f"[depth_comp selftest] ({group.value}) |G|={n} identity={id_ok} associative={assoc_ok} inverses={inv_ok} "
        f"generators_span={gen_ok} ({len(seen)}/{n})  {'PASS' if ok else 'FAIL'}"
    )
    return ok


def _check_product_and_data(group: Group) -> bool:
    """A hand-checked short product + data-tensor shape/one-hot/label-consistency checks."""
    grp = build_group(group)
    # product of the single-generator word [g0] must equal generator element g0 (fold starts at identity).
    g0 = grp["generators"][0]
    single = word_product(np.array([[g0]], dtype=np.int64), grp)[0]
    single_ok = int(single) == g0

    # product of [g0, g1] must equal mult[g1, g0] (g0 applied first, then g1).
    g1 = grp["generators"][1]
    pair = word_product(np.array([[g0, g1]], dtype=np.int64), grp)[0]
    pair_ok = int(pair) == int(grp["mult"][g1, g0])

    data = make_word_data(group, seq_len=4, seed=0)
    n_gen, seq_len = data["n_gen"], data["seq_len"]
    x = np.concatenate([data["x_tr"], data["x_val"]], axis=0)
    shape_ok = x.shape[1] == seq_len * n_gen
    onehot_ok = bool(np.allclose(x.reshape(x.shape[0], seq_len, n_gen).sum(axis=2), 1.0))  # exactly one generator per position
    disjoint_ok = data["x_tr"].shape[0] > 0 and data["x_val"].shape[0] > 0

    ok = bool(single_ok and pair_ok and shape_ok and onehot_ok and disjoint_ok)
    prod_str = f"product single={single_ok} pair={pair_ok}"
    data_str = f"data shape={shape_ok} onehot={onehot_ok} split={disjoint_ok}"
    print(f"[depth_comp selftest] ({group.value}) {prod_str} | {data_str}  {'PASS' if ok else 'FAIL'}")
    return ok


def run_selftest() -> bool:
    """Group axioms + generator validity + product correctness + data shape, for both groups. No training."""
    ok = True
    for group in (Group.S5, Group.Z120, Group.A5):
        ok = _check_group_axioms(group) and ok
        ok = _check_product_and_data(group) and ok
    # net wiring: each builder returns the right output width on a dummy batch.
    dummy = torch.zeros(3, 4 * 4)  # seq_len=4, n_gen=4
    for name, net in (("narrow", build_narrow_clf(3, 8, 16, 120)), ("wide", build_wide_shallow_clf(32, 16, 120)), ("recurrent", RecurrentComposer(8, 4, 120))):
        out = net(dummy)
        shape_ok = tuple(out.shape) == (3, 120)
        print(f"[depth_comp selftest] net '{name}' output shape {tuple(out.shape)} (expect (3, 120))  {'PASS' if shape_ok else 'FAIL'}")
        ok = shape_ok and ok
    print(f"[depth_comp selftest] {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> None:
    """Parses args and runs `--selftest` or a `--pilot`, else prints help."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true", help="No-training group-axiom / product / data / net-wiring checks.")
    parser.add_argument("--pilot", action="store_true", help="Train the narrow depth ladder + wide-shallow control and report held-out accuracy.")
    parser.add_argument("--group", type=str, choices=[g.value for g in Group], default=Group.S5.value, help="Group to compose over (default s5).")
    parser.add_argument("--net", type=str, choices=[k.value for k in NetKind], default=NetKind.MLP.value, help="Architecture (default mlp).")
    parser.add_argument("--seq-len", type=int, default=5, help="Word length = number of composition steps (per-input depth difficulty knob).")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed.")
    parser.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR, help="Directory for pilot JSON output.")
    args = parser.parse_args()

    if args.selftest:
        sys.exit(0 if run_selftest() else 1)

    if args.pilot:
        device = os.environ.get("AUTOML_DEVICE", "cpu")
        group, net_kind = Group(args.group), NetKind(args.net)
        pilot = run_and_save_pilot(group, args.seq_len, args.seed, net_kind=net_kind, out_dir=args.out_dir, device=device)
        path = _pilot_path(args.out_dir, group, net_kind, args.seq_len, args.seed)
        print(f"[depth_comp] wrote {path}")
        print(f"  group={pilot['group']} net={pilot['net_kind']} seq_len={pilot['seq_len']} chance={pilot['chance']:.4f} n_train={pilot['n_train']} n_val={pilot['n_val']}")
        for d, row in pilot["narrow"].items():
            print(f"  narrow depth={d:>2} (w={row['width']}): val_acc={row['val_acc']:.3f} train_acc={row['train_acc']:.3f} trustworthy={row['trustworthy']}")
        w = pilot["wide_shallow"]
        print(f"  wide-shallow (w={w['width']}): val_acc={w['val_acc']:.3f} train_acc={w['train_acc']:.3f} trustworthy={w['trustworthy']}")
        sys.exit(0)

    parser.print_help()


if __name__ == "__main__":
    main()
