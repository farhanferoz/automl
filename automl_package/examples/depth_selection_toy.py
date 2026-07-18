"""Depth-SELECTION toy — per-input depth choice WITHOUT an oracle (D8b).

Design doc: `docs/depth_capacity/depth_selection_toy_design.md` (esp. §2 arithmetic, §3.1 generative
spec, §3.2 net+dial, §3.4 gradedness, §3.5 router, §3.6 bars S1-S5, §6 RESOLVED = Option A: concealment
DROPPED as a kill criterion, a surface-baseline control added instead; construction = A5, L=10 —
shortened from the design doc's L=16 after root-causing the anytime net's chance-level training, see
the root-cause note below).
Plan: `docs/plans/capacity_programme/depth.md` Task D8b.

**Construction (design §6, C1'''):** words of length L=10 over a 4-letter alphabet of A5 involutions
(order-2 generators; `depth_composition_toy.Group.A5` / `build_group` — extended there, not here, so S5/Z120 share it).
Each word has a hidden REALIZED commitment point t*(x): the smallest t such that the length-t prefix
product already equals the word's full product (equivalently, the length-(L-t) suffix folds to the
identity). Words are drawn UNIFORMLY from a realized stratum {x : t*(x) = t} via first-hit dynamic
programming (design §3.1): the final label f is chosen with probability proportional to the number of
length-t paths that first-hit f at step t (never earlier), the length-t PREFIX is backward-sampled
under that first-hit ("taboo") DP, and the length-(L-t) SUFFIX is independently backward-sampled from
the unconstrained identity-fold DP. Because the whole word is uniform given its stratum, there is no
prefix/tail distributional seam for a surface net to key on (design §3.0's fix for the round-1/round-2
falsifier rejections). EVERY generated word is assertion-checked: `realized_commitment` recomputes
t*(x) from scratch (an O(L) scan over ALL prefixes, NO contiguity assumption — design §3.1 flags this
exact spot as where a prior probe round had a bug) and the result must equal the constructed t.

**Anytime net (design §3.2, REVISED — root-cause note below):** ONE shared recurrent block + ONE
shared linear readout (`depth_composition_toy.RecurrentComposer`, reused UNCHANGED), applied to
however many positions are present in the sliced input — the per-position dial T (2, 4, ..., 10) is
still realized by slicing the one-hot word to its first `T*n_gen` columns before the call, but the
readout is now SHARED across every T instead of per-T. Each exit T is trained against the T-step
RUNNING (prefix) product of the word (`word_all_prefix_products`), NOT the word's full product — the
recurrent state after T steps encodes exactly the prefix product, so this target is always achievable,
unlike the full product for T < t*(x). Loss = mean over `T_LADDER` of
`cross_entropy(readout(state_T), prefix_product_T)`, gradient-clipped (`GRAD_CLIP_MAX_NORM=1.0`)
before every optimizer step.

**Root-cause note (verified on A5 seed 0, do not re-litigate without new experimental evidence):** the
original design (L=16, `depth_graded_toy.RecurrentPerLengthHead`'s per-T HEADS, every head trained on
the FULL-word label) trained to CHANCE. Two compounding faults: (1) L=16 is past the plain
`RecurrentComposer`'s GD-trainable wall on A5 (it fails to even fit the training set at length 12+), so
the sequence length was shortened to L=10 (`COMMIT_LADDER=(6,8,10)`, `T_LADDER=(2,4,...,10)`) to stay
inside the trainable range. (2) for T < t*(x) the full-word answer is not determined by the length-T
prefix, so those per-T-head exits carried IMPOSSIBLE targets whose gradients corrupted the shared
block; switching to a SHARED readout trained on the RUNNING prefix product (always achievable at every
T) removes that impossible-target corruption. The DEPLOY answer is still the word's FULL product at
every T (running T steps yields the prefix product, which equals the full product iff T >= t*(x)), so
the per-input error table used for the S1/S2 bars, the router, and deploy compares the net's T-step
prediction against the FULL product — never the training-time prefix target.

**Gradedness (design §3.4/§3.6 S2, THE MAKE-OR-BREAK):** for T >= t*(x) the label is a deterministic
function of the consumed prefix (achievable acc 1.0); for T < t*(x) the Bayes ceiling with
g = t*(x)-T unread letters applies (computed exactly in `compute_arithmetic_table`, no assumption).
`--probe gradedness` trains the anytime net and checks, per stratum: mean acc(x, T=t*(x)) >= 0.95 *
mean acc(x, T=full) [[S2_KNEE_FRACTION]], and mean acc(x, T=t*(x)-2) <= Bayes(g=2)+10pp
(`S2_CEILING_A5`, frozen by `--probe arithmetic` at the CURRENT L, NOT the S5 design doc's 0.35 — A5's
alphabet has a different branching structure, see that probe's printed deviation note).

**Surface-baseline control (design §6 Option A, replaces the dropped concealment kill):**
`--probe surface` trains a plain 1-hidden-layer MLP (`depth_composition_toy.build_narrow_clf(depth=1)`,
reused verbatim) on the raw one-hot word to predict which stratum (t* value) the word belongs to, and
reports per-stratum balanced accuracy vs chance. This is reported as a COVARIATE, never a kill
criterion (user ruling, design doc §6: "difficulty-of-detection is not our concern").

Usage:
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python automl_package/examples/depth_selection_toy.py --selftest
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python automl_package/examples/depth_selection_toy.py --probe arithmetic
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python automl_package/examples/depth_selection_toy.py \
        --probe gradedness --n-per-stratum 40000 --max-epochs 40000   # the real pilot (main thread only)
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python automl_package/examples/depth_selection_toy.py --probe surface
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnf
from scipy.stats import spearmanr

_EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _EXAMPLES_DIR)
sys.path.insert(0, os.path.dirname(os.path.dirname(_EXAMPLES_DIR)))  # repo root, so `import automl_package` works

import capacity_ladder_k6 as ck6  # noqa: E402 — reuse HIDDEN/N_EPOCHS/LR router hyperparameter conventions
import convergence as cvg  # noqa: E402 — full-trajectory convergence gate, shared with the other depth toys
import sinc_width_experiment as sw  # noqa: E402 — reuse `_cheapest_within_tolerance_labels`/`_deploy_bar_mse` (metric-agnostic)
from depth_composition_toy import (  # noqa: E402 — reuse the certified group/net building blocks, don't reinvent
    REC_STATE_WIDTH,
    Group,
    RecurrentComposer,
    build_group,
    build_narrow_clf,
)
from depth_composition_toy import train_clf as _train_clf_generic  # noqa: E402 — generic CE+convergence trainer, reused for the surface probe
from depth_graded_toy import CHECK_EVERY, MIN_DELTA, PATIENCE  # noqa: E402 — shared convergence-gate hyperparameter convention (anytime-net training loop)

# ---------------------------------------------------------------------------
# Construction constants (design §6 RESOLVED: C1''' = A5 involutions; L SHORTENED 16->10 and the ladder
# to {6,8,10} after root-causing the anytime net's chance-level training — see module docstring).
# ---------------------------------------------------------------------------

L = 10  # syntactic word length (was 16; A5's plain RecurrentComposer fails to fit at length 12+, see root-cause note)
COMMIT_LADDER = (6, 8, 10)  # realized-commitment strata (even, >=6 per design constraint 2; 10 = L = "never commits early")
T_LADDER = tuple(range(2, L + 1, 2))  # anytime-net dial: T in {2,4,6,8,10}, compute proportional to T
N_PER_STRATUM_DEFAULT = 40000  # design §3.1 dataset arithmetic (~40k words/stratum)
TRAIN_FRAC = 0.5  # 50/50 train/val split of DISTINCT words (certified-toy convention)

LR_DEFAULT = 3e-3  # canonical recurrent-arm LR (depth_graded_toy's n=10 trainability fix)
MAX_EPOCHS_DEFAULT = 40000  # inherited from depth_graded_toy's CHECK_EVERY/PATIENCE/MIN_DELTA convergence-gate convention
GRAD_CLIP_MAX_NORM = 1.0  # anytime-net trainability fix (root-cause #1: L=10 needs clipping to stay GD-trainable)

S1_FIT_ACC = 0.90  # S1 substrate bar: full-T held-out acc >= this, per stratum with t* >= 6
S2_KNEE_FRACTION = 0.95  # S2 knee bar: acc(x, T=t*) >= this fraction of acc(x, T=full)
S2_CEILING_MARGIN_PP = 0.10  # S2 ceiling bar margin over the exact Bayes(g=2) accuracy
BAYES_G_FOR_CEILING = 2  # the "T = t*(x) - 2" read point (g=2 unread letters)

SURFACE_MLP_WIDTH = 32  # shallow-probe hidden width (falsifier/surface-baseline control, design §3.3/§6)
DELTA_TIE = sw.DELTA_TIE  # cheapest-within-tolerance router tolerance, reused verbatim (width-MSE program convention)
S3_ROUTER_SPEARMAN_MIN = 0.7  # S3 router-quality bar: rank corr between the router's deployed T and realized t*(x)
MIN_RELATION_LENGTH = 2  # A5 involution alphabet (order-2 gens): shortest identity relation is g*g at length 2

DEFAULT_OUT_DIR = os.path.join(_EXAMPLES_DIR, "capacity_ladder_results", "D_TOY_PROBES")

_A5_GRP = build_group(Group.A5)  # built once at import time: {elements, index, mult, identity, generators}
_A5_MULT = _A5_GRP["mult"]
_A5_IDENTITY = _A5_GRP["identity"]
_A5_GENERATORS = _A5_GRP["generators"]  # 4 element-ids (into _A5_GRP["elements"]), the word alphabet
_A5_N_CLASSES = len(_A5_GRP["elements"])
_A5_N_GEN = len(_A5_GENERATORS)


def _invert_permutation(perm: tuple[int, ...]) -> tuple[int, ...]:
    """The inverse permutation of `perm` (`perm[i]` = image of `i`)."""
    inv = [0] * len(perm)
    for i, p in enumerate(perm):
        inv[p] = i
    return tuple(inv)


def _generator_inverse_ids(grp: dict) -> tuple[int, ...]:
    """Element-id of the inverse of each generator in `grp['generators']` (needed for backward sampling)."""
    elements, index = grp["elements"], grp["index"]
    return tuple(index[_invert_permutation(elements[g])] for g in grp["generators"])


_A5_GEN_INV = _generator_inverse_ids(_A5_GRP)


# ---------------------------------------------------------------------------
# Exact arithmetic (design §2's S5 table, re-derived for A5): reachable-class support, identity-fold
# word counts, and the Bayes-optimal accuracy with g unread letters, all by word length. No RNG.
# ---------------------------------------------------------------------------


def _step(v: np.ndarray, gens: tuple[int, ...], mult: np.ndarray) -> np.ndarray:
    """One forward step of the alphabet random walk: `v_new[x] = sum_g v[y]` over `y` with `mult[g,y]==x`."""
    new_v = np.zeros_like(v)
    for g in gens:
        np.add.at(new_v, mult[g], v)
    return new_v


def compute_arithmetic_table(max_len: int = L, grp: dict | None = None) -> dict:
    """Exact per-length table: reachable-class support, identity-fold word count, Bayes(g) ceiling, TV-to-uniform.

    The A5 analogue of the design doc's §2 S5 table. `bayes_acc` at length `g` is exactly the
    Bayes-optimal accuracy for predicting a length-g random word's product when g letters are unread
    (the argument for why this is independent of the known prefix is in this module's docstring / the
    design doc §3.4: left-multiplication by a fixed prefix is a bijection of the group, so it permutes
    the tail distribution without changing its mode).
    """
    grp = grp or _A5_GRP
    mult, ident, gens = grp["mult"], grp["identity"], grp["generators"]
    n = len(grp["elements"])
    uniform = 1.0 / n
    v = np.zeros(n, dtype=np.int64)
    v[ident] = 1
    rows = []
    for t in range(1, max_len + 1):
        v = _step(v, gens, mult)
        total = len(gens) ** t
        support = int((v > 0).sum())
        id_count = int(v[ident])
        probs = v.astype(np.float64) / total
        bayes = float(probs.max())
        tv = 0.5 * float(np.abs(probs - uniform).sum())
        rows.append({"t": t, "support": support, "identity_count": id_count, "bayes_acc": bayes, "tv_to_uniform": tv})
    return {"n_classes": n, "n_gen": len(gens), "rows": rows}


def s2_ceiling_a5(table: dict, g: int = BAYES_G_FOR_CEILING, margin: float = S2_CEILING_MARGIN_PP) -> float:
    """Freeze the S2 make-or-break ceiling: `Bayes(g) + margin`, computed from THIS group's own arithmetic."""
    row = next(r for r in table["rows"] if r["t"] == g)
    return row["bayes_acc"] + margin


def confirm_ladder(table: dict, ladder: tuple[int, ...] = COMMIT_LADDER) -> dict:
    """Per-rung class-coverage check (design constraint 2): the stratum's word length must reach full support."""
    n_classes = table["n_classes"]
    by_t = {r["t"]: r for r in table["rows"]}
    per_t = {t: {"support": by_t[t]["support"], "full_coverage": bool(by_t[t]["support"] == n_classes)} for t in ladder if t in by_t}
    return {"ladder": list(ladder), "n_classes": n_classes, "per_t": per_t, "coverage_safe": bool(all(c["full_coverage"] for c in per_t.values()))}


# ---------------------------------------------------------------------------
# Realized commitment point — the smallest t with prefix(t) == full product. NO contiguity assumption:
# scans every prefix (design §3.1 flags exactly this spot as where a prior probe round had a bug that
# assumed the answer was monotonic/contiguous and scored every word t*=L).
# ---------------------------------------------------------------------------


def word_prefix_products(word_gen_ids: np.ndarray | list[int], grp: dict | None = None) -> list[int]:
    """`[prefix_product(1), ..., prefix_product(len(word))]` for one word (generator ALPHABET indices)."""
    grp = grp or _A5_GRP
    mult, gens, ident = grp["mult"], grp["generators"], grp["identity"]
    prod = ident
    prods = []
    for gi in word_gen_ids:
        prod = int(mult[gens[gi], prod])
        prods.append(prod)
    return prods


def word_all_prefix_products(word_gen_ids: np.ndarray, grp: dict | None = None) -> np.ndarray:
    """Vectorized batch twin of `word_prefix_products`.

    `(n, word_len)` generator-ALPHABET-index words -> `(n, word_len)`
    running prefix-product ids (column `t-1` = the length-`t` prefix product), used as the anytime net's
    per-T TRAINING target (the running product, not the word's full product — see module docstring's
    root-cause note). Must agree with the scalar `word_prefix_products` on every row (checked in selftest).
    """
    grp = grp or _A5_GRP
    mult, gens, ident = grp["mult"], grp["generators"], grp["identity"]
    n, word_len = word_gen_ids.shape
    gens_arr = np.array(gens, dtype=np.int64)
    prod = np.full(n, ident, dtype=np.int64)
    out = np.empty((n, word_len), dtype=np.int64)
    for t in range(word_len):
        prod = mult[gens_arr[word_gen_ids[:, t]], prod]
        out[:, t] = prod
    return out


def realized_commitment(word_gen_ids: np.ndarray | list[int], grp: dict | None = None) -> int:
    """Smallest `t` such that `prefix_product(t) == full_product` — an O(L) full scan, no shortcuts."""
    prods = word_prefix_products(word_gen_ids, grp)
    full = prods[-1]
    for t, p in enumerate(prods, start=1):
        if p == full:
            return t
    return len(prods)  # unreachable (t=len always matches full by definition), kept as a defensive fallback


# ---------------------------------------------------------------------------
# Commitment-point word generator (design §3.1): first-hit ("taboo") DP for the prefix, unconstrained
# DP for the identity-fold suffix, both backward-sampled from cached forward tables.
# ---------------------------------------------------------------------------


def _build_taboo_tables(t: int, forbidden: int, gens: tuple[int, ...], mult: np.ndarray, n_classes: int, ident: int) -> list[np.ndarray]:
    """`u_0..u_{t-1}`: length-s paths from identity that avoid state `forbidden` at every step 1..s.

    `u_0` is the trivial delta at `identity` (no steps taken yet, so there is nothing to avoid). Taboo
    zeroing is applied AFTER each step so `u_s` never has mass on `forbidden` for `s < t`.
    """
    u = np.zeros(n_classes, dtype=np.float64)
    u[ident] = 1.0
    tables = [u.copy()]
    for _s in range(1, t):
        u = _step(u, gens, mult)
        u[forbidden] = 0.0
        tables.append(u.copy())
    return tables


def _build_unconstrained_tables(m: int, gens: tuple[int, ...], mult: np.ndarray, n_classes: int, ident: int) -> list[np.ndarray]:
    """`w_0..w_{m-1}`: length-s paths from identity, UNCONSTRAINED (no taboo) — for the suffix's backward sampler."""
    w = np.zeros(n_classes, dtype=np.float64)
    w[ident] = 1.0
    tables = [w.copy()]
    for _s in range(1, m):
        w = _step(w, gens, mult)
        tables.append(w.copy())
    return tables


def _first_hit_distribution(t: int, gens: tuple[int, ...], mult: np.ndarray, n_classes: int, ident: int) -> tuple[np.ndarray, dict[int, list[np.ndarray]]]:
    """For every candidate final label `f`: the taboo tables (cached for reuse) and the first-hit-at-`f` count.

    `weights[f]` = number of length-`t` words whose running prefix product hits `f` for the FIRST time
    exactly at step `t` (never at steps `1..t-1`) — the design's "#length-t paths first-hitting f" factor.
    The companion "#length-(L-t) identity-fold suffixes" factor is the SAME constant for every `f` (an
    identity-product suffix composes onto ANY current state and returns it unchanged), so it does not
    affect the relative weighting across `f` and is applied separately, once, when the suffix is sampled.
    """
    weights = np.zeros(n_classes, dtype=np.float64)
    cache: dict[int, list[np.ndarray]] = {}
    for f in range(n_classes):
        tables = _build_taboo_tables(t, f, gens, mult, n_classes, ident)
        final = _step(tables[-1], gens, mult)  # one more (untaboo'd) step: mass landing exactly on f at step t
        weights[f] = final[f]
        cache[f] = tables
    return weights, cache


def _backward_sample(target: int, length: int, tables: list[np.ndarray], gens: tuple[int, ...], gen_inv: tuple[int, ...], mult: np.ndarray, rng: np.random.Generator) -> list[int]:
    """Backward-sample a length-`length` path ending at `target`, given cached forward tables `tables` (`tables[s]` = the length-s distribution the path must have come from).

    At each step (working backward from `length` to `1`), the previous state for generator `g` is
    `mult[gen_inv[g], target]` (the unique `y` with `mult[g,y]==target`); weight each candidate `g` by
    `tables[s-1][y]` and sample. Returns generator ALPHABET indices in forward (left-to-right) word order.
    """
    if length == 0:
        return []
    letters = [0] * length
    for s in range(length, 0, -1):
        prev = tables[s - 1]
        candidates = [(gi, int(mult[gen_inv[gi], target])) for gi in range(len(gens))]
        weights = np.array([prev[y] for _gi, y in candidates], dtype=np.float64)
        total = weights.sum()
        probs = weights / total
        choice = int(rng.choice(len(gens), p=probs))
        gi, y = candidates[choice]
        letters[s - 1] = gi
        target = y
    return letters


def sample_stratum(t_construct: int, n_samples: int, seed: int, grp: dict | None = None, word_len: int = L) -> dict:
    """Draws `n_samples` words UNIFORMLY from the realized stratum `{x : t*(x) == t_construct}` (design §3.1).

    Returns `{word_gen_ids (n, word_len), labels (n,), realized_t_star (n,)}`; every sample is
    assertion-checked (`realized_commitment` recomputed and compared to `t_construct`) before return.
    """
    grp = grp or _A5_GRP
    mult, gens, ident = grp["mult"], grp["generators"], grp["identity"]
    gen_inv = _generator_inverse_ids(grp)
    n_classes = len(grp["elements"])
    if not (0 <= t_construct <= word_len):
        raise ValueError(f"t_construct={t_construct} must be in [0, {word_len}]")
    m = word_len - t_construct

    rng = np.random.default_rng(seed)
    f_weights, taboo_cache = _first_hit_distribution(t_construct, gens, mult, n_classes, ident)
    total_weight = float(f_weights.sum())
    if total_weight <= 0.0:
        raise RuntimeError(f"no length-{t_construct} first-hit paths exist for this alphabet -- ladder/floor is broken (constraint 2)")
    f_probs = f_weights / total_weight

    suffix_tables = _build_unconstrained_tables(m, gens, mult, n_classes, ident) if m > 0 else []

    words = np.empty((n_samples, word_len), dtype=np.int64)
    labels = np.empty(n_samples, dtype=np.int64)
    for i in range(n_samples):
        f = int(rng.choice(n_classes, p=f_probs))
        prefix = _backward_sample(f, t_construct, taboo_cache[f], gens, gen_inv, mult, rng)
        suffix = _backward_sample(ident, m, suffix_tables, gens, gen_inv, mult, rng)
        words[i] = prefix + suffix
        labels[i] = f

    realized = np.array([realized_commitment(words[i], grp) for i in range(n_samples)], dtype=np.int64)
    mismatches = int((realized != t_construct).sum())
    if mismatches:
        bad = int(np.flatnonzero(realized != t_construct)[0])
        raise AssertionError(
            f"stratum t*={t_construct}: {mismatches}/{n_samples} words have a realized commitment point != constructed "
            f"(sampler/scan bug) -- e.g. row {bad}: word={words[bad].tolist()} realized={int(realized[bad])}"
        )

    return {"word_gen_ids": words, "labels": labels, "realized_t_star": realized, "t_construct": t_construct}


# ---------------------------------------------------------------------------
# Dataset assembly: one-hot encode, split train/val per stratum, pool across strata for anytime training.
# ---------------------------------------------------------------------------


def _onehot(word_gen_ids: np.ndarray, n_gen: int) -> np.ndarray:
    """`(n, word_len)` generator-index words -> `(n, word_len*n_gen)` flattened one-hot, float32."""
    n, word_len = word_gen_ids.shape
    return np.eye(n_gen, dtype=np.float32)[word_gen_ids].reshape(n, word_len * n_gen)


def make_selection_data(
    ladder: tuple[int, ...] = COMMIT_LADDER, n_per_stratum: int = N_PER_STRATUM_DEFAULT, seed: int = 0, train_frac: float = TRAIN_FRAC, grp: dict | None = None, word_len: int = L,
) -> dict:
    """Builds the full (train, val) selection dataset.

    `n_per_stratum` words per rung of `ladder`, plus each
    word's full per-position prefix-product table (`word_all_prefix_products`) — the anytime net's per-T
    training targets (root-cause fix: the RUNNING product, not the word's full product).
    """
    grp = grp or _A5_GRP
    n_gen = len(grp["generators"])
    per_stratum: dict[int, dict] = {}
    for t in ladder:
        drawn = sample_stratum(t, n_per_stratum, seed=seed * 1000 + t, grp=grp, word_len=word_len)
        n_tr = round(train_frac * n_per_stratum)
        x = _onehot(drawn["word_gen_ids"], n_gen)
        prefix_products = word_all_prefix_products(drawn["word_gen_ids"], grp)  # (n_per_stratum, word_len)
        per_stratum[t] = {
            "x_tr": x[:n_tr], "y_tr": drawn["labels"][:n_tr], "t_star_tr": drawn["realized_t_star"][:n_tr], "prefix_products_tr": prefix_products[:n_tr],
            "x_val": x[n_tr:], "y_val": drawn["labels"][n_tr:], "t_star_val": drawn["realized_t_star"][n_tr:], "prefix_products_val": prefix_products[n_tr:],
        }
    return {"per_stratum": per_stratum, "n_gen": n_gen, "n_classes": len(grp["elements"]), "ladder": list(ladder), "word_len": word_len}


def _pool_strata(data: dict, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Concatenates `x/y/t_star/prefix_products` across all strata for the given `split` ('tr' or 'val')."""
    xs, ys, ts, pps = [], [], [], []
    for d in data["per_stratum"].values():
        xs.append(d[f"x_{split}"])
        ys.append(d[f"y_{split}"])
        ts.append(d[f"t_star_{split}"])
        pps.append(d[f"prefix_products_{split}"])
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(ts), np.concatenate(pps)


# ---------------------------------------------------------------------------
# Anytime net (design §3.2, REVISED) — `depth_composition_toy.RecurrentComposer` reused verbatim: ONE
# shared block + ONE shared readout, fold length = whatever `x_flat` contains, so truncating `x_flat` to
# `T*n_gen` columns before the call IS "execute only T steps" (compute proportional to T, not a full
# fold with a truncated readout). Each exit T trains against the T-step RUNNING (prefix) product, not
# the word's full product — see module docstring's root-cause note.
# ---------------------------------------------------------------------------


def build_anytime_net(n_gen: int, n_classes: int, state_width: int = REC_STATE_WIDTH) -> RecurrentComposer:
    """Builds the anytime net shared across every T-exit.

    ONE shared recurrent block + ONE shared linear readout (`depth_composition_toy.RecurrentComposer`,
    reused verbatim, unchanged) — the readout is applied to whatever fold length `forward(x_flat)` is
    given, so the caller truncating `x_flat` to `T*n_gen` columns before the call selects the T-step exit.
    """
    return RecurrentComposer(state_width, n_gen, n_classes)


def train_anytime(
    data: dict, seed: int, device: str, lr: float = LR_DEFAULT, max_epochs: int = MAX_EPOCHS_DEFAULT, t_ladder: tuple[int, ...] = T_LADDER,
) -> tuple[RecurrentComposer, cvg.ConvergenceResult]:
    """Trains the anytime net on the POOLED (all-strata) data.

    For each T in `t_ladder`, the shared
    readout is trained against the T-step RUNNING (prefix) product (`prefix_products_{tr,val}` column
    `T-1`), NOT the word's full product — every exit's target is achievable from the state the net
    actually holds after T steps, so there is no impossible-target corruption (root-cause fix). Loss is
    the MEAN (not sum) cross-entropy over `t_ladder`; gradients are clipped to `GRAD_CLIP_MAX_NORM`
    before every optimizer step (root-cause fix #1, the L=10 GD-trainable wall). Convergence-gated on
    the mean-over-T val CE with the SAME hyperparameter convention `depth_graded_toy._train_mixed` uses
    (`CHECK_EVERY`/`PATIENCE`/`MIN_DELTA`), reimplemented here (not called directly) because the per-T
    target now varies within one net, which `_train_mixed`'s fixed-tensor-per-key API does not support.
    """
    n_gen, n_classes = data["n_gen"], data["n_classes"]
    x_tr, _y_tr, _t_star_tr, pp_tr = _pool_strata(data, "tr")
    x_val, _y_val, _t_star_val, pp_val = _pool_strata(data, "val")

    torch.manual_seed(seed)
    net = build_anytime_net(n_gen, n_classes)
    net.to(device)

    x_tr_t = torch.as_tensor(x_tr, dtype=torch.float32, device=device)
    x_val_t = torch.as_tensor(x_val, dtype=torch.float32, device=device)
    y_tr_by_t = {t: torch.as_tensor(pp_tr[:, t - 1], dtype=torch.long, device=device) for t in t_ladder}
    y_val_by_t = {t: torch.as_tensor(pp_val[:, t - 1], dtype=torch.long, device=device) for t in t_ladder}

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()

    def step_fn() -> None:
        opt.zero_grad()
        loss = sum(ce(net(x_tr_t[:, : t * n_gen]), y_tr_by_t[t]) for t in t_ladder) / len(t_ladder)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=GRAD_CLIP_MAX_NORM)
        opt.step()

    def val_fn() -> float:
        net.eval()
        with torch.no_grad():
            v = sum(ce(net(x_val_t[:, : t * n_gen]), y_val_by_t[t]).item() for t in t_ladder) / len(t_ladder)
        net.train()
        return v

    result = cvg.fit_to_convergence(net, step_fn, val_fn, max_epochs=max_epochs, check_every=CHECK_EVERY, patience=PATIENCE, min_delta=MIN_DELTA)
    return net, result


def per_input_accuracy_table(net: RecurrentComposer, x: np.ndarray, y: np.ndarray, n_gen: int, t_ladder: tuple[int, ...], device: str) -> dict[int, np.ndarray]:
    """Per-input correctness table `{T: (N,) bool correct}` on the given split.

    One forward pass per T (each truncated to `T*n_gen`
    columns). `y` is the word's FULL product (the deploy answer), never the T-step training target —
    correct iff T >= t*(x) (module docstring's root-cause note).
    """
    net.eval()
    x_t = torch.as_tensor(x, dtype=torch.float32, device=device)
    y_t = torch.as_tensor(y, dtype=torch.long, device=device)
    out = {}
    with torch.no_grad():
        for t in t_ladder:
            pred = net(x_t[:, : t * n_gen]).argmax(1)
            out[t] = (pred == y_t).cpu().numpy()
    return out


# ---------------------------------------------------------------------------
# S1/S2 bars (design §3.6 / plan D8b) — per-stratum, from the per-input accuracy table.
# ---------------------------------------------------------------------------


def check_gradedness_bars(
    acc_by_t: dict[int, np.ndarray], t_star: np.ndarray, ladder: tuple[int, ...], s2_ceiling: float,
    full_t: int = L, fit_acc: float = S1_FIT_ACC, knee_fraction: float = S2_KNEE_FRACTION,
) -> dict:
    """S1 (substrate) and S2 (THE MAKE-OR-BREAK knee), per stratum, from a per-input accuracy table."""
    s1_per_stratum = {}
    for t in ladder:
        mask = t_star == t
        if not mask.any():
            continue
        s1_per_stratum[t] = float(acc_by_t[full_t][mask].mean())
    s1_pass = bool(s1_per_stratum) and all(v >= fit_acc for v in s1_per_stratum.values())

    s2_per_stratum = {}
    for t in ladder:
        mask = t_star == t
        if not mask.any():
            continue
        acc_at_t_star = float(acc_by_t[t][mask].mean())
        acc_at_full = float(acc_by_t[full_t][mask].mean())
        knee_ratio = acc_at_t_star / acc_at_full if acc_at_full > 0 else float("nan")
        t_minus_2 = t - BAYES_G_FOR_CEILING
        acc_at_t_minus_2 = float(acc_by_t[t_minus_2][mask].mean()) if t_minus_2 in acc_by_t else None
        s2_per_stratum[t] = {
            "acc_at_t_star": acc_at_t_star,
            "acc_at_full_t": acc_at_full,
            "knee_ratio": knee_ratio,
            "knee_pass": bool(knee_ratio >= knee_fraction),
            "acc_at_t_star_minus_2": acc_at_t_minus_2,
            "s2_ceiling": s2_ceiling,
            "ceiling_pass": bool(acc_at_t_minus_2 is None or acc_at_t_minus_2 <= s2_ceiling),
        }
    s2_pass = bool(s2_per_stratum) and all(v["knee_pass"] and v["ceiling_pass"] for v in s2_per_stratum.values())

    return {"s1_per_stratum": s1_per_stratum, "s1_pass": s1_pass, "s2_per_stratum": s2_per_stratum, "s2_pass": s2_pass}


# ---------------------------------------------------------------------------
# Surface-baseline control (design §6 Option A / plan D8b S5) — COVARIATE, never a kill criterion.
# ---------------------------------------------------------------------------


def run_surface_probe(data: dict, seed: int, device: str, width: int = SURFACE_MLP_WIDTH, max_epochs: int = MAX_EPOCHS_DEFAULT) -> dict:
    """Shallow 1-hidden-layer MLP on the raw one-hot word -> predict the stratum (t* value); per-stratum balanced acc vs chance."""
    ladder = data["ladder"]
    ladder_index = {t: i for i, t in enumerate(ladder)}
    n_gen, word_len = data["n_gen"], data["word_len"]
    x_tr, _y_tr, t_star_tr, _pp_tr = _pool_strata(data, "tr")
    x_val, _y_val, t_star_val, _pp_val = _pool_strata(data, "val")
    strat_y_tr = np.array([ladder_index[int(t)] for t in t_star_tr], dtype=np.int64)
    strat_y_val = np.array([ladder_index[int(t)] for t in t_star_val], dtype=np.int64)

    torch.manual_seed(seed)
    n_strata = len(ladder)
    net = build_narrow_clf(1, width, word_len * n_gen, n_strata)
    probe_data = {"x_tr": x_tr, "y_tr": strat_y_tr, "x_val": x_val, "y_val": strat_y_val}
    result, train_acc, val_acc = _train_clf_generic(net, probe_data, device=device, max_epochs=max_epochs)

    net.eval()
    with torch.no_grad():
        x_val_t = torch.as_tensor(x_val, dtype=torch.float32, device=device)
        pred = net(x_val_t).argmax(1).cpu().numpy()
    per_stratum = {}
    for t, idx in ladder_index.items():
        mask = strat_y_val == idx
        per_stratum[t] = {"n": int(mask.sum()), "acc": float((pred[mask] == idx).mean()) if mask.any() else None}

    return {
        "chance": 1.0 / n_strata, "overall_train_acc": train_acc, "overall_val_acc": val_acc,
        "per_stratum": per_stratum, "trustworthy": bool(result.trustworthy), "convergence": result.summary(),
    }


# ---------------------------------------------------------------------------
# Router / deploy (design §3.5, plan D8b Step 3) — ADAPTS `sinc_width_experiment`'s metric-agnostic
# {level->err} SELECTOR-LABEL and DEPLOY-BAR machinery for classification error (1-acc) per input per T:
# `sw._cheapest_within_tolerance_labels` and `sw._deploy_bar_mse` are reused VERBATIM (both are generic
# over the err array / dict keys the caller passes; `_deploy_bar_mse` is fed a dict keyed by the actual
# T value, not a column index, so its `best_fixed_k` output IS the best-fixed-T value directly).
#
# The router MODEL itself is NOT reused: `capacity_ladder_k6._RouterMLP`/`_train_router`/`_route` are
# hardcoded to a SCALAR input (`x.reshape(-1, 1)`, `_RouterMLP.__init__`'s `dims = [1, *hidden]`) because
# every width toy's router reads a single continuous x-coordinate. Our router reads the raw one-hot WORD
# (a `word_len*n_gen`-dim vector, per design §3.5 S4 "no-oracle" input), so calling `ck6._train_router`
# directly is a genuine shape mismatch (verified: it raises inside `nnf.cross_entropy`, batch dimension
# corrupted by `.reshape(-1, 1)` flattening the whole 2-D array). No existing router in the codebase
# accepts a vector input (grep across `automl_package/examples/*.py` turned up only scalar-input
# routers), so `_VectorRouterMLP`/`_train_vector_router`/`_route_vector` below are a small, deliberately
# minimal analogue of `ck6._RouterMLP`'s architecture and training loop (same hidden sizes, same Adam+CE
# objective), parametrized by `in_dim` instead of hardcoding `1`.
# ---------------------------------------------------------------------------


class _VectorRouterMLP(nn.Module):
    """`x (in_dim vector) -> logits over n_cols columns`, hidden `(32, 32)` + ReLU (mirrors `ck6._RouterMLP`'s architecture, generalized to a vector input)."""

    def __init__(self, in_dim: int, n_cols: int, hidden: tuple[int, ...] = ck6.HIDDEN) -> None:
        super().__init__()
        dims = [in_dim, *hidden]
        layers: list[nn.Module] = []
        for d_in, d_out in itertools.pairwise(dims):
            layers += [nn.Linear(d_in, d_out), nn.ReLU()]
        layers.append(nn.Linear(dims[-1], n_cols))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`(N, in_dim) -> (N, n_cols)` logits."""
        return self.net(x)


def _train_vector_router(
    x: np.ndarray, n_cols: int, device: str, hard_labels: np.ndarray, seed: int, n_epochs: int = ck6.N_EPOCHS, lr: float = ck6.LR, hidden: tuple[int, ...] = ck6.HIDDEN,
) -> _VectorRouterMLP:
    """Hard-label CE training loop for `_VectorRouterMLP` (same objective as `ck6._train_router`'s hard-label branch)."""
    torch.manual_seed(seed)
    model = _VectorRouterMLP(x.shape[1], n_cols, hidden=hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    x_t = torch.as_tensor(x, dtype=torch.float32, device=device)
    y_t = torch.as_tensor(hard_labels, dtype=torch.long, device=device)
    model.train()
    for _ in range(n_epochs):
        opt.zero_grad()
        loss = nnf.cross_entropy(model(x_t), y_t)
        loss.backward()
        opt.step()
    model.eval()
    return model


def _route_vector(model: _VectorRouterMLP, x: np.ndarray, device: str) -> np.ndarray:
    """Argmax-routed column index per row of the vector input `x` (mirrors `ck6._route`)."""
    with torch.no_grad():
        x_t = torch.as_tensor(x, dtype=torch.float32, device=device)
        logits = model(x_t)
    return logits.argmax(dim=1).cpu().numpy()


def build_error_table(net: RecurrentComposer, x: np.ndarray, y: np.ndarray, n_gen: int, t_ladder: tuple[int, ...], device: str) -> dict[int, np.ndarray]:
    """`{T: (N,) 0/1 error}` — the classification twin of `sinc_width_experiment`'s per-width MSE table."""
    acc = per_input_accuracy_table(net, x, y, n_gen, t_ladder, device)
    return {t: (~correct).astype(np.float64) for t, correct in acc.items()}


def fit_router(err_tr: dict[int, np.ndarray], x_tr: np.ndarray, seed: int, device: str, t_ladder: tuple[int, ...] = T_LADDER, delta_tie: float = DELTA_TIE) -> _VectorRouterMLP:
    """Distills the PRIMARY hard-label cheapest-within-tolerance router (`sw._cheapest_within_tolerance_labels`, reused verbatim) over the T-ladder's error table."""
    t_sorted = tuple(sorted(t_ladder))
    err_matrix = np.stack([err_tr[t] for t in t_sorted], axis=1)  # (N, n_cols), ascending by T
    labels = sw._cheapest_within_tolerance_labels(err_matrix, delta_tie)
    return _train_vector_router(x_tr, len(t_sorted), device, labels, seed)


def deploy_battery(
    router: _VectorRouterMLP, err_test: dict[int, np.ndarray], x_test: np.ndarray, t_star_test: np.ndarray, device: str, t_ladder: tuple[int, ...] = T_LADDER,
) -> dict:
    """Hard-pick routing + the deploy bar (`sw._deploy_bar_mse`, reused verbatim, keyed by actual T not column index).

    Augments the shared deploy bar with the S3 router-QUALITY check: the Spearman rank correlation between
    the router's per-input deployed T and the realized commitment point t*(x). A faithful per-input depth
    dial should route deeper for inputs that commit later, so `router_spearman >= S3_ROUTER_SPEARMAN_MIN`.
    A collapsed router (constant T) yields a NaN correlation, which fails the bar.
    """
    t_sorted = tuple(sorted(t_ladder))
    col_idx = _route_vector(router, x_test, device)
    executed_t = np.array([t_sorted[c] for c in col_idx], dtype=np.float64)
    rows = np.arange(col_idx.shape[0])
    err_matrix = np.stack([err_test[t] for t in t_sorted], axis=1)
    err_hardpick = err_matrix[rows, col_idx]
    bar = sw._deploy_bar_mse(err_hardpick, executed_t, {t: err_test[t] for t in t_sorted})
    rho = float(spearmanr(executed_t, t_star_test.astype(np.float64)).statistic)
    router_spearman_pass = bool(np.isfinite(rho) and rho >= S3_ROUTER_SPEARMAN_MIN)
    bar["router_spearman"] = rho
    bar["router_spearman_pass"] = router_spearman_pass
    bar["s3_pass"] = bool(bar["deploy_pass"] and router_spearman_pass)
    return bar


# ---------------------------------------------------------------------------
# Selftest — FAST (seconds), no real training: A5 axioms + generation (via depth_composition_toy, run
# separately), sampler assertion, hand-constructed realized-t* cases, net forward-shape checks.
# ---------------------------------------------------------------------------


def _check_hand_constructed_realized_t_star() -> bool:
    """Hand-verified realized-t* cases (design §3.1's "no contiguity assumption" is exactly what these exercise)."""
    ok = True
    cases = [
        # (word as generator-alphabet indices, expected realized t*, label). Alphabet = A5 involutions
        # (order 2), so identity folds happen via same-letter PAIRS (g*g = identity), not order-5 wraps.
        ((0, 1, 2, 3, 0, 1), 6, "plain: no early commitment, full length used"),
        ((0, 1, 2, 2), 2, "identity-tail: prefix [0,1] then involution pair [2,2] (g*g=identity) folds -> commits at t=2"),
        ((0, 1, 3, 3, 2, 2), 2, "non-contiguous tail folds: [3,3] then [2,2] each fold to identity -> commits at t=2 (the 'no contiguity' property)"),
    ]
    for word, expected, label in cases:
        got = realized_commitment(word)
        case_ok = got == expected
        print(f"[selection selftest] realized_commitment {label}: word={word} expected={expected} got={got}  {'PASS' if case_ok else 'FAIL'}")
        ok = ok and case_ok
    return ok


def _check_prefix_products() -> bool:
    """Selftest that `word_all_prefix_products` matches the certified scalar `word_prefix_products`.

    `word_all_prefix_products` (vectorized, feeds the anytime net's per-T training targets) must agree
    with the already-certified `word_prefix_products` (scalar, backs `realized_commitment`) on hand-built
    words — this is exactly the quantity the root-cause fix trains against, so a mismatch here would
    silently corrupt every exit's label.
    """
    ok = True
    words = [(0, 0, 0, 0, 0, 0, 0, 0), (0, 0, 1, 2, 2, 2, 2, 2)]  # same length so they batch (reuses hand cases above)
    batch = np.array(words, dtype=np.int64)
    batch_pp = word_all_prefix_products(batch)
    for i, word in enumerate(words):
        expected = word_prefix_products(word)
        got = batch_pp[i].tolist()
        case_ok = got == expected
        print(f"[selection selftest] prefix products word={word}: expected={expected} got={got}  {'PASS' if case_ok else 'FAIL'}")
        ok = ok and case_ok
    return ok


def _check_sampler_assertion(n_samples: int = 200) -> bool:
    """Small-scale `sample_stratum` runs for every ladder rung.

    The function itself raises on any realized/constructed mismatch, so reaching here without an
    exception already proves the assertion held; this additionally re-checks independently.
    """
    ok = True
    for t in COMMIT_LADDER:
        drawn = sample_stratum(t, n_samples, seed=0)
        recomputed = np.array([realized_commitment(drawn["word_gen_ids"][i]) for i in range(n_samples)])
        case_ok = bool(np.array_equal(recomputed, drawn["realized_t_star"])) and bool((recomputed == t).all())
        print(f"[selection selftest] sample_stratum t={t}: n={n_samples} all_realized_eq_construct={case_ok}  {'PASS' if case_ok else 'FAIL'}")
        ok = ok and case_ok
    return ok


def _check_net_forward_shapes() -> bool:
    """Anytime net (shared block + ONE shared readout) forward-shape check at two T values (small dummy net, no training)."""
    ok = True
    n_gen, n_classes = 4, 10
    net = build_anytime_net(n_gen, n_classes, state_width=8)
    for t in (2, 6):
        x = torch.zeros(3, t * n_gen)
        out = net(x)
        shape_ok = tuple(out.shape) == (3, n_classes)
        print(f"[selection selftest] anytime net forward T={t}: out {tuple(out.shape)} (expect (3,{n_classes}))  {'PASS' if shape_ok else 'FAIL'}")
        ok = ok and shape_ok
    return ok


def _check_arithmetic_sane() -> bool:
    """Sanity on the arithmetic table.

    Support is non-decreasing, reaches all 60 classes, and there are no identity words below
    `MIN_RELATION_LENGTH` (2 for the involution alphabet: g*g is the shortest identity relation).
    """
    table = compute_arithmetic_table(max_len=8)
    rows = table["rows"]
    n_classes = table["n_classes"]
    supports = [r["support"] for r in rows]
    monotone_ok = all(b >= a for a, b in itertools.pairwise(supports))
    reaches_all_ok = supports[-1] == n_classes
    no_short_relations_ok = all(r["identity_count"] == 0 for r in rows if r["t"] < MIN_RELATION_LENGTH)
    ok = monotone_ok and reaches_all_ok and no_short_relations_ok
    print(
        f"[selection selftest] arithmetic sanity: support_monotone={monotone_ok} reaches_all_60={reaches_all_ok} "
        f"no_relations_below_{MIN_RELATION_LENGTH}={no_short_relations_ok}  {'PASS' if ok else 'FAIL'}"
    )
    return ok


def run_selftest() -> bool:
    """All fast (seconds), no-real-training checks: arithmetic sanity, hand-cases, prefix products, sampler assertion, net shapes."""
    ok = True
    ok = _check_arithmetic_sane() and ok
    ok = _check_hand_constructed_realized_t_star() and ok
    ok = _check_prefix_products() and ok
    ok = _check_sampler_assertion() and ok
    ok = _check_net_forward_shapes() and ok
    print(f"[selection selftest] {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _print_arithmetic(table: dict, ceiling: float, ladder_check: dict) -> None:
    print(f"[selection arithmetic] A5 involution alphabet: n_classes={table['n_classes']} n_gen={table['n_gen']}")
    print(f"{'t':>2} {'support':>7} {'id_count':>12} {'bayes_acc':>10} {'tv_to_uniform':>14}")
    for r in table["rows"]:
        print(f"{r['t']:2d} {r['support']:7d} {r['identity_count']:12d} {r['bayes_acc']:10.4f} {r['tv_to_uniform']:14.4f}")
    print(f"S2_CEILING_A5={ceiling:.4f}  (Bayes(g={BAYES_G_FOR_CEILING}) + {S2_CEILING_MARGIN_PP:.2f})")
    print(f"ladder={ladder_check['ladder']} coverage_safe={ladder_check['coverage_safe']}  per_t={ladder_check['per_t']}")


def main() -> None:
    """Parses args and runs `--selftest` or one `--probe {arithmetic,gradedness,surface,deploy}`."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--selftest", action="store_true", help="Fast (seconds), no-real-training checks.")
    parser.add_argument("--probe", type=str, choices=["arithmetic", "gradedness", "surface", "deploy"], default=None, help="Which probe to run.")
    parser.add_argument("--n-per-stratum", type=int, default=N_PER_STRATUM_DEFAULT, help="Words per commitment stratum (default 40000; use a small value for a smoke run).")
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS_DEFAULT, help="Max epochs for the anytime-net / surface-probe convergence gate.")
    parser.add_argument("--lr", type=float, default=LR_DEFAULT, help="Adam learning rate for the anytime net.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed.")
    parser.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR, help="Directory for probe JSON output.")
    args = parser.parse_args()

    if args.selftest:
        sys.exit(0 if run_selftest() else 1)

    if args.probe == "arithmetic":
        table = compute_arithmetic_table()
        ceiling = s2_ceiling_a5(table)
        ladder_check = confirm_ladder(table)
        _print_arithmetic(table, ceiling, ladder_check)
        os.makedirs(args.out_dir, exist_ok=True)
        path = os.path.join(args.out_dir, "depth_selection_arithmetic_a5.json")
        with open(path, "w") as f:
            json.dump({"table": table, "s2_ceiling_a5": ceiling, "ladder_check": ladder_check}, f, indent=2)
        print(f"[selection] wrote {path}")
        sys.exit(0)

    if args.probe == "gradedness":
        device = os.environ.get("AUTOML_DEVICE", "cpu")
        data = make_selection_data(n_per_stratum=args.n_per_stratum, seed=args.seed)
        net, result = train_anytime(data, args.seed, device, lr=args.lr, max_epochs=args.max_epochs)
        x_val, y_val, t_star_val, _pp_val = _pool_strata(data, "val")
        acc_by_t = per_input_accuracy_table(net, x_val, y_val, data["n_gen"], T_LADDER, device)
        arithmetic = compute_arithmetic_table()
        ceiling = s2_ceiling_a5(arithmetic)
        bars = check_gradedness_bars(acc_by_t, t_star_val, tuple(data["ladder"]), ceiling)
        print(f"[selection gradedness] trustworthy={result.trustworthy} diverged={result.diverged} hit_cap={result.hit_cap}")
        if not result.trustworthy:
            print("[selection gradedness] NOT CONCLUSIVE -- training did not converge trustworthily; bars below are informational only.")
        print(f"  S1 (per stratum, full-T acc >= {S1_FIT_ACC}): {bars['s1_per_stratum']}  pass={bars['s1_pass']}")
        print(f"  S2 (make-or-break knee): {bars['s2_per_stratum']}  pass={bars['s2_pass']}")
        os.makedirs(args.out_dir, exist_ok=True)
        path = os.path.join(args.out_dir, f"depth_selection_gradedness_seed{args.seed}.json")
        with open(path, "w") as f:
            json.dump({"seed": args.seed, "n_per_stratum": args.n_per_stratum, "lr": args.lr, "convergence": result.summary(), "s2_ceiling_a5": ceiling, "bars": bars}, f, indent=2)
        print(f"[selection] wrote {path}")
        sys.exit(0)

    if args.probe == "surface":
        device = os.environ.get("AUTOML_DEVICE", "cpu")
        data = make_selection_data(n_per_stratum=args.n_per_stratum, seed=args.seed)
        surface = run_surface_probe(data, args.seed, device, max_epochs=args.max_epochs)
        print(f"[selection surface] chance={surface['chance']:.4f} overall_val_acc={surface['overall_val_acc']:.4f} trustworthy={surface['trustworthy']}")
        print(f"  per_stratum={surface['per_stratum']}")
        os.makedirs(args.out_dir, exist_ok=True)
        path = os.path.join(args.out_dir, f"depth_selection_surface_seed{args.seed}.json")
        with open(path, "w") as f:
            json.dump(surface, f, indent=2)
        print(f"[selection] wrote {path}")
        sys.exit(0)

    if args.probe == "deploy":
        device = os.environ.get("AUTOML_DEVICE", "cpu")
        data = make_selection_data(n_per_stratum=args.n_per_stratum, seed=args.seed)
        net, result = train_anytime(data, args.seed, device, lr=args.lr, max_epochs=args.max_epochs)
        x_tr, y_tr, _t_star_tr, _pp_tr = _pool_strata(data, "tr")
        x_val, y_val, t_star_val, _pp_val = _pool_strata(data, "val")
        err_tr = build_error_table(net, x_tr, y_tr, data["n_gen"], T_LADDER, device)
        err_val = build_error_table(net, x_val, y_val, data["n_gen"], T_LADDER, device)
        router = fit_router(err_tr, x_tr, args.seed, device)
        deploy = deploy_battery(router, err_val, x_val, t_star_val, device)
        # S4 no-oracle documentation (design D8b S4): the router sees the RAW one-hot word only — never
        # t*(x), the stratum, or any length/depth label — and is distilled from the held-out per-T error
        # table, not from an oracle. Recorded in the JSON so the no-oracle claim is self-evidencing.
        router_features = {
            "input": "flattened raw one-hot word",
            "in_dim": int(x_val.shape[1]),
            "word_len_x_n_gen": f"{data['word_len']}x{data['n_gen']}",
            "n_route_columns": len(sorted(T_LADDER)),
            "label_source": "cheapest-within-tolerance over the held-out per-T error table (sw._cheapest_within_tolerance_labels); NO t*/stratum/length oracle",
            "oracle_depth_labels_used": False,
        }
        print(f"[selection deploy] trustworthy={result.trustworthy} deploy={deploy}")
        os.makedirs(args.out_dir, exist_ok=True)
        path = os.path.join(args.out_dir, f"depth_selection_deploy_seed{args.seed}.json")
        with open(path, "w") as f:
            json.dump({"seed": args.seed, "convergence": result.summary(), "router_features": router_features, "deploy": deploy}, f, indent=2)
        print(f"[selection] wrote {path}")
        sys.exit(0)

    parser.print_help()


if __name__ == "__main__":
    main()
