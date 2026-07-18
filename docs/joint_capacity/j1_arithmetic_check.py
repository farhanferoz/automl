"""J-1 (parallel A5 tracks) arithmetic verification — data well-posedness for the joint width+depth toy.

Reuses the certified A5 machinery from `depth_selection_toy` (no reinvention). Verifies, on the DATA only
(no net): (a) width-demand `a` (active tracks) is uncorrelated with depth-demand `t_star`; (b) per-track
class balance; (c) the joint Bayes-accuracy ceiling per (read-width, read-depth) cell for the depth axis;
(d) no starvation per (a, t_star) cell; (e) `t_star <= 10` (the A5 GD-trainable wall, MOD-1). The WIDTH
floor is information-theoretic (an analytic lower bound on state units to hold `a` independent A5
elements) — the actual width-accuracy curve is a PILOT measurement, not a data property (MOD-2).
"""
from __future__ import annotations

import math
import sys

import numpy as np

_EX = "/home/ff235/dev/MLResearch/automl/automl_package/examples"
if _EX not in sys.path:
    sys.path.insert(0, _EX)

import depth_selection_toy as dst  # noqa: E402

K_MAX = 4                      # width dial: number of parallel track slots
T_STAR_LADDER = (6, 8, 10)     # depth dial: shared commitment point of the active tracks (<= L)
L = 10
N_CLASSES = dst._A5_N_CLASSES  # 60
NOOP = 4                       # padding symbol for inactive tracks (the 4 involutions are indices 0..3)


def build_input(a_active: int, t_star: int, rng: np.random.Generator) -> dict:
    """One J-1 input: `a_active` active tracks + `K_MAX - a_active` inactive no-op tracks.

    Each active track is a length-L A5 word with realized commitment `t_star` (distinct products);
    returns the per-track words/labels/commitments plus the input's realized (width, depth) demand.
    """
    active_slots = sorted(rng.choice(K_MAX, size=a_active, replace=False).tolist())
    tracks = np.full((K_MAX, L), NOOP, dtype=np.int64)
    labels = np.full(K_MAX, dst._A5_IDENTITY, dtype=np.int64)
    commit = np.zeros(K_MAX, dtype=np.int64)  # 0 = inactive (no-op, product = identity)
    for slot in active_slots:
        drawn = dst.sample_stratum(t_star, 1, seed=int(rng.integers(1 << 30)))
        tracks[slot] = drawn["word_gen_ids"][0]
        labels[slot] = drawn["labels"][0]
        commit[slot] = t_star
    depth_demand = int(commit[active_slots].max())  # == t_star by construction
    return {"tracks": tracks, "labels": labels, "commit": commit, "a_active": a_active, "depth_demand": depth_demand}


def check_orthogonality(n: int = 4000, seed: int = 0) -> dict:
    """Draw `a` ~ U{1..K_MAX} and `t_star` ~ U(T_STAR_LADDER) INDEPENDENTLY.

    Confirms the realized (width-demand, depth-demand) are uncorrelated and every (a, t_star) cell is
    populated (the max-of-`a`-draws skew is avoided by giving all active tracks the SAME t_star).
    """
    rng = np.random.default_rng(seed)
    a_list, t_list = [], []
    cell = {(a, t): 0 for a in range(1, K_MAX + 1) for t in T_STAR_LADDER}
    for _ in range(n):
        a_active = int(rng.integers(1, K_MAX + 1))
        t_star = int(rng.choice(T_STAR_LADDER))
        x = build_input(a_active, t_star, rng)
        assert x["depth_demand"] == t_star  # orthogonality precondition: depth-demand is exactly t_star, not max-skewed
        a_list.append(x["a_active"])
        t_list.append(x["depth_demand"])
        cell[(a_active, t_star)] += 1
    corr = float(np.corrcoef(np.array(a_list), np.array(t_list))[0, 1])
    min_cell = min(cell.values())
    return {"n": n, "pearson_a_vs_t": corr, "min_cell_count": min_cell, "n_cells": len(cell), "cells_all_populated": bool(min_cell > 0)}


def joint_bayes_ceiling() -> dict:
    """Depth-axis Bayes ceiling for the JOINT (all-active-tracks-correct) accuracy, per (read-T, t_star).

    A track with `g = t_star - read_t` unread letters has per-track Bayes = `table.bayes_acc(g)` (reused,
    exact); the joint all-`a`-correct ceiling at unlimited width is `bayes(g) ** a`. At read_t >= t_star it
    is 1.0 (width permitting).
    """
    table = dst.compute_arithmetic_table(max_len=L)
    bayes_by_g = {r["t"]: r["bayes_acc"] for r in table["rows"]}
    bayes_by_g[0] = 1.0
    out = {}
    for t_star in T_STAR_LADDER:
        for read_t in range(2, L + 1, 2):
            g = max(0, t_star - read_t)
            per_track = bayes_by_g[g]
            out[f"t*={t_star},readT={read_t}"] = {"g": g, "per_track_bayes": round(per_track, 4),
                                                  "joint_a1": round(per_track, 4), "joint_a4": round(per_track ** 4, 4)}
    return out


def width_info_floor() -> dict:
    """Analytic (soft) width floor: bits needed to hold `a` independent A5 elements = `a * log2(60)`.

    An information LOWER bound on state capacity, NOT a Bayes-accuracy — the realized width-accuracy curve
    is a PILOT measurement (MOD-2: the classification joint toy has no analytic sigma^2 floor like the
    width-MSE toy; its width 'floor' is capacity, measured empirically).
    """
    bits_per_elem = math.log2(N_CLASSES)
    return {a: {"bits_needed": round(a * bits_per_elem, 2)} for a in range(1, K_MAX + 1)}


def class_balance(t_star: int = 8, n: int = 3000, seed: int = 1) -> dict:
    """Per-track label balance at one stratum (reuses `sample_stratum`'s own uniform-first-hit draw)."""
    drawn = dst.sample_stratum(t_star, n, seed=seed)
    counts = np.bincount(drawn["labels"], minlength=N_CLASSES)
    nz = counts[counts > 0]
    return {"t_star": t_star, "n": n, "classes_hit": int((counts > 0).sum()), "min_count": int(nz.min()), "max_count": int(nz.max()), "mean": round(float(nz.mean()), 1)}


def main() -> None:
    """Run all five checks and print the table quoted in `joint_toy_design.md` §5."""
    print("=== J-1 arithmetic verification (parallel A5 tracks) ===")
    print(f"K_MAX(width dial)={K_MAX}  t*_ladder(depth dial)={T_STAR_LADDER}  L={L}  n_classes={N_CLASSES}")
    ortho = check_orthogonality()
    print("\n[a] orthogonality (a drawn indep of t*):")
    print(f"    pearson(a, depth_demand) = {ortho['pearson_a_vs_t']:+.4f}  (target ~0)")
    print(f"    all {ortho['n_cells']} (a,t*) cells populated = {ortho['cells_all_populated']} (min cell {ortho['min_cell_count']})")
    cb = class_balance()
    print("\n[b] per-track class balance (stratum t*=8):")
    print(f"    classes_hit={cb['classes_hit']}/60  min/mean/max per-class count = {cb['min_count']}/{cb['mean']}/{cb['max_count']}")
    print("\n[c] joint depth Bayes ceiling (per_track = bayes(g); joint_a4 = bayes(g)^4):")
    for key, v in joint_bayes_ceiling().items():
        print(f"    {key:20s} g={v['g']} per_track={v['per_track_bayes']:.4f}  joint_a1={v['joint_a1']:.4f} joint_a4={v['joint_a4']:.4f}")
    print("\n[d] width info-floor (bits to hold a elements = a*log2(60)):")
    for a, v in width_info_floor().items():
        print(f"    a={a}: {v['bits_needed']} bits")
    print("\n[e] depth wall: all t* <=", L, "->", all(t <= L for t in T_STAR_LADDER))


if __name__ == "__main__":
    main()
