"""WSEL-19 v2 multi-feature slice: the pre-registered A1 verdict aggregation.

Reads every untagged per-cell JSON in this directory, applies the redesign spec's §5.5
aggregation rule (`shared/wsel19-toy-redesign.md` amendment A1: verdict weight only for cells
with regime_visible & fit ok & routing ok; a (d, geometry) level is DECIDED only if >= 2 of 3
seeds yield regime-visible, non-void triples — the survivor-bias closure), prints the
ruling-relevant readouts, and writes the aggregate ledger `frozen_mf.json` next to the cells.
Like `warp_trace.py`, this is a promoted analysis script living beside the artifacts it reads.

Usage: ~/dev/.venv/bin/python automl_package/examples/capacity_ladder_results/WSEL19/mf_aggregate.py
"""

import glob
import json
import os
import re
from collections import defaultdict

RESULTS = os.path.dirname(os.path.abspath(__file__))
MIN_SEEDS = 2

cells = []
for path in sorted(glob.glob(os.path.join(RESULTS, "wsel19_mf_d*.json"))):
    if re.search(r"_(authsmoke|f6fallback)\.json$", path):
        continue
    with open(path) as f:
        cells.append(json.load(f))

print(f"cells loaded: {len(cells)} (expected 432)")

# --- per-triple validity (shared by all 8 backend x mode cells at a (d, geometry, seed, n_sel)...
# regime/fit/routing are sweep-level: identical across backend/mode AND n_sel within a triple).
triple_status = {}
for c in cells:
    key = (c["d"], c["geometry"], c["seed"])
    v = c["validity_checks"]
    ok = v["regime_visible"] and v["fit_status"] == "ok" and v["routing_status"] == "ok"
    prev = triple_status.get(key)
    entry = (ok, v["regime_visible"], v["fit_status"], v["routing_status"], round(v["ratio_to_noise_floor"], 3),
             v["practical_floor_gap"], c["n_train"], c["n_train_fallback_applied"])
    if prev is None:
        triple_status[key] = entry
    elif prev != entry:
        print(f"WARNING: inconsistent validity within triple {key}: {prev} vs {entry}")

print("\n=== triple validity (A1 units) ===")
level_alive = {}
for d in (2, 8, 32):
    for geo in ("axis", "oblique"):
        seeds_ok = [s for s in (0, 1, 2) if triple_status.get((d, geo, s), (False,))[0]]
        decided = len(seeds_ok) >= MIN_SEEDS
        level_alive[(d, geo)] = (decided, seeds_ok)
        for s in (0, 1, 2):
            st = triple_status.get((d, geo, s))
            desc = f"regime={st[1]} fit={st[2]} ratio={st[4]} gap={st[5]} n_train={st[6]} fb={st[7]}" if st else "MISSING"
            print(f"  d={d:2d} {geo:8s} seed {s}: {'OK  ' if st and st[0] else 'VOID'} {desc}")
        print(f"  -> level (d={d}, {geo}): {'DECIDED' if decided else 'OPEN'} ({len(seeds_ok)}/3 verdict-weight seeds)")

# --- backend x mode x n_sel aggregates over verdict-weight cells, per decided level ---
print("\n=== backend comparison per DECIDED level (mean routed_held_out_quality over verdict-weight seeds; deployed flops) ===")
for (d, geo), (decided, seeds_ok) in level_alive.items():
    if not decided:
        continue
    print(f"\n-- d={d} {geo} (seeds {seeds_ok}) --")
    groups = defaultdict(list)
    for c in cells:
        if c["d"] == d and c["geometry"] == geo and c["seed"] in seeds_ok:
            groups[(c["backend"], c["mode"], c["n_sel"])].append(c)
    header = f"{'backend':10s} {'mode':6s} " + " ".join(f"{'q@'+str(n):>12s} {'fl@'+str(n):>8s}" for n in (75, 300, 1200))
    print(header)
    for backend in ("frozen_mlp", "rule_mlp", "xgboost", "constant"):
        for mode in ("hard", "blend"):
            row = f"{backend:10s} {mode:6s} "
            for n_sel in (75, 300, 1200):
                cs = groups.get((backend, mode, n_sel), [])
                if cs:
                    q = sum(x["routed_held_out_quality"] for x in cs) / len(cs)
                    fl = sum(x["mean_deployed_flops"] for x in cs) / len(cs)
                    row += f"{q:12.6f} {fl:8.1f} "
                else:
                    row += f"{'--':>12s} {'--':>8s} "
            print(row)
    # oracle agreement (hard mode only)
    oa = {}
    for backend in ("frozen_mlp", "rule_mlp", "xgboost", "constant"):
        vals = [x["oracle_agreement"] for x in cells
                if x["d"] == d and x["geometry"] == geo and x["seed"] in seeds_ok
                and x["backend"] == backend and x["mode"] == "hard" and x["oracle_agreement"] is not None]
        oa[backend] = round(sum(vals) / len(vals), 3) if vals else None
    print(f"oracle_agreement (hard): {oa}")

print("\n=== rulings readout ===")
for d in (2, 8, 32):
    axis_dec = level_alive.get((d, "axis"), (False, []))[0]
    obl_dec = level_alive.get((d, "oblique"), (False, []))[0]
    print(f"d={d}: axis {'DECIDED' if axis_dec else 'OPEN'}, oblique {'DECIDED' if obl_dec else 'OPEN'}")

# --- the frozen aggregate ledger (the citable source for width.md's verdict leaves) ---
out = {
    "aggregation_rule": "redesign spec A1: verdict weight iff regime_visible & fit ok & routing ok; level DECIDED iff >= 2/3 seeds verdict-weight",
    "n_cells": len(cells),
    "triples": {f"d{d}_{geo}_seed{s}": {"verdict_weight": st[0], "regime_visible": st[1], "fit_status": st[2],
                                        "routing_status": st[3], "ratio_to_noise_floor": st[4], "practical_floor_gap": st[5],
                                        "n_train": st[6], "n_train_fallback_applied": st[7]}
                for (d, geo, s), st in sorted(triple_status.items())},
    "levels": {f"d{d}_{geo}": {"decided": dec, "verdict_weight_seeds": seeds_ok} for (d, geo), (dec, seeds_ok) in level_alive.items()},
    "per_group": {},
}
for (d, geo), (decided, seeds_ok) in level_alive.items():
    if not decided:
        continue
    for c in cells:
        if c["d"] == d and c["geometry"] == geo and c["seed"] in seeds_ok:
            key = f"d{d}:{geo}:{c['backend']}:{c['mode']}:{c['n_sel']}"
            g = out["per_group"].setdefault(key, {"routed_held_out_quality": [], "mean_deployed_flops": [], "oracle_agreement": []})
            g["routed_held_out_quality"].append(c["routed_held_out_quality"])
            g["mean_deployed_flops"].append(c["mean_deployed_flops"])
            if c["oracle_agreement"] is not None:
                g["oracle_agreement"].append(c["oracle_agreement"])
for key, g in out["per_group"].items():
    out["per_group"][key] = {
        "routed_held_out_quality_mean": sum(g["routed_held_out_quality"]) / len(g["routed_held_out_quality"]),
        "mean_deployed_flops_mean": sum(g["mean_deployed_flops"]) / len(g["mean_deployed_flops"]),
        "oracle_agreement_mean": (sum(g["oracle_agreement"]) / len(g["oracle_agreement"])) if g["oracle_agreement"] else None,
        "n_seeds": len(g["routed_held_out_quality"]),
    }
ledger_path = os.path.join(RESULTS, "frozen_mf.json")
with open(ledger_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nwrote {ledger_path}")
