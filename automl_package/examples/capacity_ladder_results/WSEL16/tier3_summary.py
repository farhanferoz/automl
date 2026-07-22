"""Aggregate the WSEL-16 stage-3 tier-3 coverage grid (36 cells, b_heads single-arm).

The driver's own ``summarize()`` predates tier 3 and probes unsuffixed filenames, so this
root-side aggregator is the tier-3 ledger producer (width.md WSEL-16 stage-3 sign-off ruling,
2026-07-22). Deterministic: reads only the landed cell JSONs beside it; timestamp is keyed off
the newest input's mtime so reruns are byte-identical.

Coverage readouts per (n_train, sigma): full-width held-out MSE per seed, ratio to the
generator-true noise floor sigma^2 (the §3.7 fixed-at-truth constant — for tier-3 cells the
driver trains the fixed-sigma weighted objective, so sigma^2 is the exact irreducible floor),
and the convergence-gate integrity counts (trustworthy widths, hit_cap, diverged).
"""

import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "tier3_summary.json")


def main() -> None:
    """Aggregate all landed tier-3 cell JSONs into tier3_summary.json and print the table."""
    cells = sorted(glob.glob(os.path.join(HERE, "wsel16_b_heads_tier3_seed*_n*_s*.json")))
    rows = []
    for path in cells:
        with open(path) as fh:
            d = json.load(fh)
        conv = d["convergence"]
        rows.append(
            {
                "n_train": d["n_train"],
                "sigma": d["sigma"],
                "seed": d["seed"],
                "full_width_held_out_mse": d["full_width_held_out_mse"],
                "mse_over_noise_floor": d["full_width_held_out_mse"] / (d["sigma"] ** 2),
                "n_widths_trustworthy": d["n_widths_trustworthy"],
                "hit_cap": d["hit_cap"],
                "n_diverged": sum(c["diverged"] for c in conv.values()),
                "file": os.path.basename(path),
            }
        )
    rows.sort(key=lambda r: (r["n_train"], r["sigma"], r["seed"]))
    newest_mtime = max(os.path.getmtime(p) for p in cells)
    summary = {
        "n_cells": len(rows),
        "all_hit_cap_false": all(not r["hit_cap"] for r in rows),
        "min_widths_trustworthy": min(r["n_widths_trustworthy"] for r in rows),
        "n_cells_any_diverged": sum(1 for r in rows if r["n_diverged"] > 0),
        "worst_mse_over_noise_floor": max(r["mse_over_noise_floor"] for r in rows),
        "cells": rows,
        "provenance": {"input_mtime_newest": newest_mtime, "n_inputs": len(cells)},
    }
    with open(OUT, "w") as fh:
        json.dump(summary, fh, indent=1, sort_keys=True)
    print(
        f"{len(rows)} cells | all hit_cap=False: {summary['all_hit_cap_false']} | "
        f"min trustworthy widths: {summary['min_widths_trustworthy']}/12 | "
        f"cells w/ divergence: {summary['n_cells_any_diverged']} | "
        f"worst MSE/floor: {summary['worst_mse_over_noise_floor']:.3f}"
    )
    for n in sorted({r["n_train"] for r in rows}):
        line = [f"n={n:>4}:"]
        for s in sorted({r["sigma"] for r in rows}):
            vals = [r["mse_over_noise_floor"] for r in rows if r["n_train"] == n and r["sigma"] == s]
            line.append(f"s={s}: {min(vals):.2f}-{max(vals):.2f}")
        print("  ".join(line))


if __name__ == "__main__":
    main()
