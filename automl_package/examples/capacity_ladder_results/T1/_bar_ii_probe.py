"""Probe: does the DECOMPOSED per-region read recover the planted +0.8-nat B signal
that the POOLED read (bar ii as written) is blind to?

Builds the exact selftest part-(c) planted table (`_synthetic_t1_table` default flat-A /
+0.8-nat-B) and runs `_repeated_perbin_by_region(n_bins=2)`, printing region-A and region-B
t_corrected for BOTH the independent and hierarchical arms. Also re-prints the pooled read for
side-by-side comparison, and repeats on the (b)-style genuinely-conflicting table as a positive
control for the decomposed reader.
"""

import os
import sys

import numpy as np

_EXAMPLES = "/home/ff235/dev/MLResearch/automl/automl_package/examples"
sys.path.insert(0, _EXAMPLES)
sys.path.insert(0, "/home/ff235/dev/MLResearch/automl")

import capacity_ladder_t1 as t1  # noqa: E402
import capacity_ladder_x1 as x1mod  # noqa: E402
import capacity_ladder_x3 as x3mod  # noqa: E402


def _report(tag, score, x):
    print(f"\n=== {tag} ===")
    # pooled (bar ii as written)
    cf = x3mod.run_repeated_crossfit(score, x, n_splits=t1.N_SPLITS, n_bins=2)
    hi = x1mod.run_repeated(score, x, n_splits=t1.N_SPLITS, n_bins=2)["hier_vs_global"]
    print(f"  POOLED (bar ii as written):  crossfit_t={cf['t_corrected']:+.2f}  hier_t={hi['t_corrected']:+.2f}")
    # decomposed per-region (the proposed fix)
    dec = t1._repeated_perbin_by_region(score, x, n_bins=2, n_splits=t1.N_SPLITS)
    bin_ids = sorted(dec.keys())
    lo_bin, hi_bin = bin_ids[0], bin_ids[-1]  # bin 0 = region A (x<0.5), bin 1 = region B (x>=0.5)
    for label, b in (("region_A (x<0.5)", lo_bin), ("region_B (x>=0.5)", hi_bin)):
        di = dec[b]["indep"]
        dh = dec[b]["hier"]
        print(f"  DECOMPOSED {label}: indep_t={di['t_corrected']:+.2f} (mu={di['mu_bar']:+.4f}, passfrac={di['split_pass_fraction']:.2f}) "
              f"| hier_t={dh['t_corrected']:+.2f} (mu={dh['mu_bar']:+.4f}, passfrac={dh['split_pass_fraction']:.2f})")


# (c) planted: true-flat A + +0.8-nat B (the literal T1 region shape)
score_flat, x_flat = t1._synthetic_t1_table()
_report("PLANTED flat-A / +0.8-nat-B  (selftest part c; pooled reads ~null)", score_flat, x_flat)

# (b) positive control: genuinely conflicting regions
means_a_conflict = np.array([0.8, 0.4, 0.0, 0.0, 0.0, 0.0])
score_conf, x_conf = t1._synthetic_t1_table(means_a=means_a_conflict)
_report("CONFLICTING regions (selftest part b; pooled reads t~76)", score_conf, x_conf)
