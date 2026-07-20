"""Generate the figures for the ProbReg k-selection report.

Reads only the certified result artifacts under
`automl_package/examples/capacity_ladder_results/` and writes PNGs into `figures/`.
Every figure traces to a named artifact; the report text states the same numbers.
Run: AUTOML_DEVICE=cpu ~/dev/.venv/bin/python docs/reports/probreg_kselection/make_figures.py
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = "/home/ff235/dev/MLResearch/automl"
RES = os.path.join(REPO, "automl_package/examples/capacity_ladder_results")
FIG = os.path.join(REPO, "docs/reports/probreg_kselection/figures")
os.makedirs(FIG, exist_ok=True)

BLUE, ORANGE, GREEN, VERMIL, GREY = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#8a8a8a"
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 140})


def _save(fig, name):
    p = os.path.join(FIG, name)
    fig.tight_layout()
    fig.savefig(p, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"wrote {p}")


# ---------------------------------------------------------------------------
# Fig 1 — why prefix-masking an already-trained, unordered mixture fails.
# Column means of held-out log-likelihood by prefix length, toy D seed 0:
# the invalid instrument (bins seeded by location, killed at arbitrary
# positions by a separate sparsity prior) vs the valid same-architecture
# per-count reference (separately trained fixed-count models).
# ---------------------------------------------------------------------------
def fig_kselect_invalid():
    d = torch.load(os.path.join(RES, "K0/aggregate_sparsity_toyD_seed0.pt"), weights_only=False, map_location="cpu")
    sm = d["score_mat"]
    invalid_colmeans = sm.mean(dim=0).numpy()
    c_grid = d["c_grid"]

    k4 = json.load(open(os.path.join(RES, "K4/k4_summary.json")))
    ref_colmeans = np.array(k4["structured"]["D"][0]["k0_mdn75_colmeans"])

    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.plot(c_grid, invalid_colmeans, "o-", color=VERMIL, lw=2.0, ms=7,
            label="prefix of an unordered, sparsity-pruned classifier (invalid)")
    ax.plot(c_grid, ref_colmeans, "s-", color=BLUE, lw=2.0, ms=7,
            label="separately trained models, one per class count (reference)")
    ax.set_xlabel("number of classes used, c")
    ax.set_ylabel("held-out log-likelihood, averaged over inputs\n(nats; higher = better fit)")
    ax.set_title("Reading off the first c classes of an already-trained classifier\ndoes not give a valid c-class model (the staircase problem, one repeat)")
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.9)
    _save(fig, "fig-kselect-invalid.png")


# ---------------------------------------------------------------------------
# Fig 2 — the held-out arbiter recovers the staircase; the per-input knee
# does not. Toy D, k5_summary region_arbiter_mean / region_modal_knee.
# ---------------------------------------------------------------------------
def fig_kselect_arbiter():
    k5 = json.load(open(os.path.join(RES, "K5/k5_summary.json")))
    D = k5["structured"]["D"]
    regions = ["first\n(truth=1)", "second\n(truth=2)", "third\n(truth=3)"]
    x = np.arange(3)
    seed_cols = [BLUE, ORANGE, GREEN]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 4.0))
    for s in range(3):
        vals = [D[s]["region_arbiter_mean"][k] for k in ("first", "second", "third")]
        a1.plot(x, vals, "o-", color=seed_cols[s], lw=1.8, ms=6, label=f"repeat {s + 1}")
    a1.axhline(0, color="k", lw=0.8)
    a1.set_xticks(x)
    a1.set_xticklabels(regions)
    a1.set_ylabel("held-out advantage of the full class budget\nover a single class (nats)")
    a1.set_title("The arbiter: rises with the true count\non every repeat")
    a1.legend(loc="upper left", fontsize=8.5, framealpha=0.9)

    width = 0.25
    for s in range(3):
        vals = [D[s]["region_modal_knee"][k] for k in ("first", "second", "third")]
        a2.bar(x + (s - 1) * width, vals, width, color=seed_cols[s], label=f"repeat {s + 1}")
    a2.plot(x, [1, 2, 3], "k*--", ms=12, lw=1.2, label="true count")
    a2.set_xticks(x)
    a2.set_xticklabels(regions)
    a2.set_ylabel("most common per-input elbow reading")
    a2.set_title("The per-input elbow: noisy, and wrong\non most repeats")
    a2.legend(loc="upper left", fontsize=8.5, framealpha=0.9)
    fig.suptitle("Class count on the staircase problem — the arbiter reads it, the elbow does not", y=1.03)
    _save(fig, "fig-kselect-arbiter.png")


# ---------------------------------------------------------------------------
# Fig 3 — the distilled router: nine held-out cases, four ways of choosing
# the per-input route. K6/k6_summary.json.
# ---------------------------------------------------------------------------
def fig_kselect_router():
    k6 = json.load(open(os.path.join(RES, "K6/k6_summary.json")))
    s = k6["structured"]
    name = {"C": "step", "D": "stair", "E": "hump"}
    cases, glob, soft, hard, pilot = [], [], [], [], []
    for toy in ("C", "D", "E"):
        for row in s[toy]:
            cases.append(f"{name[toy]}\n{row['seed'] + 1}")
            glob.append(row["nll_global"])
            soft.append(row["nll_soft"])
            hard.append(row["nll_hard"])
            pilot.append(row["nll_pilot"])

    x = np.arange(len(cases))
    width = 0.2
    fig, ax = plt.subplots(figsize=(9.6, 4.2))
    ax.bar(x - 1.5 * width, glob, width, color=GREY, label="best single fixed count (baseline)")
    ax.bar(x - 0.5 * width, soft, width, color=BLUE, label="selector, soft-labelled")
    ax.bar(x + 0.5 * width, hard, width, color=VERMIL, label="selector, elbow-labelled")
    ax.bar(x + 1.5 * width, pilot, width, color=ORANGE, label="selector, raw-argmax-labelled")
    ax.set_xticks(x)
    ax.set_xticklabels(cases)
    ax.set_xlabel("case (test problem, repeat)")
    ax.set_ylabel("held-out negative log-likelihood (nats; lower = better)")
    ax.set_title("The soft-labelled selector matches or beats a single fixed count\non all nine held-out cases")
    ax.legend(loc="upper left", fontsize=8, ncol=2, framealpha=0.9)
    _save(fig, "fig-kselect-router.png")


# ---------------------------------------------------------------------------
# Fig 4 — the two honest negatives: the moving-mode toy (E) does not recover
# on 2 of 3 seeds, and the held-out knee under-resolves the noise-function
# selection problem it shares its mechanism with (V3).
# ---------------------------------------------------------------------------
def fig_kselect_negatives():
    k5 = json.load(open(os.path.join(RES, "K5/k5_summary.json")))
    E = k5["structured"]["E"]
    regions = ["low x", "mid x", "high x"]
    x = np.arange(3)
    seed_cols = [BLUE, ORANGE, GREEN]

    v3 = json.load(open(os.path.join(RES, "V3/v3_summary.json")))
    units = [u for u in v3["units"] if u["toy"] == "v_toy1"]
    rungs = ["v0", "v1", "v2", "v3"]
    labels = ["one scalar", "3-step", "smooth\nfunction", "flexible\nnetwork"]
    sre = np.array([[u["rungs"][r]["sigma_ratio_error"] for r in rungs] for u in units]).mean(0)
    nll = np.array([[u["rungs"][r]["nll"] for r in rungs] for u in units]).mean(0)

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(12.6, 4.0))
    for si in range(3):
        vals = [E[si]["region_arbiter_mean"][k] for k in ("low", "mid", "high")]
        a1.plot(x, vals, "o-", color=seed_cols[si], lw=1.8, ms=6, label=f"repeat {si + 1}")
    a1.axhline(0, color="k", lw=0.8)
    a1.set_xticks(x)
    a1.set_xticklabels(regions)
    a1.set_ylabel("held-out advantage of the full class budget\nover a single class (nats)")
    a1.set_title("Moving-mode problem: no shared rise-then-fall\nacross repeats at this sample size")
    a1.legend(loc="upper right", fontsize=8, framealpha=0.9)

    a2.plot(np.arange(4), sre, "o-", color=GREEN, lw=2.2, ms=8)
    a2.plot(2, sre[2], "*", color=VERMIL, ms=18)
    a2.annotate("best recovery", (2, sre[2]), xytext=(1.55, sre[2] + 0.011), fontsize=8.5, color=VERMIL)
    a2.set_xticks(np.arange(4))
    a2.set_xticklabels(labels)
    a2.set_ylabel("noise-recovery error (lower = better)")
    a2.set_title("How well each option recovers\nthe true noise function")

    a3.plot(np.arange(4), nll, "s-", color=BLUE, lw=2.2, ms=8)
    a3.plot(1, nll[1], "D", color=VERMIL, ms=11)
    gap = nll[1] - nll[2]
    a3.annotate(f"the selection score stops here\n(gap to the next option\n{gap:.3f} nat, below noise)",
                (1, nll[1]), xytext=(1.15, nll[0] - 0.001),
                arrowprops=dict(arrowstyle="->", color=GREY), fontsize=8, color="#333")
    a3.set_ylim(nll.min() - 0.003, nll[0] + 0.002)
    a3.set_xticks(np.arange(4))
    a3.set_xticklabels(labels)
    a3.set_ylabel("held-out negative log-likelihood\n(lower = better; the selection score)")
    a3.set_title("The score itself is nearly flat\npast the first step")
    fig.suptitle("Two honest boundaries: an unrecovered case, and a resolution limit in the selection score", y=1.03)
    _save(fig, "fig-kselect-negatives.png")


if __name__ == "__main__":
    fig_kselect_invalid()
    fig_kselect_arbiter()
    fig_kselect_router()
    fig_kselect_negatives()
    print("all figures written")
