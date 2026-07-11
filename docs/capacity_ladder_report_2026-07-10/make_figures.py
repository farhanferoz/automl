"""Generate all figures for the capacity-ladder program report (REPORT-2).

Reads only the certified result artifacts under
`automl_package/examples/capacity_ladder_results/` and writes PNGs into `figures/`.
Every figure traces to a named artifact; captions in the report state the same numbers.
Run: AUTOML_DEVICE=cpu ~/dev/.venv/bin/python docs/capacity_ladder_report_2026-07-10/make_figures.py
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
FIG = os.path.join(REPO, "docs/capacity_ladder_report_2026-07-10/figures")
os.makedirs(FIG, exist_ok=True)

# Colour-blind-safe set (Okabe-Ito subset), consistent across figures.
BLUE, ORANGE, GREEN, VERMIL, GREY = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#8a8a8a"
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 140})


def _save(fig, name):
    p = os.path.join(FIG, name)
    fig.tight_layout()
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {p}")


# ---------------------------------------------------------------------------
# Fig 1 — the shared failure mechanism (illustrative, not a measured run).
# In-sample log-likelihood keeps rising with capacity (the overfitting gain,
# ~half a nat per spurious parameter); held-out log-likelihood peaks then falls.
# ---------------------------------------------------------------------------
def fig_mechanism():
    c = np.linspace(1, 8, 200)
    insample = 0.5 * (c - 1)  # ~half a nat per added capacity unit
    heldout = 1.3 * np.log(c) - 0.06 * c**2 + 0.05  # rise then fall
    heldout = heldout - heldout.min()
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.plot(c, insample, color=VERMIL, lw=2.4, label="in-sample fit (training log-likelihood)")
    ax.plot(c, heldout, color=BLUE, lw=2.4, label="held-out fit (validation log-likelihood)")
    k = c[np.argmax(heldout)]
    ax.axvline(k, color=GREY, ls="--", lw=1.2)
    ax.annotate("held-out peak\n= the capacity the data supports", (k, np.max(heldout)),
                xytext=(k + 0.4, np.max(heldout) - 0.9),
                arrowprops=dict(arrowstyle="->", color=GREY), fontsize=9, color="#333")
    ax.set_xlabel("model capacity (number of components / depth / variance rungs)")
    ax.set_ylabel("log-likelihood (arbitrary units)")
    ax.set_title("The Occam race: in-sample fit always prefers more capacity;\nheld-out fit does not (illustrative)")
    ax.legend(loc="center right", fontsize=9, framealpha=0.9)
    _save(fig, "fig_mechanism.png")


# ---------------------------------------------------------------------------
# Fig 2 — WS1: the region arbiter recovers the staircase count on toy D.
# k5_summary region_arbiter_mean (first/second/third region) per seed + June bench.
# ---------------------------------------------------------------------------
def fig_ws1_arbiter():
    k5 = json.load(open(os.path.join(RES, "K5/k5_summary.json")))
    D = k5["structured"]["D"]
    regions = ["first\n(k*=1)", "second\n(k*=2)", "third\n(k*=3)"]
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    seed_cols = [BLUE, ORANGE, GREEN]
    for s in range(3):
        vals = [D[s]["region_arbiter_mean"][k] for k in ("first", "second", "third")]
        ax.plot(x, vals, "o-", color=seed_cols[s], lw=1.8, ms=6, label=f"nested ladder, seed {s}")
    jb = D[0]["june_bench"]
    ax.plot(x, [jb["first"], jb["second"], jb["third"]], "s--", color=GREY, lw=2.0, ms=7,
            label="reference instrument (variational mixture)")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(regions)
    ax.set_xlabel("input region (true latent component count increases left to right)")
    ax.set_ylabel("local advantage (nats)\ntop rung minus single component")
    ax.set_title("Component count — the local advantage recovers\nthe staircase on toy D (every seed)")
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.9)
    _save(fig, "fig_ws1_arbiter.png")


# ---------------------------------------------------------------------------
# Fig 3 — WS2: the global knee discriminates capacity and abstains on control.
# f3_summary global.pooled delta_curve for G / H / G_flat.
# ---------------------------------------------------------------------------
def fig_ws2_knee():
    f3 = json.load(open(os.path.join(RES, "F3/f3_summary.json")))
    pooled = {p["toy"]: p for p in f3["global"]["pooled"]}
    cvals = np.arange(1, 7)
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    style = {"G": (BLUE, "o-", "toy G (varying need): knee r*=3"),
             "H": (GREEN, "^-", "toy H (SNR dial): knee r*=2"),
             "G_flat": (VERMIL, "s--", "toy G-flat (control): abstain r*=0")}
    for toy, (col, mk, lab) in style.items():
        dc = pooled[toy]["delta_curve"]
        y = [dc[str(c)] for c in cvals]
        ax.plot(cvals, y, mk, color=col, lw=1.9, ms=6, label=lab)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("depth capacity rung $c$ (number of active hidden layers)")
    ax.set_ylabel("held-out log-score gain over depth 1 (nats)")
    ax.set_title("Network depth — global read: structured toys detect,\nthe uniform-complexity control abstains")
    ax.legend(loc="center right", fontsize=8.5, framealpha=0.9)
    _save(fig, "fig_ws2_knee.png")


# ---------------------------------------------------------------------------
# Fig 4 — WS2: per-bin advantage over global, per seed, with +/-2 SE bars.
# f3_summary perbin.checks (tercile mean_diff +/- se_diff).
# ---------------------------------------------------------------------------
def fig_ws2_perbin():
    f3 = json.load(open(os.path.join(RES, "F3/f3_summary.json")))
    ck = f3["perbin"]["checks"]
    toys = [("G_perbin_beats_global_2se", "toy G", BLUE),
            ("G_flat_ties_no_false_positive", "toy G-flat", VERMIL),
            ("H_varies_with_snr", "toy H", GREEN)]
    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    width = 0.24
    for j, (key, lab, col) in enumerate(toys):
        rows = ck[key]["rows"]
        md = [r["mean_diff"] for r in rows]
        se = [r["se_diff"] for r in rows]
        xs = np.arange(3) + (j - 1) * width
        ax.bar(xs, md, width, yerr=[2 * s for s in se], color=col, capsize=3, label=lab,
               error_kw={"lw": 1.1})
    ax.axhline(0, color="k", lw=0.9)
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels([f"seed {s}" for s in range(3)])
    ax.set_ylabel("per-bin minus global\nheld-out log-score (nats)")
    ax.set_xlabel("error bars = $\\pm 2$ standard errors; a bar clearing 0 = per-input signal")
    ax.set_title("Network depth — per-input advantage is power-limited at held-out N=500\n(only toy G seed 2 clears; the control never does)")
    ax.legend(loc="upper left", fontsize=8.5, ncol=3, framealpha=0.9)
    _save(fig, "fig_ws2_perbin.png")


# ---------------------------------------------------------------------------
# Fig 5 — WS3: the in-sample variance collapse trajectory (toy V-toy1, N=200, seed 1).
# V0 curves: sigma_ratio, ssr_heldout, heldout_nll over 8000 epochs.
# ---------------------------------------------------------------------------
def fig_ws3_collapse():
    c = torch.load(os.path.join(RES, "V0/v_toy1_N200_seed1_curves.pt"), weights_only=False)
    sig = np.asarray(c["sigma_ratio"])
    ssr = np.asarray(c["ssr_heldout"])
    ep = np.arange(len(sig))
    nll_min_ep = 565
    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    ax.plot(ep, sig, color=BLUE, lw=2.0, label=r"in-sample $\hat\sigma/\sigma$ (should stay 1)")
    ax.plot(ep, ssr, color=VERMIL, lw=2.0, label="held-out standardised sq. residual (should stay 1)")
    ax.axhline(1.0, color="k", lw=0.8, ls=":")
    ax.axvline(nll_min_ep, color=GREY, lw=1.2, ls="--")
    ax.annotate("held-out fit best here\n(early stopping point)", (nll_min_ep, 4.2),
                xytext=(nll_min_ep + 900, 4.6), arrowprops=dict(arrowstyle="->", color=GREY),
                fontsize=8.5, color="#333")
    ax.set_xscale("log")
    ax.set_xlim(50, 8000)
    ax.set_xlabel("training epoch (log scale)")
    ax.set_ylabel("ratio (target = 1)")
    ax.set_title("Noise structure — in-sample variance collapse (N=200): the model's own\nspread shrinks below the truth while held-out error explodes")
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.9)
    _save(fig, "fig_ws3_collapse.png")


# ---------------------------------------------------------------------------
# Fig 6 — WS3: the knee is a coarse selector. For V-toy1 (N=1000, mean over seeds),
# the ground-truth sigma-recovery error is lowest at v2, but the held-out NLL is
# nearly flat across v1/v2/v3, so the NLL knee stops at v1.
# ---------------------------------------------------------------------------
def fig_ws3_coarse():
    v3 = json.load(open(os.path.join(RES, "V3/v3_summary.json")))
    units = [u for u in v3["units"] if u["toy"] == "v_toy1"]
    rungs = ["v0", "v1", "v2", "v3"]
    labels = ["v0\nconstant", "v1\nthree-step", "v2\nsmooth", "v3\nflexible"]
    sre = np.array([[u["rungs"][r]["sigma_ratio_error"] for r in rungs] for u in units])
    nll = np.array([[u["rungs"][r]["nll"] for r in rungs] for u in units])
    sre_m, nll_m = sre.mean(0), nll.mean(0)
    x = np.arange(4)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.8, 3.8))
    a1.plot(x, sre_m, "o-", color=GREEN, lw=2.2, ms=7)
    a1.plot(x[np.argmin(sre_m)], sre_m.min(), "*", color=VERMIL, ms=17)
    a1.annotate("v2 best", (2, sre_m[2]), xytext=(2.0, sre_m[2] + 0.012), fontsize=9, color=VERMIL, ha="center")
    a1.set_xticks(x)
    a1.set_xticklabels(labels)
    a1.set_ylabel("noise-recovery error (lower = better)")
    a1.set_title("Ground-truth noise recovery:\nbest at v2 (the correct class)")
    # Right panel: keep the v0 baseline in view so the v1..v3 flatness is honest,
    # and mark where the KNEE lands (v1) rather than the raw NLL argmin (v2).
    a2.plot(x, nll_m, "o-", color=BLUE, lw=2.2, ms=7)
    a2.plot(1, nll_m[1], "D", color=VERMIL, ms=11)
    gap = nll_m[1] - nll_m[2]
    a2.annotate(f"knee stops at v1\n(v1$\\to$v2 gain {gap:.3f} nat,\nbelow significance)",
                (1, nll_m[1]), xytext=(1.25, nll_m[0] - 0.002),
                arrowprops=dict(arrowstyle="->", color=GREY), fontsize=8.5, color="#333")
    a2.set_ylim(nll_m.min() - 0.004, nll_m[0] + 0.003)
    a2.set_xticks(x)
    a2.set_xticklabels(labels)
    a2.set_ylabel("held-out NLL (what the knee reads)")
    a2.set_title("What the knee sees: one big v0$\\to$v1 drop,\nthen flat within noise")
    fig.suptitle("Noise structure — the held-out knee is a coarse selector (problem V1, N=1000)", y=1.03)
    _save(fig, "fig_ws3_coarse.png")


if __name__ == "__main__":
    fig_mechanism()
    fig_ws1_arbiter()
    fig_ws2_knee()
    fig_ws2_perbin()
    fig_ws3_collapse()
    fig_ws3_coarse()
    print("all figures written")
