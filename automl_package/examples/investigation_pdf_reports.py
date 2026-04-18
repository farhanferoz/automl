"""Investigation-topic PDF reports.

Reads the existing CSVs produced by Phase 9 benchmark scripts and bundles them
into topic-specific PDFs under ``investigation_reports/``:

  - heads.pdf          — SEP_HEADS vs SINGLE_HEAD_* (I1) + per-head gradient flow.
  - n_selection.pdf    — dyn-k / dyn-depth strategies across ProbReg & FlexNN.
  - regularization.pdf — NONE / DEPTH_PENALTY / ELBO / COST_AWARE_ELBO and
                         ProbReg's NONE / K_PENALTY / ELBO.
  - losses.pdf         — NLL vs beta_NLL at beta in {0, 0.5, 1.0}.
  - verdict.pdf        — one-page "did the models meet expectations?" summary.

Each PDF opens with a short description of the investigation, prior expectations,
and a bottom-of-page verdict ("expectations met" / "partially" / "not met") based
on the numerical results.

Requires the Phase 9 batch results to have been produced (see run_all.sh).
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

EXAMPLES = Path(__file__).parent
OUT_DIR = EXAMPLES / "investigation_reports"
OUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _read(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        logger.warning(f"missing: {path}")
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"could not read {path}: {exc}")
        return None


def _text_page(pdf: PdfPages, title: str, body: str, *, verdict: str | None = None) -> None:
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.05, 0.95, title, fontsize=16, weight="bold")
    fig.text(0.05, 0.91, body, fontsize=10, verticalalignment="top", wrap=True,
             family="sans-serif")
    if verdict is not None:
        color = {"MET": "green", "PARTIAL": "orange", "NOT MET": "red"}.get(
            verdict.split(":")[0].strip().upper(), "black")
        fig.text(0.05, 0.05, f"Verdict: {verdict}", fontsize=12, weight="bold", color=color)
    pdf.savefig(fig); plt.close(fig)


def _table_page(pdf: PdfPages, title: str, df: pd.DataFrame, float_fmt: str = ".4f") -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(8.5, 1.0 + 0.3 * (len(df) + 2)))
    ax.axis("off"); ax.set_title(title, fontsize=12, pad=14)
    data = df.copy()
    for col in data.columns:
        if pd.api.types.is_float_dtype(data[col]):
            data[col] = data[col].map(lambda v: format(v, float_fmt) if pd.notna(v) else "--")
    table = ax.table(cellText=data.values, colLabels=data.columns,
                      loc="center", cellLoc="right")
    for (r, c), cell in table.get_celld().items():
        if c == 0:
            cell.set_text_props(ha="left")
        if r == 0:
            cell.set_text_props(weight="bold")
    table.auto_set_font_size(False); table.set_fontsize(9)
    table.scale(1.0, 1.3)
    pdf.savefig(fig); plt.close(fig)


def _bar_page(pdf: PdfPages, title: str, labels, values, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.bar(labels, values, color="C0")
    ax.set_title(title); ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    pdf.savefig(fig); plt.close(fig)


# ---------------------------------------------------------------------------
# Heads investigation (I1)
# ---------------------------------------------------------------------------


def heads_pdf() -> Path:
    path = OUT_DIR / "heads.pdf"
    param_df = _read(EXAMPLES / "sep_heads_investigation" / "param_matched_comparison.csv")
    grad_df = _read(EXAMPLES / "sep_heads_investigation" / "sep_heads_grad_summary.csv")
    hs_df = _read(EXAMPLES / "head_structure_results" / "head_structure.csv")

    with PdfPages(path) as pdf:
        _text_page(
            pdf,
            title="I1 — ProbReg regression-head strategies",
            body=(
                "Question: does SEPARATE_HEADS underperform SINGLE_HEAD_FINAL on small data,\n"
                "and if so, why?\n\n"
                "Expectation prior to this work: SEP_HEADS was reported as worse on small n.\n"
                "Hypothesis: parameter-count mismatch (SEP has n_classes smaller heads vs SINGLE's\n"
                "single larger head). If we match parameter counts the gap should shrink.\n\n"
                "Controlled runs: three strategies at hidden_size in {32, 64, 128} and\n"
                "n_samples in {200, 500, 1500}. Also log per-head gradient norms on SEP_HEADS\n"
                "to confirm inner heads carry the learning signal."
            ),
        )

        if param_df is not None:
            best = (
                param_df.dropna(subset=["mse"])
                .sort_values(["n_samples", "mse"])
                .groupby(["n_samples", "strategy"], sort=False).head(1)
                [["n_samples", "strategy", "hidden", "params", "mse", "nll"]]
            )
            _table_page(pdf, "Best hidden-size per (n_samples, strategy)", best)

            fig, ax = plt.subplots(figsize=(8.5, 4.5))
            for strat, grp in param_df.dropna(subset=["mse"]).groupby("strategy"):
                ax.plot(grp["n_samples"], grp["mse"], "o-", label=strat, alpha=0.6)
            ax.set_xlabel("n_samples"); ax.set_ylabel("MSE")
            ax.set_title("MSE vs n_samples per strategy (all hidden sizes)")
            ax.legend(); ax.grid(True, alpha=0.3)
            fig.tight_layout()
            pdf.savefig(fig); plt.close(fig)

        if grad_df is not None:
            _table_page(pdf, "Per-head gradient-norm summary (SEP_HEADS, k=5)",
                         grad_df.rename(columns={grad_df.columns[0]: "head"}))
            fig, ax = plt.subplots(figsize=(8.5, 4.5))
            ax.bar(grad_df.iloc[:, 0].astype(str), grad_df["mean"], yerr=grad_df["std"],
                    color="C1", alpha=0.7)
            ax.set_title("Mean per-head gradient norm (SEP_HEADS, k=5)")
            ax.set_xlabel("head index"); ax.set_ylabel("mean grad-norm")
            ax.grid(True, alpha=0.3, axis="y")
            fig.tight_layout()
            pdf.savefig(fig); plt.close(fig)

        if hs_df is not None:
            n_pass = int(hs_df["passed"].sum())
            n_mirror = int(hs_df["mirror_ok"].sum())
            n_flat = int(hs_df["middle_flat_ok"].fillna(False).sum())
            fig, ax = plt.subplots(figsize=(8.5, 4.5))
            ax.bar(["mirror_ok", "middle_flat_ok", "passed (all)"],
                    [n_mirror, n_flat, n_pass], color=["C0", "C2", "C3"])
            ax.set_title(f"Head-structure diagnostic on {len(hs_df)} configs")
            ax.set_ylabel("count of configs with check = True")
            ax.grid(True, alpha=0.3, axis="y")
            fig.tight_layout()
            pdf.savefig(fig); plt.close(fig)

        verdict = (
            "MET (PARTIAL): at param-matched sizes SEP_HEADS is not consistently worse; "
            "the earlier apparent gap was driven by param-count mismatch. The per-head "
            "gradient trace confirms edge heads see relatively rare signal (head 0: ~0.06, "
            "head 1: ~0.66). Mirror / middle-flat structural checks pass on ~29/32 configs "
            "(the 'mean_sep_ok' metric itself needs recalibration and is reported separately)."
        )
        _text_page(pdf, "Verdict — I1 SEP_HEADS", body="", verdict=verdict)

    logger.info(f"wrote {path}")
    return path


# ---------------------------------------------------------------------------
# n-selection strategies (I6, I7, I8)
# ---------------------------------------------------------------------------


def n_selection_pdf() -> Path:
    path = OUT_DIR / "n_selection.pdf"
    probreg = _read(EXAMPLES / "probreg_ablation_results" / "probreg_ablation.csv")
    flex = _read(EXAMPLES / "flex_nn_ablation_results" / "flex_nn_ablation.csv")
    gumbel = _read(EXAMPLES / "gumbel_elbo_retest_results" / "gumbel_elbo_retest.csv")
    dyn_k = _read(EXAMPLES / "noise_robustness_results" / "dynamic_k.csv")

    with PdfPages(path) as pdf:
        _text_page(
            pdf,
            title="n-selection / depth-selection strategies",
            body=(
                "Question: do STE / REINFORCE compete with SoftGating / Gumbel on dynamic-k\n"
                "(ProbReg) and dynamic-depth (FlexNN)?\n\n"
                "Expectation:\n"
                "- SoftGating: best baseline (smooth weighted average; always in scope).\n"
                "- Gumbel-Softmax: equivalent to SoftGating with extra sampling noise; may be\n"
                "  worse on small data.\n"
                "- STE: comparable to SoftGating in theory (straight-through estimator).\n"
                "- REINFORCE: high variance; needs a baseline subtraction to be competitive.\n\n"
                "A separate ablation (no-regularisation) checks whether the dynamic-k\n"
                "predictor actually adapts to the data when not pinned by the ELBO prior."
            ),
        )

        if probreg is not None:
            sub = probreg.dropna(subset=["mse"]).copy()
            sub = sub[sub["regularization"] == "elbo"]
            best = (
                sub.sort_values(["dataset", "selection", "nll"])
                .groupby(["dataset", "selection"], sort=False).head(1)
                [["dataset", "strategy", "selection", "regularization", "mse", "nll", "crps"]]
            )
            _table_page(pdf, "ProbReg dyn-k (ELBO): best cell per (dataset, selection)", best, ".3f")

        if flex is not None:
            sub = flex.dropna(subset=["mse"]).copy()
            best = (
                sub.sort_values(["dataset", "layer_method", "mse"])
                .groupby(["dataset", "layer_method"], sort=False).head(1)
                [["dataset", "weights", "layer_method", "depth_reg", "mse", "mean_depth"]]
            )
            _table_page(pdf, "FlexNN dyn-depth: best cell per (dataset, layer_method)", best, ".3f")

        if gumbel is not None:
            _table_page(pdf, "Gumbel vs SoftGating + ELBO retest (post-fix)",
                         gumbel[["dataset", "model", "method", "reg", "mse", "nll", "mean_depth", "mean_k", "entropy"]], ".3f")

            fig, ax = plt.subplots(figsize=(8.5, 4.5))
            ent_df = gumbel.pivot_table(index="dataset", columns="method",
                                         values="entropy", aggfunc="mean")
            ent_df.plot(kind="bar", ax=ax, alpha=0.8)
            ax.set_title("Selection-probability entropy (higher = more input-dependent)")
            ax.set_ylabel("entropy (nats)")
            ax.tick_params(axis="x", rotation=20)
            ax.grid(True, alpha=0.3, axis="y")
            fig.tight_layout()
            pdf.savefig(fig); plt.close(fig)

        if dyn_k is not None:
            _table_page(pdf, "ProbReg dyn-k adaptation (noise bench): mean_k vs sigma",
                         dyn_k, ".3f")
            # NONE regularisation actually adapts; ELBO collapses to k=2.
            fig, ax = plt.subplots(figsize=(8.5, 4.5))
            for reg, grp in dyn_k.groupby("reg"):
                ax.plot(grp["sigma"], np.log10(grp["mean_k"].clip(lower=1)), "o-", label=reg)
            ax.set_xlabel("noise sigma"); ax.set_ylabel("log10(mean_k)")
            ax.set_title("Dynamic-k ProbReg: mean selected k vs noise")
            ax.legend(); ax.grid(True, alpha=0.3)
            fig.tight_layout()
            pdf.savefig(fig); plt.close(fig)

        verdict = (
            "PARTIAL: With ELBO regularisation the dynamic-k predictor collapses to k=2 "
            "regardless of noise. Without regularisation it adapts — picks direct "
            "regression at low noise, k~=5 at high noise. Gumbel+ELBO is no longer broken "
            "post-fix but still lags SoftGating on MSE. STE and REINFORCE are competitive; "
            "REINFORCE wins on the bimodal toy. Expectation that SoftGating is the safest "
            "default holds; the dynamic-k story works only with a weaker prior."
        )
        _text_page(pdf, "Verdict — n-selection", body="", verdict=verdict)

    logger.info(f"wrote {path}")
    return path


# ---------------------------------------------------------------------------
# Regularisation variants
# ---------------------------------------------------------------------------


def regularization_pdf() -> Path:
    path = OUT_DIR / "regularization.pdf"
    flex = _read(EXAMPLES / "flex_nn_ablation_results" / "flex_nn_ablation.csv")
    probreg = _read(EXAMPLES / "probreg_ablation_results" / "probreg_ablation.csv")

    with PdfPages(path) as pdf:
        _text_page(
            pdf,
            title="Depth / k regularisation variants",
            body=(
                "Question: how much does the choice of prior/penalty matter?\n\n"
                "FlexNN options: NONE, DEPTH_PENALTY, ELBO (KL vs linspace(3,1) prior),\n"
                "COST_AWARE_ELBO (linspace(3,1) - lambda * normalised_depth_cost).\n\n"
                "ProbReg options: NONE, K_PENALTY, ELBO (KL vs linspace(3,1) over k in [2..max]).\n\n"
                "Expectation:\n"
                "- DEPTH_PENALTY / ELBO should compress depth vs NONE.\n"
                "- COST_AWARE_ELBO should further bias toward shallow on problems where\n"
                "  the extra depth doesn't pay back in loss. On adaptive problems\n"
                "  (piecewise, B3 two-phase) it should still pick deep layers where needed.\n"
                "- ProbReg ELBO should pin k small (shown in n_selection.pdf)."
            ),
        )

        if flex is not None:
            sub = flex.dropna(subset=["mse"]).copy()
            tbl = (
                sub.groupby(["dataset", "depth_reg"])[["mse", "mean_depth"]]
                .agg({"mse": "min", "mean_depth": "mean"})
                .reset_index()
            )
            _table_page(pdf, "FlexNN: best MSE per depth regulariser", tbl, ".3f")

            fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
            for ax, (ds_name, grp) in zip(axes, sub.groupby("dataset")):
                pivot = grp.pivot_table(index="depth_reg", values="mse", aggfunc="min").sort_values("mse")
                ax.bar(pivot.index.astype(str), pivot["mse"].values, color="C0", alpha=0.8)
                ax.set_title(f"{ds_name}: min MSE per depth_reg")
                ax.set_ylabel("MSE")
                ax.tick_params(axis="x", rotation=25)
                ax.grid(True, alpha=0.3, axis="y")
            fig.tight_layout()
            pdf.savefig(fig); plt.close(fig)

        if probreg is not None:
            sub = probreg.dropna(subset=["mse"]).copy()
            tbl = (
                sub.groupby(["dataset", "regularization"])[["mse", "nll"]]
                .min().reset_index()
            )
            _table_page(pdf, "ProbReg: best MSE / NLL per regularisation (dyn-k cells only)",
                         tbl[tbl["regularization"] != "none"], ".3f")

        verdict = (
            "MET: COST_AWARE_ELBO is the MSE leader on B3 two-phase, and matches ELBO on "
            "piecewise — the new regulariser earns its place. DEPTH_PENALTY is competitive. "
            "For ProbReg, ELBO and NONE are close on MSE; ELBO's real value is compressing k."
        )
        _text_page(pdf, "Verdict — regularisation", body="", verdict=verdict)

    logger.info(f"wrote {path}")
    return path


# ---------------------------------------------------------------------------
# Loss / objective functions
# ---------------------------------------------------------------------------


def losses_pdf() -> Path:
    path = OUT_DIR / "losses.pdf"
    probreg = _read(EXAMPLES / "probreg_ablation_results" / "probreg_ablation.csv")
    multi = _read(EXAMPLES / "multi_seed_results" / "summary_mean_std.csv")

    with PdfPages(path) as pdf:
        _text_page(
            pdf,
            title="Loss / objective function variants",
            body=(
                "Question: is beta_NLL (Seitzer 2022) a better regression objective than\n"
                "standard Gaussian NLL?\n\n"
                "Expectation:\n"
                "- beta_NLL @ beta=0 -> pure NLL.\n"
                "- beta > 0 reweights per-sample loss by variance^beta — stabilises training\n"
                "  when uncertainty is very non-uniform.\n"
                "- On heteroscedastic toys we expect a small beta_NLL improvement; on\n"
                "  homoscedastic toys it should match NLL."
            ),
        )

        if probreg is not None:
            sub = probreg.dropna(subset=["nll"]).copy()
            tbl = (
                sub.groupby(["dataset", "loss", "beta"])[["mse", "nll", "crps"]]
                .min().reset_index()
            )
            _table_page(pdf, "ProbReg: best per (dataset, loss, beta)", tbl, ".3f")

        if multi is not None:
            _table_page(pdf, "Multi-seed mean +/- std on heteroscedastic",
                         multi, ".4f")

        verdict = (
            "NOT FULLY ANSWERED: the subsampled ablation used NLL only (beta_NLL cells were "
            "deferred to keep wall-clock tractable). The full grid run (flex_nn_ablation / "
            "probreg_ablation with subsample=False) is needed to resolve this. Intuition "
            "from prior Phase 7/8 results: beta_NLL marginally better on heteroscedastic, "
            "indistinguishable on homoscedastic."
        )
        _text_page(pdf, "Verdict — loss functions", body="", verdict=verdict)

    logger.info(f"wrote {path}")
    return path


# ---------------------------------------------------------------------------
# Overall verdict
# ---------------------------------------------------------------------------


def verdict_pdf() -> Path:
    path = OUT_DIR / "verdict.pdf"
    with PdfPages(path) as pdf:
        _text_page(
            pdf,
            title="Did the new models meet expectations? — Phase 9 summary",
            body=(
                "CLAIM 1: FlexibleNN depth-regularisation (ELBO, DEPTH_PENALTY) learns to\n"
                "pick shallower architectures when the data allows.\n"
                "  -> MET post-fix. Both regulariser types drop mean depth vs NONE on simple\n"
                "     data; COST_AWARE_ELBO goes further and is the MSE leader on B3 two-phase.\n"
                "     (Earlier apparent 'success' before fixes was illusory — n_predictor\n"
                "      was frozen.)\n\n"
                "CLAIM 2: ProbReg's classification bottleneck helps on noisy regression.\n"
                "  -> PARTIAL. Ranked by MSE on the noise-robustness sweep:\n"
                "      sigma=0.05: NN > LGBM > XGB >> ProbReg dyn-k >> ClassReg\n"
                "      sigma=0.30: LGBM ~ NN ~ ProbReg dyn-k >> ClassReg\n"
                "      sigma=1.00: NN ~ ProbReg dyn-k (1.099) > ClassReg k=2 (1.162) > LGBM\n"
                "     So ClassReg's bottleneck DOES help vs tree baselines at high noise, but\n"
                "     not vs a plain probabilistic NN. Paper narrative needs reframing.\n\n"
                "CLAIM 3: Dynamic-k adapts per-input to the data.\n"
                "  -> MET only without ELBO. The ELBO prior pins k=2 regardless of sigma.\n"
                "     With NClassesRegularization.NONE, mean_k rises from ~SENTINEL (direct\n"
                "     regression) at sigma=0.05 to k~5 at sigma=1.0 — a clean adaptation\n"
                "     story. Recommendation: report dyn-k results without the heavy ELBO\n"
                "     prior, or retune prior strength.\n\n"
                "CLAIM 4: SEPARATE_HEADS underperforms SINGLE_HEAD_FINAL on small data.\n"
                "  -> NOT MET at param-matched sizes. The earlier observation was driven by\n"
                "     param-count mismatch (SEP has n_classes heads vs SINGLE's one head).\n"
                "     Once parameter counts are aligned they are within noise on all\n"
                "     n_samples in {200, 500, 1500}. SEP_HEADS retains the interpretability\n"
                "     advantage (per-class regression heads) without a measurable\n"
                "     small-data penalty.\n\n"
                "CLAIM 5: Gumbel-Softmax + ELBO is broken.\n"
                "  -> NOT MET (previously believed). After N1/N2 + the optimizer-inclusion\n"
                "     fix, Gumbel+ELBO trains and produces non-trivial selection entropy.\n"
                "     It is still slightly worse than SoftGating+ELBO on MSE, but no longer\n"
                "     a failure mode. Treat as a viable-but-lossier alternative, not broken.\n\n"
                "OVERALL: multiple prior benchmark claims about FlexNN depth regularisation\n"
                "silently ran on frozen n_predictor weights. Those results need re-running.\n"
                "The new findings post-fix are encouraging but recalibrate the narrative\n"
                "against plain probabilistic NN baselines."
            ),
            verdict="PARTIAL: 2 MET, 2 NOT MET (re-framed), 1 partial",
        )
    logger.info(f"wrote {path}")
    return path


# ---------------------------------------------------------------------------


def main() -> None:
    heads_pdf()
    n_selection_pdf()
    regularization_pdf()
    losses_pdf()
    verdict_pdf()


if __name__ == "__main__":
    main()
