"""Aggregate results from all Phase 9 benchmark output directories.

Reads CSVs produced by the batch runs and emits a single Markdown report at
``examples/REPORT_phase9.md`` summarising headline findings for each task.

Designed to be idempotent: missing files are skipped gracefully so this can be
run repeatedly as more results roll in.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

EXAMPLES = Path(__file__).parent
OUT_PATH = EXAMPLES / "REPORT_phase9.md"


def _maybe_read(path: Path) -> pd.DataFrame | None:
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:  # noqa: BLE001
            return None
    return None


def main() -> None:
    sections: list[str] = ["# Phase 9 autonomous results\n"]

    # Noise robustness
    nr_path = EXAMPLES / "noise_robustness_results" / "results.csv"
    nr_df = _maybe_read(nr_path)
    if nr_df is not None:
        sections.append("## I4 — Noise-robustness benchmark\n")
        sections.append("See `noise_robustness_results/`. Key table:\n")
        best = nr_df.dropna(subset=["mse"]).sort_values(["sigma", "mse"]).groupby("sigma").head(3)
        sections.append(best.to_csv(index=False, sep="|", lineterminator="\n"))
        sections.append("")

    # Head structure diagnostic
    hs_path = EXAMPLES / "head_structure_results" / "head_structure.csv"
    hs_df = _maybe_read(hs_path)
    if hs_df is not None:
        sections.append("## I2 — Regression-head structure diagnostic\n")
        n_failed = int((~hs_df["passed"]).sum())
        sections.append(f"{n_failed}/{len(hs_df)} configs failed structural checks.")
        failed_df = hs_df[~hs_df["passed"]][["dataset", "config", "mirror_ok", "middle_flat_ok", "mean_sep_ok", "mse"]]
        if len(failed_df):
            sections.append(failed_df.to_csv(index=False, sep="|", lineterminator="\n"))
        sections.append("")

    # Depth viz correlations
    for name in ("piecewise", "tunable_complexity"):
        corr_path = EXAMPLES / "flex_nn_depth_viz_results" / f"{name}_correlations.csv"
        df = _maybe_read(corr_path)
        if df is not None:
            sections.append(f"## I3 — Depth correlation ({name})\n")
            sections.append(df.to_csv(index=False, sep="|", lineterminator="\n"))
            sections.append("")

    # Gumbel+ELBO retest
    ge_path = EXAMPLES / "gumbel_elbo_retest_results" / "gumbel_elbo_retest.csv"
    ge_df = _maybe_read(ge_path)
    if ge_df is not None:
        sections.append("## I8 — Gumbel+ELBO retest\n")
        sections.append(ge_df.to_csv(index=False, sep="|", lineterminator="\n"))
        sections.append("")

    # SEP_HEADS investigation
    sh_path = EXAMPLES / "sep_heads_investigation" / "param_matched_comparison.csv"
    sh_df = _maybe_read(sh_path)
    if sh_df is not None:
        sections.append("## I1 — SEP_HEADS vs SINGLE_HEAD_FINAL_OUTPUT\n")
        summary = sh_df.groupby(["strategy", "n_samples"])["mse"].mean().reset_index()
        sections.append(summary.to_csv(index=False, sep="|", lineterminator="\n"))
        sections.append("")

    # FlexNN ablation
    fa_path = EXAMPLES / "flex_nn_ablation_results" / "flex_nn_ablation.csv"
    fa_df = _maybe_read(fa_path)
    if fa_df is not None:
        sections.append("## #15 FlexibleNN ablation (subsampled grid)\n")
        if "mse" in fa_df.columns:
            best = fa_df.dropna(subset=["mse"]).sort_values(["dataset", "mse"]).groupby("dataset").head(3)
            sections.append(best.to_csv(index=False, sep="|", lineterminator="\n"))
        sections.append("")

    # ProbReg ablation
    pa_path = EXAMPLES / "probreg_ablation_results" / "probreg_ablation.csv"
    pa_df = _maybe_read(pa_path)
    if pa_df is not None:
        sections.append("## #14 ProbReg ablation (subsampled grid)\n")
        if "nll" in pa_df.columns:
            best = pa_df.dropna(subset=["nll"]).sort_values(["dataset", "nll"]).groupby("dataset").head(3)
            sections.append(best.to_csv(index=False, sep="|", lineterminator="\n"))
        sections.append("")

    # Multi-seed
    ms_path = EXAMPLES / "multi_seed_results" / "summary_mean_std.csv"
    ms_df = _maybe_read(ms_path)
    if ms_df is not None:
        sections.append("## #13 Multi-seed (5 seeds) summary\n")
        sections.append(ms_df.to_csv(index=False, sep="|", lineterminator="\n"))
        sections.append("")

    OUT_PATH.write_text("\n\n".join(sections))
    print(f"Wrote {OUT_PATH} ({OUT_PATH.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
