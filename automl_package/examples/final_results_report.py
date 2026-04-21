"""Aggregate every sweep result into a single final_report.pdf.

Designed for post-``run_all_sweeps.sh`` consumption. Tolerates missing outputs:
any sweep that did not run (or crashed) is reported with a status page but
does not fail the aggregation.

Output (under ``final_results_report/``):
  final_report.pdf  — one page per sweep: description + headline table + verdict
  status.json       — machine-readable roll-up of per-sweep status + best rows

Usage:
  python -m automl_package.examples.final_results_report \\
      --status-file automl_package/examples/sweep_runs/<stamp>/status.tsv \\
      --out-dir automl_package/examples/final_results_report
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

EXAMPLES = Path(__file__).parent


@dataclass
class SweepResult:
    name: str
    title: str
    description: str
    results_dir: Path
    csv_paths: list[Path]
    pdf_paths: list[Path] = field(default_factory=list)
    status: str = "UNKNOWN"       # OK / FAIL / TIMEOUT / SKIP / MISSING
    elapsed_s: int | None = None
    headline_df: pd.DataFrame | None = None
    findings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# PDF helpers (mirrors automl_package.examples.investigation_pdf_reports)
# ---------------------------------------------------------------------------


def _text_page(pdf: PdfPages, title: str, body: str, *, verdict: str | None = None) -> None:
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.05, 0.95, title, fontsize=16, weight="bold")
    fig.text(0.05, 0.91, body, fontsize=10, verticalalignment="top", wrap=True, family="sans-serif")
    if verdict is not None:
        key = verdict.split(":")[0].strip().upper()
        color = {"OK": "green", "PARTIAL": "orange", "FAIL": "red", "TIMEOUT": "red", "SKIP": "gray", "MISSING": "gray"}.get(key, "black")
        fig.text(0.05, 0.05, f"Status: {verdict}", fontsize=12, weight="bold", color=color)
    pdf.savefig(fig)
    plt.close(fig)


def _table_page(pdf: PdfPages, title: str, df: pd.DataFrame, float_fmt: str = ".4f", max_rows: int = 30) -> None:
    if df is None or df.empty:
        return
    data = df.head(max_rows).copy()
    fig, ax = plt.subplots(figsize=(11, 1.0 + 0.3 * (len(data) + 2)))
    ax.axis("off")
    ax.set_title(title, fontsize=12, pad=14)
    for col in data.columns:
        if pd.api.types.is_float_dtype(data[col]):
            data[col] = data[col].map(lambda v: format(v, float_fmt) if pd.notna(v) else "--")
    data = data.astype(str)
    table = ax.table(cellText=data.values, colLabels=data.columns, loc="center", cellLoc="right")
    for (r, c), cell in table.get_celld().items():
        if c == 0:
            cell.set_text_props(ha="left")
        if r == 0:
            cell.set_text_props(weight="bold")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.25)
    pdf.savefig(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Sweep definitions + headline extractors
# ---------------------------------------------------------------------------


def _read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not read %s: %s", path, exc)
        return None


def _headline_identifiability(results_dir: Path) -> tuple[pd.DataFrame | None, list[str]]:
    df = _read_csv(results_dir / "summary.csv")
    if df is None:
        return None, []
    findings: list[str] = []
    if {"dataset", "cell", "mse_mean"}.issubset(df.columns):
        nll_col = "nll_mean" if "nll_mean" in df.columns else "nll_own_mean"
        cols = ["dataset", "cell", "mse_mean"] + ([nll_col] if nll_col in df.columns else [])
        best = df.loc[df.groupby("dataset")["mse_mean"].idxmin()][cols].reset_index(drop=True)
        for _, row in best.iterrows():
            findings.append(f"best cell on {row['dataset']}: {row['cell']} (MSE={row['mse_mean']:.3f})")
        return best, findings
    return df, findings


def _headline_classreg_k(results_dir: Path) -> tuple[pd.DataFrame | None, list[str]]:
    df = _read_csv(results_dir / "k_sweep.csv")
    if df is None or "mse" not in df.columns:
        return df, []
    best = df.loc[df.groupby("dataset")["mse"].idxmin()].reset_index(drop=True)
    findings = [f"best k on {r['dataset']}: k={int(r['k'])} (MSE={r['mse']:.3f})" for _, r in best.iterrows()]
    return best, findings


def _headline_probreg_ablation(results_dir: Path) -> tuple[pd.DataFrame | None, list[str]]:
    top3 = _read_csv(results_dir / "top3_per_dataset.csv")
    if top3 is None:
        top3 = _read_csv(results_dir / "probreg_ablation.csv")
        if top3 is None or "mse" not in top3.columns:
            return top3, []
        top3 = top3.sort_values("mse").groupby("dataset").head(3).reset_index(drop=True)
    findings: list[str] = []
    if "dataset" in top3.columns and "strategy" in top3.columns:
        for ds, grp in top3.groupby("dataset"):
            best = grp.iloc[0]
            findings.append(f"{ds}: best = {best.get('strategy', '?')}/{best.get('selection', '?')}/{best.get('loss', '?')} (MSE={best.get('mse', float('nan')):.3f})")
    return top3, findings


def _headline_flex_nn_ablation(results_dir: Path) -> tuple[pd.DataFrame | None, list[str]]:
    for fname in ("top3_per_dataset.csv", "flex_nn_ablation.csv", "ablation.csv"):
        df = _read_csv(results_dir / fname)
        if df is not None:
            if "mse" in df.columns and "dataset" in df.columns:
                df = df.sort_values(["dataset", "mse"]).groupby("dataset").head(3).reset_index(drop=True)
            return df, []
    return None, []


def _headline_full_benchmark(results_dir: Path) -> tuple[pd.DataFrame | None, list[str]]:
    # full_benchmark currently writes markdown/txt tables, not CSV; we link the PDF/MD
    # into the final report and surface a marker row so the aggregator page is non-empty.
    report = results_dir / "REPORT.md"
    if report.exists():
        return pd.DataFrame({"artifact": ["REPORT.md"], "path": [str(report)]}), [f"see {report}"]
    return None, []


def _headline_ablation_study(results_dir: Path) -> tuple[pd.DataFrame | None, list[str]]:
    for fname in ("ablation_study.csv", "results.csv"):
        df = _read_csv(results_dir / fname)
        if df is not None:
            return df, []
    return None, []


def _headline_hpo(results_dir: Path) -> tuple[pd.DataFrame | None, list[str]]:
    df = _read_csv(results_dir / "hpo_results.csv")
    if df is None:
        return None, []
    cols = [c for c in ("dataset", "model", "mse", "nll", "crps") if c in df.columns]
    view = df[cols] if cols else df
    findings: list[str] = []
    if {"dataset", "model", "mse"}.issubset(df.columns):
        best = df.loc[df.groupby("dataset")["mse"].idxmin()]
        findings = [f"HPO winner on {r['dataset']}: {r['model']} (MSE={r['mse']:.3f})" for _, r in best.iterrows()]
    return view, findings


SWEEPS: list[dict[str, Any]] = [
    {
        "name": "identifiability",
        "title": "ProbReg identifiability (8-cell matrix)",
        "dir": EXAMPLES / "probreg_identifiability_results",
        "description": (
            "Tests three orthogonal fixes (MDN NLL, CE_STOP_GRAD, anchored heads) across "
            "4 toy datasets × k ∈ {3, 5} × 3 seeds. Cells A-H labelled in the sweep docstring. "
            "Headline = best cell per dataset by mean MSE."
        ),
        "extractor": _headline_identifiability,
        "csv": ["results.csv", "summary.csv"],
    },
    {
        "name": "classreg_k",
        "title": "ClassifierRegression k-sweep",
        "dir": EXAMPLES / "classreg_k_sweep_results",
        "description": "k ∈ {2, 3, 5, 7, 10} across toy datasets. Headline = argmin MSE per dataset.",
        "extractor": _headline_classreg_k,
        "csv": ["k_sweep.csv"],
    },
    {
        "name": "full_benchmark",
        "title": "Full benchmark (all models × all toy+UCI)",
        "dir": EXAMPLES / "full_benchmark_results",
        "description": (
            "15 models on 4 toy + 4 UCI datasets. Metrics: MSE, NLL, CRPS, ECE, PICP@95, "
            "MPIW@95, sharpness. Human-readable report in REPORT.md alongside the PDF."
        ),
        "extractor": _headline_full_benchmark,
        "csv": ["REPORT.md"],
    },
    {
        "name": "probreg_ablation",
        "title": "ProbReg ablation (Paper A primary table)",
        "dir": EXAMPLES / "probreg_ablation_results",
        "description": (
            "Cross-product over regression_strategy × loss_type × target_transform × "
            "opt_strategy × dynamic-k selection × n_classes_regularization. Headline = "
            "top-3 configs per dataset by MSE."
        ),
        "extractor": _headline_probreg_ablation,
        "csv": ["probreg_ablation.csv", "top3_per_dataset.csv"],
    },
    {
        "name": "flex_nn_ablation",
        "title": "FlexibleNN ablation (Paper B primary table)",
        "dir": EXAMPLES / "flex_nn_ablation_results",
        "description": (
            "max_hidden_layers × depth_regularization × layer_selection × weights × "
            "inference × uncertainty. Headline = top-3 per dataset."
        ),
        "extractor": _headline_flex_nn_ablation,
        "csv": ["flex_nn_ablation.csv", "top3_per_dataset.csv"],
    },
    {
        "name": "ablation_study",
        "title": "Ablation study (smaller probe)",
        "dir": EXAMPLES / "ablation_study_results",
        "description": "Compact ablation over FlexNN and ProbReg configurations on heteroscedastic + exponential.",
        "extractor": _headline_ablation_study,
        "csv": ["ablation_study.csv", "results.csv"],
    },
    {
        "name": "hpo",
        "title": "HPO sweep (Optuna, tuned UCI baselines)",
        "dir": EXAMPLES / "hpo_sweep_results",
        "description": "N=50 Optuna trials per (model, dataset). Default dataset: uci-yacht. ALL_DATASETS=1 adds uci-california (3k subsample).",
        "extractor": _headline_hpo,
        "csv": ["hpo_results.csv", "hpo_results.json"],
    },
]


# ---------------------------------------------------------------------------
# Status file parsing
# ---------------------------------------------------------------------------


def load_status(status_file: Path | None) -> dict[str, dict[str, Any]]:
    """Return {sweep_name: {status, elapsed_s, log_path}} from status.tsv."""
    if status_file is None or not status_file.exists():
        return {}
    df = pd.read_csv(status_file, sep="\t")
    out: dict[str, dict[str, Any]] = {}
    terminal = {"OK", "FAIL", "TIMEOUT", "SKIP"}
    for _, row in df.iterrows():
        if row["event"] in terminal:
            out[row["sweep"]] = {
                "status": row["event"],
                "elapsed_s": int(row["elapsed_s"]) if pd.notna(row["elapsed_s"]) and str(row["elapsed_s"]).strip() else None,
                "log_path": row.get("log_path", ""),
            }
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_report(status_file: Path | None, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    status_map = load_status(status_file)
    results: list[SweepResult] = []

    for spec in SWEEPS:
        sweep_dir: Path = spec["dir"]
        st = status_map.get(spec["name"], {})
        status = st.get("status") or ("MISSING" if not sweep_dir.exists() else "OK")
        try:
            headline_df, findings = spec["extractor"](sweep_dir)
        except Exception as exc:  # noqa: BLE001
            logger.exception("extractor for %s failed", spec["name"])
            headline_df, findings = None, [f"extractor error: {exc}"]
        results.append(SweepResult(
            name=spec["name"],
            title=spec["title"],
            description=spec["description"],
            results_dir=sweep_dir,
            csv_paths=[sweep_dir / c for c in spec["csv"]],
            pdf_paths=sorted(sweep_dir.glob("*.pdf")) if sweep_dir.exists() else [],
            status=status,
            elapsed_s=st.get("elapsed_s"),
            headline_df=headline_df,
            findings=findings,
        ))

    pdf_path = out_dir / "final_report.pdf"
    with PdfPages(pdf_path) as pdf:
        _write_cover(pdf, results)
        for r in results:
            _write_sweep_pages(pdf, r)

    status_json = {
        "sweeps": [
            {
                "name": r.name, "status": r.status, "elapsed_s": r.elapsed_s,
                "results_dir": str(r.results_dir), "per_sweep_pdfs": [str(p) for p in r.pdf_paths],
                "findings": r.findings,
            }
            for r in results
        ],
        "pdf": str(pdf_path),
    }
    (out_dir / "status.json").write_text(json.dumps(status_json, indent=2))
    logger.info("wrote %s and status.json", pdf_path)


def _write_cover(pdf: PdfPages, results: list[SweepResult]) -> None:
    lines = ["Sweep status roll-up:", ""]
    for r in results:
        mins = f"{r.elapsed_s // 60}m" if r.elapsed_s else "--"
        lines.append(f"  [{r.status:<7}] {r.name:<18} ({mins})  →  {r.results_dir}")
    lines += ["", "Each per-sweep page below links back to its own CSV/PDF artefacts.", "Missing sweeps render a status-only page (no data to plot)."]
    _text_page(pdf, "Final sweep roll-up", "\n".join(lines))


def _write_sweep_pages(pdf: PdfPages, r: SweepResult) -> None:
    body_lines = [r.description, ""]
    body_lines.append(f"Results dir: {r.results_dir}")
    if r.pdf_paths:
        body_lines.append("Per-sweep PDFs:")
        body_lines.extend(f"  - {p}" for p in r.pdf_paths)
    if r.findings:
        body_lines += ["", "Key findings:"]
        body_lines.extend(f"  - {f}" for f in r.findings)
    verdict = r.status if not r.elapsed_s else f"{r.status} ({r.elapsed_s // 60}m)"
    _text_page(pdf, r.title, "\n".join(body_lines), verdict=verdict)
    if r.headline_df is not None and not r.headline_df.empty:
        _table_page(pdf, f"{r.title} — headline", r.headline_df)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--status-file", type=Path, default=None, help="sweep_runs/<stamp>/status.tsv from run_all_sweeps.sh")
    parser.add_argument("--out-dir", type=Path, default=EXAMPLES / "final_results_report")
    args = parser.parse_args()
    build_report(args.status_file, args.out_dir)


if __name__ == "__main__":
    main()
