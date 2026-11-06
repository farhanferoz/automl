"""Builds the reports from whatever records exist for a plan file: CSV, Markdown and one static HTML page per fingerprint.

    python -m automl_package.experiments.report <cells.jsonl> [--eval-set selection|test] [--out DIR]

COMPARISON studies get the one table (design section 5): a row per variant, a column per toy x size; a cell is the mean over
seeds of the paired change in noise-free excess against the reference variant on the same draws, as a percentage of the
reference's noise-free excess (negative = better), beside it the min and max over seeds; greyed when the mean is smaller than
the reference's seed-to-seed noise at that toy and size (relative sd over seeds, from the 5a size-sweep records; no 5a record
for that toy and size means the cell is not shaded and is marked "n/a"; when the card's own environment has none, the noise of the
environment with the most 5a records there is used and the cell is labelled "(noise from X)", column ``noise_fp``). No variant is picked automatically. SIZE_SWEEP studies
(card 5a) get the reference predictors against their expected values, the validation trajectories and the excess against n.

A pair is only ever made inside one environment fingerprint: records of other fingerprints get their own report. The exception is a
variant the study lists in ``report.environment_independent`` (a deterministic fit with no search): where an environment has no record
of it for a cell, another environment's record stands in (`environment_view`); its pairs still need the reference's draws (hash), a
pair whose draws differ across environments is dropped and counted, and the table and figures name the environment it came from.

Reused: ``cell.all_record_paths`` / ``cell.load_record`` for the records and their layout, ``study.read_plan`` for the plan,
``Toy.dir_name`` for column names, the curve PNGs written by ``curves.write_curves`` (linked, not redrawn).
"""

from __future__ import annotations

import argparse
import html
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from automl_package.experiments.cell import all_record_paths, load_record
from automl_package.experiments.config import CellSpec, EvalSet, PredictorKind, ReportKind
from automl_package.experiments.study import REPORTS_DIR, layout_root, read_plan
from automl_package.experiments.toys import Toy

KEY = ["variant", "toy", "n", "seed", "fp"]
NOT_AVAILABLE = "n/a"
NOISE_COLUMNS = ["toy", "n", "fp", "noise", "n_records"]
GREY_STYLE = "background:#ddd;color:#666"


def load_records(cells: list[CellSpec], eval_set: EvalSet) -> pd.DataFrame:
    """One row per record found for the plan's cells, any fingerprint; ``record`` holds the full JSON."""
    rows, seen = [], set()
    for cell in cells:
        for path in all_record_paths(cell):
            if path in seen:  # a record in several cards' plans (the shared reference) counts once
                continue
            seen.add(path)
            rec = load_record(path)
            metrics = rec["metrics"][eval_set.value]
            rows.append(
                {
                    "variant": cell.variant,
                    "toy": Toy.dir_name(cell.toy),
                    "n": cell.n_train,
                    "seed": cell.seed,
                    "fp": rec["fingerprint_id"],
                    "nfe": metrics["noise_free_excess"],
                    "nfe_se": metrics["noise_free_excess_se"],
                    "excess": metrics["excess"],
                    "excess_se": metrics.get("excess_se", np.nan),
                    "expected": metrics.get("expected_noise_free_excess", np.nan),
                    "path": str(path),
                    "trajectory": (rec.get("convergence") or {}).get("trajectory"),
                    "kind": cell.model.kind.value,
                    "data": json.dumps(rec["data_hashes"], sort_keys=True),
                }
            )
    return pd.DataFrame(rows, columns=[*KEY, "nfe", "nfe_se", "excess", "excess_se", "expected", "path", "trajectory", "kind", "data"])


def seed_noise(cells: list[CellSpec], eval_set: EvalSet) -> pd.DataFrame:
    """Relative seed-to-seed sd of the noise-free excess of the reference model (from the 5a size-sweep records), per toy, size and fingerprint, with the record count."""
    report = cells[0].report
    if report.noise_results_subdir is None or report.noise_variant is None:
        return pd.DataFrame(columns=NOISE_COLUMNS)
    root = Path(cells[0].results_dir).parent / report.noise_results_subdir / report.noise_variant
    rows = []
    for path in sorted(root.glob("*/n*/s*.json")) if root.exists() else []:
        rec = load_record(path)
        rows.append({"toy": path.parents[1].name, "n": int(path.parent.name[1:]), "fp": rec["fingerprint_id"], "nfe": rec["metrics"][eval_set.value]["noise_free_excess"]})
    if not rows:
        return pd.DataFrame(columns=NOISE_COLUMNS)
    grouped = pd.DataFrame(rows).groupby(["toy", "n", "fp"])["nfe"]
    return pd.concat([(grouped.std(ddof=1) / grouped.mean()).rename("noise"), grouped.count().rename("n_records")], axis=1).reset_index()


def environment_view(records: pd.DataFrame, fp: str, environment_independent: list[str]) -> pd.DataFrame:
    """The records environment ``fp`` is reported from: its own, plus stand-ins of the environment-independent variants.

    For each variant in ``environment_independent``, another environment's record of every cell ``fp`` lacks (the lowest fingerprint id
    when several have one). ``record_fp`` names where a row was written; ``fp`` is ``fp`` on every row, so the per-environment tables and
    figures take the stand-ins as they are.
    """
    records = records.assign(record_fp=records["fp"])
    own = records[records["fp"] == fp]
    others = records[(records["fp"] != fp) & records["variant"].isin(environment_independent)]
    cell = ["variant", "toy", "n", "seed"]
    stand_in = others[~others.set_index(cell).index.isin(own.set_index(cell).index)].sort_values("record_fp").drop_duplicates(cell)
    return pd.concat([own, stand_in.assign(fp=fp)], ignore_index=True)


def comparison_table(records: pd.DataFrame, noise: pd.DataFrame, reference: str, fp: str) -> pd.DataFrame:
    """Long table, one row per (variant, toy, n): mean/min/max paired relative change (%), seeds paired, noise, greyed.

    ``records`` may be an `environment_view`: a pair with a stand-in side whose draws differ from the other side's is dropped and counted
    (``pairs_dropped``); ``stand_in_from`` names the environments of the stand-ins used. Within one environment different draws raise.
    """
    records = records if "record_fp" in records else records.assign(record_fp=records["fp"])
    own = records[records["fp"] == fp]
    ref_columns = {"nfe": "nfe_ref", "data": "data_ref", "record_fp": "record_fp_ref"}
    ref = own[own["variant"] == reference][["toy", "n", "seed", *ref_columns]].rename(columns=ref_columns)
    paired = own[own["variant"] != reference].merge(ref, on=["toy", "n", "seed"])
    crossing = (paired["record_fp"] != fp) | (paired["record_fp_ref"] != fp)
    mismatched = paired["data"] != paired["data_ref"]
    if (mismatched & ~crossing).any():  # the slot-in guarantee: a pair is the same four draws, checked by their hashes
        within = paired[mismatched & ~crossing]
        raise ValueError(f"{len(within)} pair(s) were drawn from different data than the reference, e.g. {within['path'].iloc[0]}")
    paired["dropped"] = mismatched & crossing  # a stand-in drawn differently on another machine (the last bits of a generator)
    paired["stand_in"] = np.where(crossing, np.where(paired["record_fp"] != fp, paired["record_fp"], paired["record_fp_ref"]), "")
    kept = paired[~paired["dropped"]].copy()
    kept["change_pct"] = np.where(kept["nfe_ref"] > 0, 100.0 * (kept["nfe"] - kept["nfe_ref"]) / kept["nfe_ref"], np.nan)
    table = kept.groupby(["variant", "toy", "n"])["change_pct"].agg(mean_pct="mean", min_pct="min", max_pct="max", n_seeds="count").reset_index()
    origin = paired.groupby(["variant", "toy", "n"]).agg(pairs_dropped=("dropped", "sum"), stand_in_from=("stand_in", lambda s: ",".join(sorted(set(s) - {""})))).reset_index()
    table = origin.merge(table, on=["variant", "toy", "n"], how="left").fillna({"n_seeds": 0}).astype({"n_seeds": int})  # all pairs dropped: kept, 0 seeds
    table = table.merge(_noise_for(noise, fp), on=["toy", "n"], how="left")
    table["fp"] = fp
    table["noise_pct"] = 100.0 * pd.to_numeric(table["noise"])  # an empty noise frame is object-typed
    table["greyed"] = table["mean_pct"].abs() < table["noise_pct"].fillna(-1.0)  # no known noise: never shaded
    table["noise_known"] = table["noise_pct"].notna()
    return table.drop(columns="noise")


def _noise_for(noise: pd.DataFrame, fp: str) -> pd.DataFrame:
    """Per (toy, n): the noise of fingerprint ``fp``, else that of the fingerprint with the most 5a records there; ``noise_fp`` says which (a borrowed margin is labelled)."""
    own = noise[noise["fp"] == fp]
    borrowed = noise.sort_values(["n_records", "fp"]).groupby(["toy", "n"]).tail(1)
    chosen = pd.concat([own, borrowed[~borrowed.set_index(["toy", "n"]).index.isin(own.set_index(["toy", "n"]).index)]])
    return chosen.rename(columns={"fp": "noise_fp"})[["toy", "n", "noise", "noise_fp"]]


def _markdown(frame: pd.DataFrame, index: bool = True) -> str:
    """A pipe table (no ``tabulate`` dependency)."""
    flat = frame.reset_index() if index else frame
    lines = ["| " + " | ".join(map(str, flat.columns)) + " |", "|" + "---|" * len(flat.columns)]
    lines += ["| " + " | ".join(f"{v:.6g}" if isinstance(v, float) else str(v) for v in row) + " |" for row in flat.itertuples(index=False)]
    return "\n".join(lines)


def _cell_text(row: pd.Series) -> str:
    text = f"{row['mean_pct']:+.1f}% [{row['min_pct']:+.1f}, {row['max_pct']:+.1f}]" if row.get("n_seeds", 1) else "no pair"
    if row.get("stand_in_from"):
        text += f" (record from {row['stand_in_from']})"
    if row.get("pairs_dropped"):
        text += f" ({int(row['pairs_dropped'])} dropped: draws differ)"
    if not row["noise_known"]:
        return f"{text} {NOT_AVAILABLE}"
    return text if row["noise_fp"] == row["fp"] else f"{text} (noise from {row['noise_fp']})"


def _pivot(table: pd.DataFrame) -> pd.DataFrame:
    text = table.assign(text=table.apply(_cell_text, axis=1), column=table["toy"] + " n=" + table["n"].astype(str))
    return text.pivot(index="variant", columns="column", values="text")


def _html_table(table: pd.DataFrame) -> str:
    grey = table.assign(column=table["toy"] + " n=" + table["n"].astype(str)).set_index(["variant", "column"])["greyed"]
    pivot = _pivot(table)
    head = "".join(f"<th>{html.escape(c)}</th>" for c in pivot.columns)
    rows = []
    for variant, line in pivot.iterrows():
        cells = "".join(
            f"<td style='{GREY_STYLE if grey.get((variant, col), False) else ''}'>{html.escape(v) if isinstance(v, str) else ''}</td>" for col, v in line.items()
        )
        rows.append(f"<tr><th>{html.escape(str(variant))}</th>{cells}</tr>")
    return f"<table border='1' cellspacing='0' cellpadding='4'><tr><th>variant</th>{head}</tr>{''.join(rows)}</table>"


def _figure_links(cells: list[CellSpec], out_dir: Path) -> str:
    pngs = sorted({p.with_suffix(".png") for c in cells for p in all_record_paths(c) if p.with_suffix(".png").exists()})
    items = "".join(f"<li><a href='{html.escape(os.path.relpath(p, out_dir))}'>{html.escape(str(p.relative_to(Path(cells[0].results_dir))))}</a></li>" for p in pngs)
    return f"<h2>Curve figures</h2><ul>{items}</ul>" if pngs else ""


def card_label(cells: list[CellSpec]) -> tuple[str, str]:
    """(name, title prefix) of the cards in ``cells``, in plan order.

    One card is "Card 5c1"; several stacked against one reference (the one-table look across cards) are "Cards 5c1+5c2".
    """
    cards = list(dict.fromkeys(c.card for c in cells))
    label = "+".join(cards)
    return label, f"Card{'s' if len(cards) > 1 else ''} {label}"


def write_comparison(cells: list[CellSpec], eval_set: EvalSet, out_dir: Path) -> list[Path]:
    """The one table, per fingerprint present among the reference's records."""
    records = load_records(cells, eval_set)
    reference = cells[0].report.reference_variant
    noise = seed_noise(cells, eval_set)
    written = []
    for fp in sorted(records.loc[records["variant"] == reference, "fp"].unique()):
        table = comparison_table(environment_view(records, fp, cells[0].report.environment_independent), noise, reference, fp)
        stem = out_dir / f"comparison_{card_label(cells)[0]}_{eval_set.value}_{fp}"
        table.to_csv(stem.with_suffix(".csv"), index=False)
        ref_abs = records[(records["fp"] == fp) & (records["variant"] == reference)].groupby(["toy", "n"])["nfe"].mean().rename("reference noise-free excess (mean over seeds)")
        title = f"{card_label(cells)[1]}: change in noise-free excess against '{reference}', {eval_set.value} set, environment {fp}"
        legend = (
            "Cell: mean over seeds of the paired change, as % of the reference's noise-free excess (negative = better), with [min, max] over seeds. "
            "Grey: the mean is smaller than the reference's seed-to-seed noise. n/a: no 5a record for that toy and size, so not shaded. "
            "(noise from X): no 5a record under this environment, so the margin is borrowed from environment X, the one with the most 5a records there. "
            "(record from X): a deterministic variant's record written in environment X stands in (DECISIONS 2026-11-06); (k dropped: draws differ): "
            "k such pairs had draws that differ across the machines and are left out. No variant is picked automatically."
        )
        stem.with_suffix(".md").write_text(f"# {title}\n\n{legend}\n\n{_markdown(_pivot(table))}\n\n{_markdown(ref_abs.to_frame())}\n")
        stem.with_suffix(".html").write_text(
            f"<!doctype html><meta charset='utf-8'><title>{html.escape(title)}</title><h1>{html.escape(title)}</h1><p>{html.escape(legend)}</p>{_html_table(table)}"
            f"<h2>Reference</h2>{ref_abs.to_frame().to_html()}{_figure_links(cells, out_dir)}"
        )
        written += [stem.with_suffix(s) for s in (".csv", ".md", ".html")]
    return written


def _reference_table(records: pd.DataFrame, fp: str) -> pd.DataFrame:
    """Reference predictors through the full scoring path: test MSE minus the toy's floor against the expected excess.

    The noise-free excess of a reference predictor equals its expectation by construction, so it cannot test anything; the
    MSE carries the label noise, so (MSE - floor) - expected, in units of the MSE's standard error on the draw, checks the
    generator, the stated true mean and the floor together (card 5a part a1).
    """
    refs = records[(records["fp"] == fp) & (records["kind"] != PredictorKind.AUTOML.value)]
    grouped = refs.groupby(["variant", "toy", "n"])
    out = grouped.agg(excess_mse_minus_floor=("excess", "mean"), expected_excess=("expected", "mean"), n_draws=("seed", "count")).reset_index()
    out["standard_error"] = grouped["excess_se"].apply(lambda s: float(np.sqrt((s**2).sum()) / len(s))).to_numpy()
    out["difference"] = out["excess_mse_minus_floor"] - out["expected_excess"]
    out["difference_in_standard_errors"] = out["difference"] / out["standard_error"]
    return out


def _plot_trajectories(records: pd.DataFrame, fp: str, out_dir: Path) -> list[Path]:
    """Validation loss against epoch, one figure per toy, one panel per (variant, size), one line per seed."""
    paths = []
    fitted = records[(records["fp"] == fp) & records["trajectory"].notna()]
    for toy, group in fitted.groupby("toy"):
        panels = sorted(group.groupby(["variant", "n"]).groups)
        fig, axes = plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 3), squeeze=False)
        for ax, (variant, n) in zip(axes[0], panels, strict=True):
            for _, row in group[(group["variant"] == variant) & (group["n"] == n)].iterrows():
                epochs, loss = zip(*row["trajectory"], strict=True)
                ax.plot(epochs, loss, label=f"seed {row['seed']}")
            ax.set_title(f"{variant}, n={n}", fontsize=8)
            ax.set_xlabel("epoch")
            ax.set_ylabel("held-out fit term")
        axes[0][0].legend(fontsize=6)
        fig.suptitle(f"{toy}: validation trajectories", fontsize=9)
        fig.tight_layout()
        paths.append(out_dir / f"trajectories_{toy}_{fp}.png")
        fig.savefig(paths[-1], dpi=100)
        plt.close(fig)
    return paths


def _plot_excess_against_n(records: pd.DataFrame, fp: str, out_dir: Path) -> list[Path]:
    """Noise-free excess against n, one figure per toy, one line per variant (bars: min and max over seeds)."""
    paths = []
    fitted = records[(records["fp"] == fp) & (records["kind"] == PredictorKind.AUTOML.value)]
    for toy, group in fitted.groupby("toy"):
        fig, ax = plt.subplots(figsize=(5, 3.5))
        for variant, line in group.groupby("variant"):
            stats = line.groupby("n")["nfe"].agg(["mean", "min", "max"])
            origin = sorted(set(line.get("record_fp", line["fp"])) - {fp})
            label = f"{variant} (record from {','.join(origin)})" if origin else variant
            ax.errorbar(stats.index, stats["mean"], yerr=[stats["mean"] - stats["min"], stats["max"] - stats["mean"]], marker="o", capsize=2, label=label)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("n (training size)")
        ax.set_ylabel("noise-free excess (bars: min, max over seeds)")
        ax.set_title(toy, fontsize=9)
        ax.legend(fontsize=7)
        fig.tight_layout()
        paths.append(out_dir / f"excess_against_n_{toy}_{fp}.png")
        fig.savefig(paths[-1], dpi=100)
        plt.close(fig)
    return paths


def write_size_sweep(cells: list[CellSpec], eval_set: EvalSet, out_dir: Path) -> list[Path]:
    """The 5a report: reference predictors against expectation, trajectories, excess against n; per fingerprint."""
    loaded = load_records(cells, eval_set)
    portable = cells[0].report.environment_independent
    owners = loaded.loc[~loaded["variant"].isin(portable), "fp"].unique()  # an environment holding only stand-ins gets no page of its own
    written = []
    for fp in sorted(owners if len(owners) else loaded["fp"].unique()):
        records = environment_view(loaded, fp, portable)
        stem = out_dir / f"size_sweep_{card_label(cells)[0]}_{eval_set.value}_{fp}"
        refs = _reference_table(records, fp)
        refs.to_csv(stem.with_suffix(".csv"), index=False)
        figures = _plot_trajectories(records, fp, out_dir) + _plot_excess_against_n(records, fp, out_dir)
        title = f"{card_label(cells)[1]}: size sweep, {eval_set.value} set, environment {fp}"
        legend = (
            "Reference predictors: test MSE minus the toy's floor, against the expected excess (0 for the oracle; offset^2 for the planted "
            "offset; (1 - factor)^2 x mean(m(x)^2) for the planted scale), with the standard error of the MSE from the draw's label noise. "
            "A difference of more than a few standard errors means the generator, the stated true mean or the floor is wrong."
        )
        refs_md = _markdown(refs, index=False) if len(refs) else "(no reference predictors in this plan)"
        stem.with_suffix(".md").write_text(f"# {title}\n\n{legend}\n\n{refs_md}\n\n" + "".join(f"![{p.stem}]({p.name})\n" for p in figures))
        stem.with_suffix(".html").write_text(
            f"<!doctype html><meta charset='utf-8'><title>{html.escape(title)}</title><h1>{html.escape(title)}</h1><p>{html.escape(legend)}</p>"
            f"{refs.to_html(index=False)}" + "".join(f"<p><img src='{html.escape(p.name)}'></p>" for p in figures)
        )
        written += [stem.with_suffix(s) for s in (".csv", ".md", ".html")] + figures
    return written


def build_report(cells: list[CellSpec], eval_set: EvalSet | None = None, out_dir: Path | None = None) -> list[Path]:
    """Writes the study's report (kind and default eval set from the plan) and returns the files.

    Cells of several cards make one table only when they share the report kind and the reference variant.
    """
    report = cells[0].report
    if len({(c.report.kind, c.report.reference_variant) for c in cells}) > 1:
        raise ValueError(f"Cards {card_label(cells)[0]} differ in report kind or reference variant; report them separately.")
    eval_set = eval_set or report.eval_set
    out_dir = out_dir or layout_root(cells[0].results_dir) / REPORTS_DIR / card_label(cells)[0]
    out_dir.mkdir(parents=True, exist_ok=True)
    return (write_comparison if report.kind == ReportKind.COMPARISON else write_size_sweep)(cells, eval_set, out_dir)


def main(argv: list[str] | None = None) -> int:
    """CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cells", type=Path, nargs="+", help="plan files; several cards sharing a reference make one table")
    parser.add_argument("--eval-set", type=EvalSet, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    files = build_report([c for path in args.cells for c in read_plan(path)], args.eval_set, args.out)
    print(json.dumps([str(f) for f in files], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
