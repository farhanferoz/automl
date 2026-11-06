"""Expands a study into cells, skips the cells already done, and runs the rest (in-process or in a process pool).

Reused: ``cell.run_cell`` / ``cell.cell_status`` for one cell and its identity, ``config.CellSpec`` and its JSON round trip.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import multiprocessing
import os
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from automl_package.enums import ModelName
from automl_package.experiments.cell import CellStatus, cell_status, fingerprint_id, run_cell
from automl_package.experiments.config import CellSpec, PredictorKind, StudyConfig, from_dict, to_dict
from automl_package.experiments.toys import Toy

CELLS_DIR = "cells"  # <results_root>/cells/<card>.jsonl: the plan of each card
RUNS_DIR = "runs"  # <results_root>/runs/<group>/<variant>/...: the records; cards sharing reference records name one group
REPORTS_DIR = "reports"  # <results_root>/reports/<card label>/: tables and pages
PLAN_FILE = "{card}.jsonl"
POLL_SECONDS = 30  # records and plans written on the laptop arrive by Syncthing; check this often while waiting
SECONDS_PER_MINUTE = 60
EXIT_NOT_READY = 2  # exit status of a step whose input (plan file, records) never arrived


def results_dir(study: StudyConfig) -> Path:
    """Where the study's records go: ``<results_root>/runs/<results_subdir or card>``."""
    return Path(study.output.results_root) / RUNS_DIR / (study.output.results_subdir or study.card)


def plan_path(study: StudyConfig) -> Path:
    """Where the study's plan goes: ``<results_root>/cells/<card>.jsonl``."""
    return Path(study.output.results_root) / CELLS_DIR / PLAN_FILE.format(card=study.card)


def layout_root(records_dir: str | Path) -> Path:
    """The results root a cell's records folder (``<results_root>/runs/<group>``) sits in."""
    return Path(records_dir).parents[1]


def plan(study: StudyConfig) -> list[CellSpec]:
    """Every cell of the study, independent variants first (dependent ones run after their sources)."""
    reference = study.report.reference_variant
    if reference is not None and reference not in study.variants:
        raise ValueError(f"reference variant '{reference}' is not among the variants {sorted(study.variants)}")
    for vid in study.report.environment_independent:
        if vid not in study.variants or study.variants[vid].tuned:
            raise ValueError(f"report.environment_independent names '{vid}': only an untuned variant of this study can stand in across environments.")
    for vid, model in study.variants.items():
        for source in (model.match_params_to, model.k_sweep.from_variant if model.k_sweep else None):
            if source is not None and source not in study.variants:
                raise ValueError(f"variant '{vid}' depends on '{source}', which is not among the variants {sorted(study.variants)}")
    unknown = sorted(set(study.toy_filter) - set(study.toys))
    if unknown:
        raise ValueError(f"toy_filter names {unknown}; toys are {sorted(study.toys)}")
    labels = [t for t in study.toys if not study.toy_filter or t in study.toy_filter]
    curve_seeds = study.output.curve_seeds if study.output.curve_seeds is not None else study.protocol.seeds[:1]
    cells = [
        CellSpec(
            study=study.name,
            card=study.card,
            variant=vid,
            model=model,
            toy_label=label,
            toy=study.toys[label],
            n_train=n,
            seed=seed,
            protocol=study.protocol,
            hpo=study.hpo,
            curves=study.output.curves and seed in curve_seeds and model.kind == PredictorKind.AUTOML and model.name == ModelName.PROBABILISTIC_REGRESSION,
            results_dir=str(results_dir(study)),
            report=study.report,
        )
        for label in labels
        for n in study.sizes
        for seed in study.protocol.seeds
        for vid, model in study.variants.items()
    ]
    return sorted(cells, key=lambda c: c.depends_on is not None)


def write_plan(cells: list[CellSpec], path: Path) -> None:
    """One JSON line per cell."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(to_dict(c), default=str) + "\n" for c in cells))


def read_plan(path: Path) -> list[CellSpec]:
    """Cells of a plan file."""
    return [from_dict(CellSpec, json.loads(line)) for line in path.read_text().splitlines() if line.strip()]


def wait_until(condition: Callable[[], bool], wait_minutes: float) -> bool:
    """Polls ``condition`` every ``POLL_SECONDS`` for up to ``wait_minutes``; True as soon as it holds (checked once even when ``wait_minutes`` is 0)."""
    deadline = time.monotonic() + wait_minutes * SECONDS_PER_MINUTE
    while not condition():
        if time.monotonic() >= deadline:
            return False
        time.sleep(min(POLL_SECONDS, max(deadline - time.monotonic(), 0.0)))
    return True


def shard_of(cell: CellSpec, n_shards: int) -> int:
    """Shard of the cell's draw (toy version, size, seed): all variants of one draw stay together so dependents find their sources."""
    key = f"{Toy.dir_name(cell.toy)}|{cell.n_train}|{cell.seed}".encode()
    return int(hashlib.sha256(key).hexdigest()[:8], 16) % n_shards


@dataclass
class RunSummary:
    """What a run did."""

    ran: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)


def _label(cell: CellSpec) -> str:
    return f"{cell.variant}/{Toy.dir_name(cell.toy)}/n{cell.n_train}/s{cell.seed}"


def _run_stage(cells: list[CellSpec], workers: int, summary: RunSummary) -> None:
    if not cells:
        return
    if workers <= 1:
        for cell in cells:
            try:
                run_cell(cell)
                summary.ran.append(_label(cell))
            except Exception: # one failing cell must not stop the others; the summary reports it
                summary.failed[_label(cell)] = traceback.format_exc()
        return
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {pool.submit(run_cell, cell): cell for cell in cells}
        for future in concurrent.futures.as_completed(futures):
            label = _label(futures[future])
            if future.exception() is None:
                summary.ran.append(label)
            else:
                error = future.exception()
                summary.failed[label] = "".join(traceback.format_exception(type(error), error, error.__traceback__))


def run_cells(cells: list[CellSpec], workers: int = 1, shard: tuple[int, int] = (0, 1)) -> RunSummary:
    """Runs the cells of ``shard`` (index, count) that are not done on this machine: independent variants first, then dependent ones."""
    summary = RunSummary()
    fp_id = fingerprint_id()
    mine = [c for c in cells if shard_of(c, shard[1]) == shard[0]]
    todo = []
    for cell in mine:
        if cell_status(cell, fp_id) == CellStatus.DONE:
            summary.skipped.append(_label(cell))
        else:
            todo.append(cell)
    if todo:
        threads = str(todo[0].protocol.torch_threads)
        os.environ["OMP_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = threads  # inherited by the spawned workers
    _run_stage([c for c in todo if c.depends_on is None], workers, summary)
    _run_stage([c for c in todo if c.depends_on is not None], workers, summary)
    return summary

