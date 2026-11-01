"""Sets the early-stopping patience of later cards from the validation curves of an earlier card (the chain step after card 5a-2).

    python -m automl_package.experiments.patience <cells.jsonl> --out <json> [--wait-minutes M]

Reads every record of every cell of the plan, any environment fingerprint, and takes the largest stretch of checkpoints any
curve needed to reach its best. No threshold decides anything: curves that hit the epoch cap are counted and listed, never
filtered out. Records written on the laptop arrive by Syncthing, so ``--wait-minutes`` polls until none is missing; still
missing exits 2 naming the cells. The JSON is written, and the patience integer is the last line printed.

Reused: ``cell.all_record_paths`` / ``cell.load_record`` for the records, ``study.read_plan`` for the plan, ``study.wait_until``
for the wait. The online counterpart of the rule is ``utils.convergence.ConvergenceTracker`` (``since_improve >= patience`` stops);
this reads finished curves instead of running one.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from automl_package.experiments.cell import all_record_paths, load_record
from automl_package.experiments.config import CellSpec
from automl_package.experiments.study import EXIT_NOT_READY, read_plan, wait_until
from automl_package.experiments.toys import Toy


def longest_stretch_before_best(values: np.ndarray) -> int:
    """Largest number of checkpoints between consecutive new bests, up to and including the curve's best.

    A new best is a strict decrease below the running minimum (``min_delta`` 0, as every training loop's tracker uses); position 0
    is a best. The tracker stops once ``since_improve >= patience``, so a patience equal to this gap never stops a run before its
    best. Positions are trajectory entries (checkpoints), not epoch numbers. A curve whose best is its first point returns 0.

    Args:
        values: held-out loss at each checkpoint, in order.

    Returns:
        The largest gap ``b - a`` between consecutive new-best positions ``a < b <= best position``.
    """
    running_min = np.minimum.accumulate(values)
    is_best = np.concatenate([[True], values[1:] < running_min[:-1]])
    best_positions = np.flatnonzero(is_best)
    best_positions = best_positions[best_positions <= int(np.argmin(values))]
    return int(np.diff(best_positions).max(initial=0))


@dataclass(frozen=True)
class PatienceChoice:
    """The patience the curves call for, and the curves behind it."""

    patience: int
    largest_best_epoch: int
    max_epochs: int
    n_curves: int
    n_hit_cap: int
    curves: list[dict[str, Any]]  # variant, toy (directory name), n, seed, fingerprint_id, best_epoch, stretch, hit_cap


def missing_records(cells: list[CellSpec]) -> list[CellSpec]:
    """Cells with no record of any fingerprint."""
    return [cell for cell in cells if not all_record_paths(cell)]


def choose_patience(cells: list[CellSpec]) -> PatienceChoice:
    """The largest stretch over every curve of every record of the cells (records without a validation curve, such as the oracle, are skipped)."""
    max_epochs = {cell.protocol.max_epochs for cell in cells}
    if len(max_epochs) != 1:
        raise ValueError(f"the cells disagree on max_epochs: {sorted(max_epochs)}")
    curves = []
    for cell in cells:
        for path in all_record_paths(cell):
            record = load_record(path)
            convergence = record.get("convergence")
            if not convergence or not convergence.get("trajectory"):
                continue
            values = np.array([value for _, value in convergence["trajectory"]], dtype=float)
            curves.append(
                {
                    "variant": cell.variant,
                    "toy": Toy.dir_name(cell.toy),
                    "n": cell.n_train,
                    "seed": cell.seed,
                    "fingerprint_id": record["fingerprint_id"],
                    "best_epoch": int(convergence["best_epoch"]),
                    "stretch": longest_stretch_before_best(values),
                    "hit_cap": bool(convergence["hit_cap"]),
                }
            )
    if not curves:
        raise ValueError("no record carries a validation curve")
    return PatienceChoice(
        patience=max(c["stretch"] for c in curves),
        largest_best_epoch=max(c["best_epoch"] for c in curves),
        max_epochs=max_epochs.pop(),
        n_curves=len(curves),
        n_hit_cap=sum(c["hit_cap"] for c in curves),
        curves=curves,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI; exit status 2 if records are still missing after the wait."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cells", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--wait-minutes", type=float, default=0.0)
    args = parser.parse_args(argv)
    cells = read_plan(args.cells)
    if not wait_until(lambda: not missing_records(cells), args.wait_minutes):
        missing = missing_records(cells)
        print(f"{len(missing)} cell(s) have no record yet:", file=sys.stderr)
        for cell in missing:
            print(f"  {cell.variant}/{Toy.dir_name(cell.toy)}/n{cell.n_train}/s{cell.seed}", file=sys.stderr)
        return EXIT_NOT_READY
    choice = choose_patience(cells)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(asdict(choice), indent=1))
    print(f"{choice.n_curves} curves, {choice.n_hit_cap} hit the cap ({choice.max_epochs} epochs), largest best epoch {choice.largest_best_epoch}")
    print(choice.patience)
    return 0


if __name__ == "__main__":
    sys.exit(main())
