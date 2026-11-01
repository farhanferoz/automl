"""Runs the cells of a plan file. No Hydra import: any machine with the package's own dependencies can execute.

    python -m automl_package.experiments.execute <cells.jsonl> [--shard I/N] [--workers W]

A cell already done on this machine (same resolved config and fingerprint) is skipped, so a rerun resumes.

Reused: ``study.read_plan`` / ``study.run_cells``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from automl_package.experiments.study import read_plan, run_cells


def parse_shard(text: str) -> tuple[int, int]:
    """``'I/N'`` as ``(I, N)`` with ``0 <= I < N``."""
    index, _, count = text.partition("/")
    shard = (int(index), int(count))
    if not 0 <= shard[0] < shard[1]:
        raise argparse.ArgumentTypeError(f"shard must be I/N with 0 <= I < N, got {text}")
    return shard


def main(argv: list[str] | None = None) -> int:
    """Runs the plan; exit status 1 if any cell failed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cells", type=Path)
    parser.add_argument("--shard", type=parse_shard, default=(0, 1))
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    summary = run_cells(read_plan(args.cells), workers=args.workers, shard=args.shard)
    print(f"ran {len(summary.ran)}, skipped {len(summary.skipped)} (already done), failed {len(summary.failed)}")
    for label, error in summary.failed.items():
        print(f"FAILED {label}: {error}", file=sys.stderr)
    return 1 if summary.failed else 0


if __name__ == "__main__":
    sys.exit(main())
