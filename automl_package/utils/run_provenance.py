"""Captures what produced a result, so a saved number can be reproduced or explained later.

A result JSON that records only its hyper-parameters cannot answer the question that actually
matters when it fails to reproduce: *what was different?* Library version, thread count, git commit
and device all change floating-point results without changing a single configured value --
**thread count especially**, because it changes the order floating-point reductions happen in, which
perturbs the low-order bits and can flip discrete downstream decisions (a per-input `argmin`, for
instance) far more visibly than the perturbation itself would suggest.

Motivating case (capacity programme, 2026-07-21): a saved reference could not be reproduced even by
its own commit, and because the JSON stored no environment metadata at all, the cause could not be
identified from the artifact -- only re-derived by re-running history. See
`docs/plans/capacity_programme/shared/fp5-stale-reference-finding.md`.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from typing import Any

import numpy as np
import torch


def _git_state() -> dict[str, Any]:
    """Returns the current commit and whether the tree was dirty; never raises."""
    def _run(*args: str) -> str | None:
        try:
            out = subprocess.run(args, capture_output=True, text=True, timeout=10, check=False)
            return out.stdout.strip() if out.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            return None

    commit = _run("git", "rev-parse", "HEAD")
    status = _run("git", "status", "--porcelain")
    return {
        "commit": commit,
        # A dirty tree means the commit alone does NOT identify the code that ran.
        "dirty": None if status is None else bool(status),
    }


def run_provenance() -> dict[str, Any]:
    """Returns the environment facts needed to reproduce -- or explain -- a numeric result.

    Attach under a `provenance` key alongside a result's `config`. Cheap enough to call per run.
    """
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git": _git_state(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        # Thread count is recorded because it CHANGES RESULTS: BLAS/OpenMP reduction order depends
        # on it, so the same code on the same data yields slightly different floats at 4 threads vs
        # 22. Two runs are only comparable at equal thread counts.
        "threads": {
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "torch_num_threads": torch.get_num_threads(),
            "cpu_count": os.cpu_count(),
        },
        "automl_device": os.environ.get("AUTOML_DEVICE"),
    }
