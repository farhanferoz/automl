"""Every result-number in the capacity-programme plan must name the ledger cell it came from.

WHY THIS EXISTS
---------------
`test_plan_gates.py` checks that cited PATHS resolve and that lines literally containing
`RESULT:` carry a `.json` pointer. Neither can see a stale NUMBER: the path resolves fine
while the figure beside it is wrong, and only a handful of lines say `RESULT:`. That gap was
recorded on 2026-07-20 (`MASTER.md`, "THE NUMBERS GATE IS PARTIAL") after the citations gate
was reported as `6 passed` many times while all four strand plans were undispatchable.

Case law from the source programme (`~/.claude/skills/programme-plan/SKILL.md`): a headline
`1.37` while the ledger said `1.352`; a "close call" at `1.339` put to the user three times
when the ledger said `1.086` and *underperforms*. Both were invisible to a citations gate.

⛔ THE DESIGN THAT DOES NOT WORK HERE -- DO NOT REINSTATE IT
------------------------------------------------------------
The obvious port of the reference implementation is "collect every numeric leaf from every
results JSON, then check each plan number against that pool". **It was built, measured, and
discarded on 2026-07-20, because it cannot work at this ledger's density.** The 134 JSONs
under `LEDGER_ROOTS` hold ~265,000 numeric leaves, which at 3 decimal places SATURATE the
interval below 1.0:

    dec=3, value in [0.05, 1.0):  100.0% of RANDOM values spuriously "match"
    dec=3, value in [1.0, 10.0):   24.4%
    dec=4, value in [0.05, 1.0):   97.3%

A held-out accuracy or MSE is exactly a 3-decimal number below 1.0, so the pooled check has a
**100% false-negative rate precisely where this programme's numbers live**. It was verified
empty-handed: an injected stale `0.7581` passed it silently. A gate that cannot fail on the
values it exists to guard is decoration, and a green one is worse than none because it is
believed. The lesson generalises: **a value-match is only meaningful against the CELL a number
claims to come from, never against a pool.**

WHAT THIS GATE CHECKS INSTEAD -- provenance first, then a scoped value-match
---------------------------------------------------------------------------
1. `test_pointered_numbers_match_their_cell` -- when a line carries BOTH a result-number and a
   `.json` pointer, that number must actually appear in THAT file (rounded to the precision the
   plan quotes it at). Scoped to one cell, so there is no collision problem and a stale figure
   fails immediately. This is the check that has teeth.
2. `test_unpointered_result_numbers_do_not_grow` -- a ratchet on the numbers that name no cell
   at all. At install, **80 of 81 result-number lines in the plan cited no source cell** -- the
   `RESULT:` gate never saw them because it only fires on lines containing that literal word.
   Those 80 cannot be fixed in one pass, so they are a shrink-only budget: new unsourced numbers
   are blocked immediately, existing debt burns down on its own schedule. As a number gains a
   pointer it moves under check 1, where its VALUE starts being verified.

The ratchet is a COUNT, not a list of `file:line` entries, deliberately: line numbers churn on
every plan edit, so a line-keyed baseline rots into noise within a day and gets deleted.

WHAT COUNTS AS A RESULT-NUMBER
------------------------------
>= 3 decimal places, in [0.05, 1000). The shape of an MSE, an NLL, a held-out accuracy, a
log-likelihood. The bounds are deliberate:
  * the FLOOR excludes learning rates and tolerances (`1e-2`, `0.25`);
  * the CEILING excludes arXiv identifiers (`2401.04088`, `1701.06538`), which match the
    decimal shape exactly and would otherwise dominate the failure list.
Two-decimal constants (`0.90` positive-control bar, `0.25` router tolerance) are deliberately
NOT results -- they are pre-registered bars and settings, and belong to the constants tables.

ESCAPE HATCH: put `numcheck-ignore` on the line, with a reason, for a number that is
legitimately not a ledger result -- a rate-card fact, an illustrative figure, or a worked
example of a KNOWN-WRONG number. Every use is a hole in the gate; say why.

KNOW WHAT THIS GATE CANNOT SEE: it proves a number matches the cell it CITES, never that the
cell is the RIGHT one or the NEWEST one. In the source programme a plan reported `1.086/1.488`
("underperforms") while a later re-run said `1.339/1.790` -- a tie and a win, on a different
cost basis. Both were real; the gate would have passed either. A number needs its BASIS as well
as its cell. **A green gate is a floor, never a clean bill.**

Run from repo root:
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python -m pytest docs/plans/capacity_programme/test_plan_numbers.py -q
"""

from __future__ import annotations

import json
import os
import re

import pytest

# Reused rather than re-derived: the same evidence-dir pruning (`archive/`, `reviews_*/`) the
# citations gate uses, so the two gates can never disagree about what counts as live plan text.
from test_plan_gates import PLAN_DIR, REPO, _resolve, _walk_plan_dirs

LEDGER_ROOTS = [
    os.path.join(REPO, "automl_package", "examples", "capacity_ladder_results"),
]

# Ratchet on result-numbers that cite no ledger cell. Measured at install (2026-07-20).
# It may ONLY fall. Raising it to make a new unsourced number pass defeats the gate.
UNPOINTERED_INSTALLED = 80

_NUM = re.compile(r"\b\d+\.\d{3,}\b")
_JSON_REF = re.compile(r"`([A-Za-z0-9_./\-]+\.json)`")
_IGNORE = "numcheck-ignore"
_LO, _HI = 0.05, 1000.0


def _plan_lines() -> list[tuple[str, int, str]]:
    """Yields `(path, lineno, text)` for every line of live plan text."""
    out = []
    for root, files in _walk_plan_dirs():
        for name in sorted(files):
            if not name.endswith(".md"):
                continue
            path = os.path.join(root, name)
            with open(path, encoding="utf-8") as fh:
                out.extend((path, i, raw) for i, raw in enumerate(fh, 1))
    return out


def _result_numbers(text: str) -> list[str]:
    """Result-shaped number tokens on one line, empty if the line opts out."""
    if _IGNORE in text:
        return []
    return [t for t in _NUM.findall(text) if _LO <= float(t) < _HI]


def _cell_floats(path: str) -> set[str]:
    """Every numeric leaf in a results JSON, as strings rounded to 3..8 decimals.

    Returning pre-rounded strings lets a plan's `0.830` match a cell's `0.8299999998` while
    keeping the comparison exact at the precision the plan chose to quote.
    """
    raw: list[float] = []

    def walk(obj: object) -> None:
        if isinstance(obj, bool):
            return  # a `true` flag is not a result
        if isinstance(obj, (int, float)):
            raw.append(float(obj))
        elif isinstance(obj, dict):
            for v in obj.values():
                walk(v)
        elif isinstance(obj, list):
            for v in obj:
                walk(v)
        elif isinstance(obj, str):
            raw.extend(float(m.group(0)) for m in _NUM.finditer(obj))

    try:
        with open(path, encoding="utf-8") as fh:
            walk(json.load(fh))
    except (json.JSONDecodeError, OSError):
        return set()
    return {f"{v:.{dec}f}" for v in raw for dec in range(3, 9)}


def test_ledger_and_plan_are_readable() -> None:
    """Guards against a vacuous pass if the plan or the results tree moves."""
    assert _plan_lines(), "no plan lines found -- did the plan move?"
    assert any(_result_numbers(t) for _p, _i, t in _plan_lines()), "no result-numbers found -- the regex is broken"
    assert any(
        f.endswith(".json") for lr in LEDGER_ROOTS for _r, _d, files in os.walk(lr) for f in files
    ), f"no ledger JSONs under {LEDGER_ROOTS} -- did the results tree move?"


def test_pointered_numbers_match_their_cell() -> None:
    """HARD GATE: a number quoted beside a `.json` pointer must appear in that cell.

    This is the check with teeth. Scoping the match to the ONE cited cell is what makes it
    meaningful -- see the module docstring for why a pooled match cannot work here.
    """
    bad = []
    for path, lineno, text in _plan_lines():
        nums = _result_numbers(text)
        if not nums:
            continue
        for ref in _JSON_REF.findall(text):
            cell = _resolve(ref, path)
            if cell is None or not os.path.isfile(cell):
                continue  # a missing cell is the citations gate's job, not this one
            values = _cell_floats(cell)
            for tok in nums:
                dec = len(tok.split(".")[1])
                if f"{float(tok):.{dec}f}" not in values:
                    bad.append(f"  {os.path.relpath(path, PLAN_DIR)}:{lineno}  {tok} not found in `{ref}`")
    if bad:
        pytest.fail(
            f"{len(bad)} number(s) do NOT appear in the ledger cell cited on the same line.\n"
            "Either the ledger moved and the plan did not (STALE), or the number was quoted\n"
            "from the wrong cell. Re-read the cell and re-cite its value -- never edit the\n"
            "number to match a remembered figure.\n\n" + "\n".join(bad)
        )


def test_unpointered_result_numbers_do_not_grow() -> None:
    """RATCHET: result-numbers citing no ledger cell may only decrease.

    At install, 80 of 81 result-number lines named no source cell. A number that exists only
    as prose is not a result -- but 80 cannot be repaired in one pass, so this budget blocks
    NEW unsourced numbers while the existing debt burns down. Lower `UNPOINTERED_INSTALLED`
    as it falls; never raise it.
    """
    unpointered = [
        f"  {os.path.relpath(p, PLAN_DIR)}:{i}  {', '.join(nums)}"
        for p, i, t in _plan_lines()
        if (nums := _result_numbers(t)) and not _JSON_REF.search(t)
    ]
    assert len(unpointered) <= UNPOINTERED_INSTALLED, (
        f"Result-numbers with no ledger-cell pointer rose to {len(unpointered)} "
        f"(budget {UNPOINTERED_INSTALLED}). A new number was added without naming the cell it "
        "came from. Cite the `.json` on the same line, or mark it `numcheck-ignore` with a "
        "reason. Do NOT raise the budget.\n\nAll current sites:\n" + "\n".join(sorted(unpointered))
    )
