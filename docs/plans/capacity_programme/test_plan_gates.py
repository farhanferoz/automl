"""Mechanical gates for the capacity programme plan (docs/plans/capacity_programme/).

Adapted from ~/dev/Oasis/StrategyA/tests/test_plan_citations.py (the programme-plan skill's
reference implementation). Prose rules get violated by the people who just wrote them; these
gates are the backstop. Three checks, one shrink-only baseline:

1. CITATIONS -- every backticked repo path (contains "/") cited in MASTER/strands/shared must
   exist on disk, and a cited `:LINE` must be <= the file's length. `archive/` and review
   evidence dirs (`reviews_*/`) are exempt (frozen records may rot -- see EVIDENCE below).
   Lines carrying `citecheck-ignore` are exempt (worked examples).
   Deliberately-future artifacts (Create targets not yet built) are grandfathered in
   `gates_baseline.txt` and MUST be removed from it as they land -- the baseline may only
   shrink.
2. DATED NAMES -- no date-shaped filename in this plan dir (a dated plan file is what licenses
   abandoning a plan instead of fixing it). `archive/` exempt.
3. RESULT POINTERS -- any line containing `RESULT:` must name a `.json` ledger artifact. The
   plan holds pointers, not copied numbers. (This proves a pointer EXISTS, not that the number
   is CURRENT -- a green gate is a floor, never a clean bill.)

EVIDENCE DIRS -- `archive/` and `reviews_*/` hold FROZEN RECORDS, not live plan text, and are
excluded from every gate. A review that reports "the plan cites X, which does not exist" would
otherwise FAIL the citation gate for faithfully quoting the defect it found, and a review
quoting a plan's `RESULT:` line would fail the pointer gate. Gating evidence inverts both gates:
it pressures a reviewer to alter the record to make a test pass. Records are read, not repaired.

Run from repo root:
    ~/dev/.venv/bin/python -m pytest docs/plans/capacity_programme/test_plan_gates.py -q
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterator

PLAN_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(PLAN_DIR, "..", "..", ".."))
BASELINE = os.path.join(PLAN_DIR, "gates_baseline.txt")

# `tsv` matters: the protection manifest and the FP-0 dispositions are both .tsv, and without it
# they were invisible to the citation gate -- a typo in either would have gone unnoticed.
_EXTS = r"(?:py|md|json|tsv|csv|tex|sh|txt|log|yaml|yml|pdf|bib)"
_CITATION = re.compile(r"`([A-Za-z0-9_./\-]+\." + _EXTS + r")(?::(\d+))?(?:[-:]\d+)?(?:::[A-Za-z0-9_]+)?`")
_SKIP_PREFIXES = ("http://", "https://", "/tmp/", "~/", "...")
_IGNORE = "citecheck-ignore"
_DATED_NAME = re.compile(r"(\d{4}-\d{2}-\d{2}|20\d{2}[_-]\d{2}[_-]\d{2})")


def _is_evidence_dir(name: str) -> bool:
    """Frozen-record dirs excluded from every gate: `archive/` and review evidence (`reviews_*/`).

    A dated name is REQUIRED here, not forbidden -- a review is a record of what was true on a
    given date, so it is the one thing that must never be "fixed in place". See EVIDENCE DIRS in
    the module docstring for why gating a record inverts the gates it is subject to.
    """
    return name == "archive" or name == "__pycache__" or name.startswith("reviews_")


def _walk_plan_dirs() -> Iterator[tuple[str, list[str]]]:
    """Yields `(root, files)` over live plan text only, pruning evidence dirs."""
    for root, dirs, files in os.walk(PLAN_DIR):
        dirs[:] = [d for d in dirs if not _is_evidence_dir(d)]
        yield root, files


def _plan_files() -> list[str]:
    out = []
    for root, files in _walk_plan_dirs():
        out.extend(os.path.join(root, f) for f in files if f.endswith(".md"))
    return sorted(out)


def _baseline() -> set[str]:
    """Parked forward references, one per line.

    An inline `# comment` after an entry is stripped, so each parked path can carry the task
    that creates it. That provenance is the difference between a reviewable baseline and a
    place typos go to hide -- "NO typo may ever be parked here" is only checkable if a reader
    can see which task is supposed to produce each path.
    """
    if not os.path.exists(BASELINE):
        return set()
    with open(BASELINE, encoding="utf-8") as fh:
        entries = (ln.split("#", 1)[0].strip() for ln in fh)
        return {e for e in entries if e}


def _resolve(ref: str, citing_file: str) -> str | None:
    for base in (REPO, os.path.dirname(citing_file), PLAN_DIR):
        cand = os.path.normpath(os.path.join(base, ref))
        if os.path.exists(cand):
            return cand
    return None


def _collect() -> list[tuple[str, str, str | None]]:
    found = []
    for pf in _plan_files():
        with open(pf, encoding="utf-8") as fh:
            for raw in fh:
                if _IGNORE in raw:
                    continue
                for m in _CITATION.finditer(raw):
                    ref, line_no = m.group(1), m.group(2)
                    if ref.startswith(_SKIP_PREFIXES) or "{" in ref or "*" in ref:
                        continue
                    found.append((pf, ref, line_no))
    return found


def test_plan_dir_sane() -> None:
    """Guards against a vacuous pass if the plan directory moves or empties."""
    assert os.path.isdir(PLAN_DIR), f"plan dir missing: {PLAN_DIR}"
    assert _plan_files(), "no plan files found -- did the plan move?"


def test_no_dated_plan_filenames() -> None:
    """A dated plan filename is what licenses abandoning a plan instead of fixing it."""
    bad = [
        os.path.relpath(os.path.join(r, f), PLAN_DIR)
        for r, files in _walk_plan_dirs()
        for f in files
        if _DATED_NAME.search(f)
    ]
    assert not bad, "Dated plan filenames (fix the plan in place, never fork a dated copy): " + ", ".join(bad)


def test_pathed_citations_resolve() -> None:
    """Every cited repo path must exist, or be a parked forward reference."""
    base = _baseline()
    broken = []
    for pf, ref, _ln in _collect():
        if "/" not in ref:
            continue
        if _resolve(ref, pf) is None and ref not in base:
            broken.append(f"{os.path.relpath(pf, PLAN_DIR)} -> `{ref}`")
    assert not broken, (
        "Unresolvable cited paths (typo, or a future artifact missing from gates_baseline.txt):\n  "
        + "\n  ".join(broken)
    )


def test_cited_line_numbers_exist() -> None:
    """A cited `file:LINE` must not point past the end of that file."""
    bad = []
    for pf, ref, line_no in _collect():
        if line_no is None or "/" not in ref:
            continue
        path = _resolve(ref, pf)
        if path is None or not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8", errors="replace") as fh:
            n = sum(1 for _ in fh)
        if int(line_no) > n:
            bad.append(f"{os.path.relpath(pf, PLAN_DIR)} -> `{ref}:{line_no}` (file has {n} lines)")
    assert not bad, "Cited line numbers past end-of-file:\n  " + "\n  ".join(bad)


def test_result_lines_carry_json_pointer() -> None:
    """Numbers live in the ledger: a `RESULT:` line must name its `.json`."""
    bad = []
    for pf in _plan_files():
        with open(pf, encoding="utf-8") as fh:
            for i, raw in enumerate(fh, 1):
                if "RESULT:" in raw and _IGNORE not in raw and ".json" not in raw:
                    bad.append(f"{os.path.relpath(pf, PLAN_DIR)}:{i}")
    assert not bad, "RESULT: lines without a .json ledger pointer (numbers live in the ledger):\n  " + "\n  ".join(bad)


def test_baseline_shrink_only() -> None:
    """Every baseline entry must still be an unresolved ref -- landed artifacts must be removed."""
    base = _baseline()
    stale = []
    live_refs = {ref for _pf, ref, _ln in _collect() if "/" in ref}
    for entry in base:
        if entry not in live_refs:
            stale.append(f"{entry} (no longer cited -- remove from baseline)")
        elif any(_resolve(entry, pf) for pf, ref, _ln in _collect() if ref == entry):
            stale.append(f"{entry} (now exists -- remove from baseline)")
    assert not stale, "Baseline entries to remove (shrink-only):\n  " + "\n  ".join(sorted(set(stale)))
