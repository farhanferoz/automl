# T3 pre-registration — moving-mode power curve (count-lane analog of X10)

**Task:** `docs/plans/perinput_selector_2026-07-10/EXECUTION_PLAN.md` §2 T3. X4/X4b ran the
non-nested per-input arbiter (mixture-vs-best-single-Gaussian held-out advantage, multi-restart
keep-best-by-train-MAP) on toy E (moving modes: two components merge at both ends of x and resolve
only mid-band) at a single size, N_TR=1000/N_TE=2500, and closed the model-capture question there
(2/3 or 3/3 seeds depending on X4 vs X4b, with X4b's own collapse-vs-genuine-limit split). That
closes "absent at N=1000" but not "absent, or under-powered": T3 asks whether the hump emerges at
larger N, turning the X4/X4b read into a power curve, the same way X10 did for the depth lane.

## Instrument (pinned — reuse, do not reinvent)

`capacity_ladder_x4b.py`'s E-lane non-nested arm under R=8 multi-restart, keep-best-by-training-MAP
(`fit_and_score_seed_mr`), called unchanged. T3 varies ONLY N_train=N_test; K_MAX, ALPHA0, SIGMA,
SEP, N_EPOCHS, M_GOLD, N_GRID, and the keep-best criterion are X4/X4b's own instrument config,
held fixed across the sweep — they govern training/evaluation resolution, not the identifiability
question being asked. `capacity_ladder_t3.py` calls `x4b.fit_and_score_seed_mr` and
`x4.tercile_verdict` / `x4._tercile_means` / `x4._tercile_masks` verbatim; the one addition is a
bootstrap SE (G1 plain i.i.d., `_capacity_ladder._bootstrap_col_means`, same utility as
F2/F3/X1/K1K2K3) on the gold Δ*(x) middle-tercile mean, which X4/X4b report as a bare mean with no
dispersion — needed because T3's bar below requires `gold_mid > 0` judged against `2·SE`.

## Design

N_train = N_test ∈ {1000, 4000, 16000} (the June size, then two power steps up); toy E (moving
modes) + its E_broad control (variance-only humps, never bimodal); 3 seeds {0, 1, 2}; R=8 restarts
per (N, toy, seed) cell, exactly as X4b.

Reads per N, per toy, per seed: gold Δ*_mid (model-capture, mid-region — mean of the gold-standard
Δ*(x) curve's middle x-tercile, plus its bootstrap SE) and arbiter mid-region recovery (X4's own
`tercile_verdict.recovered` — significantly positive by its own two-sided band AND above both tail
terciles). Per-N summary: number of seeds (of 3) with a significant gold hump, number of seeds (of
3) with arbiter recovery, and whether the E_broad control stayed flat.

## Pre-registered outcome readings (locked — read against these, do not reframe post hoc)

- **Hump emerges:** gold Δ*_mid > 0 by 2·SE on ≥ 2/3 seeds, at some N in the sweep → **"recoverable
  at N=`<N>`"**, reporting the smallest such N (the crossing point).
- **Stays absent:** gold Δ*_mid stays ≤ 0 (or not significant) through N=16000, with the control
  flat throughout → **"moving-mode per-input count effectively absent up to N=16000"**, reporting
  the measured bound (the gold_mid values and SEs at each N).
- **Control guard (HARD, every N):** E_broad must stay flat 3/3 at every N — arbiter not recovered
  AND gold middle-tercile mean ≤ 0.05 (X4b's own flat tolerance, reused unchanged). If the control
  fires a false positive at ANY N, the instrument is invalid at that N: STOP, escalate to a
  fresh-context adjudicator (§0b). This does not resolve into either outcome above.

## Selftest

Re-runs `capacity_ladder_x4b.run_selftest_mr` UNCHANGED (the known-answer gold-standard oracle:
humps on E, flat on E_broad, at small N/epochs/R) to confirm the underlying instrument is intact,
then adds a single N=16000, R=1, 1-seed smoke-fit at the REAL instrument config (N_EPOCHS=1000,
M_GOLD=1500, N_GRID=40) to measure and print its wall-time — the "measure ONE unit before the
matrix" datum (§0b) the orchestrator needs to cost the real N=16000 row (R=8 × 3 seeds × 2 toys at
that N) before launching it. Asserts the smoke-fit's gold_mid (mean + SE) and arbiter tercile
means are finite.

## Governance

3 seeds {0,1,2} (§0b), R=8 restarts (X4b-matched). Artifacts →
`capacity_ladder_results/T3/{t3_summary.json}`. Heavy run owned by the orchestrator main thread
(`AUTOML_DEVICE=cpu`, `OMP_NUM_THREADS=4`, serialized). Strictly probabilistic: the MAP objective's
Dirichlet-usage prior is the model's own term (coefficient 1, no tuned λ) — unchanged from
X4/X4b, no penalty added by T3. A control-guard failure or any design-forcing surprise → fresh-
context adjudicator (§0c), never the producing session.
