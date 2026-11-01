# ProbReg — state, trusted records and known defects

Moved verbatim from the former `docs/plans/capacity_programme/probreg.md` §2, §2.5 and §3 on <!-- citecheck-ignore -->
2026-10-21. Index: `docs/probreg/PLAN.md`.

<!-- from docs/plans/capacity_programme/probreg.md lines 475-593 -->
## 2. State — what is established, and what is not

**Established (toys only).** The k-dropout coherence check — one k-dropout model read at each
fixed k, against a separately trained model dedicated to that k (i.e. an M1-vs-M3 *quality* check)
— has been run on 3 toy problems × 3 repeats. Write-up and numbers:
`docs/probreg/shared/reports/probreg_kselection/probreg_kselection.md` §3.2. Summary of what it shows: the
largest checked k **never** fails the coherence check in any of the 9 cases, but some middle k
fails in **8 of the 9** — a small, real, and currently **unexplained** cost concentrated in the
middle of the range.

**NOT established — and each is a task below:**
- **(b) same choice.** Nothing anywhere compares the arbiter's chosen k against the sweep's best
  k. §3.2 compares *fit quality at each k*; that is a different claim. → **P3**
- **Anything on real data.** All of the above is synthetic. → **P4**
- **Baselines.** No comparison against XGBoost / LightGBM / CatBoost / a standard NN has been
  run for this strand. → **P4**
- **Why the middle-k cost exists.** 8 of 9 is a pattern, not noise, and it is undiagnosed. → **P5**

**Report status.** `docs/probreg/shared/reports/probreg_kselection/probreg_kselection.md` exists (880 lines, PDF
built) and covers the toy work. It predates §1 and therefore uses the older framing; it must not be
extended until §1 is reflected in the code and the battery. → **P6**

---

## 2.5 Where the history lives, and which records can be trusted

Surveyed 2026-07-20 (13 files, all read in full, artifact paths and a sample of line citations
checked against disk). Two disconnected generations of ProbReg planning existed, which never
referenced each other — that disconnection is what this strand ends.

**⚠ GROUND-TRUTH RULE, learned from the survey: for the first generation, only
`automl_package/examples/capacity_ladder_results/RESULTS.md` records what actually completed.** No
inline status glyph in those plan files is trustworthy — several tasks carry adjudicated GO verdicts
in the results ledger while their own plan file still shows them open, and one file's status table
contradicts its own prose from the same day. All four are now banner-frozen in place (not moved —
moving them would have broken ~30 citations including ten frozen preregistration records).

**Still-live reasoning in the frozen files — read, never dispatch:**
- `docs/flexnn/width/history/width_dial_2026-07/cascade_execution_plan_2026-07-11.md` §3.1 — the cross-k
  class-identity conflict (one class asked to be correct at several resolutions at once). Its
  resolution there (regression-only supervision) is withdrawn: removed 2026-10-22 (mixture framing; docs/probreg/cleanup_plan.md).
- The same file's §4.7 — four ProbReg library fixes, **all verified landed at HEAD 2026-07-20**,
  none of which had a completion record in any plan document until that check.
- `docs/probreg/history/diagnoses_2026-07/hetero_nll_diagnosis.md` — the diagnosed, accepted
  hetero-NLL failure (§3 D3).

**Known citation rot (do not build on these):** three 2026-07-11 documents cite
`automl_package/models/probabilistic_regression.py:796-798` for a classifier-probability bug. That
file has grown ~200 lines since; the function moved to roughly `:1010` and the bug is already fixed
there. Re-verify any line citation from that generation before using it.

## 3. Known defects

**D1 — FIXED + VERIFIED 2026-07-20.** `fit_router` scored raw-space `y_val` against
`forward_at_k`'s symlog-space outputs — silently wrong error table ⇒ silently wrong selector, worst
on exactly the heavy-tailed data this strand targets. Fix at
`automl_package/models/probabilistic_regression.py:813`; contract documented in the docstring.
Verified in BOTH directions: fix present → 3 passed; fix deleted → 3 failed
(`tests/test_phase3_dynamic_k.py`, class `TestFitRouterSymlogSpaceAlignment`). Detail + the
measured blindness table: `docs/flexnn/archive/flexnn-core.md` (F9-fix-b block).

**D2 — ✅ FIXED 2026-07-20 (commit `84ad94d`), found already landed during the 2026-07-21 repair
audit.** The units mismatch in `_per_sample_fit_at_k`
(`automl_package/models/probabilistic_regression.py:835`) is fixed in that commit: caller-space `y`
is symlog-transformed before scoring, the contract is documented on the method and on its only
caller `held_out_arbiter_advantage`, and the regression test P1 asked for exists in the same commit
(`tests/test_phase3_dynamic_k.py:726`, `TestArbiterAdvantageSymlogSpaceAlignment`) — re-executed at
the root 2026-07-21: 2 passed.
*(Corrected 2026-07-21: the 2026-07-20 repair pass re-asserted "D2 still stands as a real bug" — prose was edited without re-checking disk while the fix already sat in history. The rule this motivates is now a `docs/flexnn/PLAN.md` Corrections entry. The remainder of this note concerned the former per-rung likelihood function: superseded 2026-11-01, see `docs/probreg/PLAN.md` "Plan of record".)*

**D4 — OPEN, inherited, user-gated.** The cross-k class-identity conflict: with cross-entropy
supervision and k-dropout, the percentile bin boundaries are recomputed for each candidate k, so
"class i" names a different slice of y at each k. What shipped is a runtime warning only — not a
behaviour change. The candidate fixes on record: superseded 2026-11-01, see `docs/probreg/PLAN.md` "Plan of record". It must not be re-discovered a fourth time — it is
recorded here as the owner. *(Rewritten 2026-10-22: the earlier text said the conflict was dodged by
freezing `REGRESSION_ONLY`; that default is withdrawn, see `docs/probreg/cleanup_plan.md`.)*

**D3 — two failing tests, both pre-existing, neither caused by current work.**
- `test_probabilistic_nll_beats_constant_on_heteroscedastic` — root-caused and deliberately
  accepted: `docs/probreg/history/diagnoses_2026-07/hetero_nll_diagnosis.md`.
- `test_prob_regression_heteroscedastic_mse` — error 2.8812 against a 2.5 bar. Verified identical
  to 16 digits on a clean worktree at commit `600460c`, so it is not new. **Undocumented anywhere**
  — the same silent-failure pattern this programme keeps paying for. → **P2**

**D5 — OPEN.** `held_out_arbiter_advantage` (`automl_package/models/probabilistic_regression.py`) is **not M1's selection mechanism**: it returns a per-input `(N,)` array comparing ONE fixed top rung with the k=1 bypass, so it cannot express "chose k=4" for a middle rung or one global k for the dataset. M1's selector reads the per-rung table `all_rung_fit_loss` instead (§1 ⚠️, PA's task text).

**D6 —** Superseded 2026-11-01: see `docs/probreg/PLAN.md` "Plan of record".

---


