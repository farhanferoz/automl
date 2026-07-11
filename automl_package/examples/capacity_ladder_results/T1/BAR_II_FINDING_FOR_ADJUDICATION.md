# T1 bar-(ii) — finding for fresh-context adjudication (2026-07-10)

**Status:** BLOCKS the T1 heavy run. Needs a fresh-context adjudicator ruling before T1 runs.
RESUME.md's stated fix ("promote the decomposed per-region read to be the gated bar-(ii)
statistic") has been **empirically falsified** by the orchestrator (probe below). This note states
what was found, the mechanism, and the exact questions the adjudicator must rule on. Do NOT accept
RESUME's premise — verify from the numbers here.

## What bar (ii) was supposed to test
"Does the per-input depth machinery detect the region-localized depth signal?" — instantiated in
`T1/PREREGISTRATION.md` as the POOLED "A-vs-B contrast": `capacity_ladder_x3.run_repeated_crossfit`
and `capacity_ladder_x1.run_repeated` `hier_vs_global`, `n_bins=2`, corrected-t > 2 on both readers,
≥2/3 seeds, exceeding the G-flat control band.

## Empirical finding (orchestrator probe, training-free, on the selftest synthetic tables)

Probe builds the exact selftest part-(c) planted table — `capacity_ladder_t1._synthetic_t1_table()`
with default flat `means_a` (region A `means=[0,0,0,0,0,0]` = true-flat; region B
`means=[0,0.4,0.8,0.8,0.8,0.8]` = planted +0.8-nat step), noise sd 0.3, N=1500/region — and runs
BOTH the pooled reader and the decomposed per-region reader
(`capacity_ladder_t1._repeated_perbin_by_region`, `n_bins=2`, 50 splits). Reproduce:

```
AUTOML_DEVICE=cpu OMP_NUM_THREADS=4 ~/dev/.venv/bin/python -u \
  /tmp/claude-1000/-home-ff235-dev-MLResearch-automl/5be055f1-2194-4c76-94dd-04c969997b57/scratchpad/probe_t1_decomposed.py
```
(or re-derive in ~10 lines: import `capacity_ladder_t1 as t1`; `score,x = t1._synthetic_t1_table()`;
`t1._repeated_perbin_by_region(score, x, n_bins=2, n_splits=50)`; read `[bin]["indep"/"hier"]["t_corrected"]`.)

**PLANTED flat-A / +0.8-nat-B (T1's literal region shape):**
| reader | crossfit/indep t | hier t |
|---|---|---|
| POOLED (bar ii as written) | **-0.82** | **-0.01** |
| DECOMPOSED region_A | -0.35 (mu -0.0012) | +0.01 (mu +0.0000) |
| DECOMPOSED region_B | **-0.84 (mu -0.0014, passfrac 0.00)** | **-0.02 (mu -0.0000, passfrac 0.02)** |

→ The decomposed region-B read is **just as blind** as the pooled read (mu ≈ 0, t < 0) despite a
large planted +0.8-nat region-B signal. **RESUME's "coded fix" does NOT fix bar (ii).**

**CONFLICTING regions (positive control: region A `means=[0.8,0.4,0,0,0,0]`, prefers shallow):**
| reader | crossfit/indep t | hier t |
|---|---|---|
| POOLED | +72.31 | +70.78 |
| DECOMPOSED region_A | +13.07 (mu +0.34) | +13.08 (mu +0.33) |
| DECOMPOSED region_B | +10.26 (mu +0.27) | +10.22 (mu +0.27) |

→ Every reader fires strongly. The machinery is SOUND — but only when the two regions genuinely
prefer DIFFERENT depths (real heterogeneity).

## Mechanism (validated by the contrast above)
Every stacker read measures per-bin mixture advantage **over the global (pooled) stack** — a
`_split_perbin_advantage` diff `indep_ls - global_ls[mask]`, aggregated Nadeau-Bengio. That is a
**heterogeneity** statistic: it is nonzero only when a region's optimal mixture differs from the
pooled optimum. In T1's construction region B dominates the pooled likelihood, so the global stack
converges onto B's own optimum → region B's per-bin advantage over global ≈ 0; and a truly flat
region A cannot pull global away from B → region A's advantage ≈ 0 too. The decomposed read uses the
SAME global baseline, so it inherits the SAME blindness. Only when region A *actively prefers a
different depth* (positive control) does any reader — pooled or decomposed — register a signal.

## The structural tension (the real problem)
- **Bar (i)** PASS condition includes: **region A flat** (no >2·SE held-out-NLL gain past d1) =
  region A is *indifferent* to depth.
- **Bar (ii)** signal is nonzero ONLY IF region A *actively prefers a different depth than B* (i.e.
  region A is NOT flat — the positive-control case).
- Therefore **passing bar (i) mechanically forces bar (ii) toward a structural zero.** The two bars
  cannot both pass by construction (short of region A being actively hurt by depth, which would fail
  bar (i)'s flatness). The plan's locked outcome semantics "PASS(i)+FAIL(ii) → genuine machinery
  failure → STOP" would therefore fire on a **structural inevitability, not a machinery defect** —
  and the machinery is independently validated SOUND by the positive control (t≈72).

## Important contingency (why the real run still matters)
The probe uses the SYNTHETIC selftest table (hand-planted means), a MODEL of the real T1 depth
table. The REAL T1 toy trains actual width-8 nets at depths 1..6. Whether the real region A is
*flat* (indifferent — linear data fit fine at all depths → bar (ii) structural zero) or *actively
hurt by depth* (deep width-8 net overfits linear region A → real heterogeneity → bar (ii) could
fire) is an EMPIRICAL question the CONSTRUCTION run (bar i) answers directly: bar (i)'s own region-A
NLL-by-depth curve reveals which regime holds. But if bar (i)'s "region A flat" passes as the plan
intends, the analysis above predicts bar (ii) is a forced structural zero.

## Questions for the adjudicator (rule explicitly on each)
1. **Verify** the finding: re-run the probe (or re-derive), confirm decomposed region-B t ≈ 0 on the
   planted table and t≈10-13 on the conflicting control; confirm the mechanism and the bar-tension
   claim. Overrule if wrong.
2. **Ruling on the fix:** is promoting the decomposed read (RESUME #1) correct? (Expected: NO — it is
   equally blind.) If NO, what is the correct disposition of bar (ii)?
   - (R1) REFRAME within plan: T1 tests bar (i) (per-region depth REQUIREMENT); bar (ii) as a
     per-input *likelihood-advantage* test is inapplicable to a flat-low-region toy (structural
     zero, not machinery failure); machinery soundness rests on the positive control; T1's per-input
     payoff is the COMPUTE saving measured by H2 bar (ii). Rewrite bar (ii)/outcome-semantics
     accordingly and run T1 for bar (i) now.
   - (R2) REDESIGN the toy so region A is actively hurt by depth (real heterogeneity) — bigger
     change; couples the depth-requirement story with an overfitting story; may weaken the
     "provably-deep-required" framing.
   - (R3) other (e.g. a per-input *requirement* metric that is not an advantage-over-global stack).
3. **Governance:** is the chosen disposition a WITHIN-PLAN correction the orchestrator may apply and
   proceed (adjudicator-approved), or a USER-LEVEL design fork (G-FORK) that must wait for the user
   (currently away)? Per EXECUTION_PLAN §0d, adjudicator rules first; only an explicit fork ruling
   holds for the user.
4. **May T1's CONSTRUCTION run (bar i) proceed now** regardless of the bar-(ii) disposition (it is
   independent and its region-A curve resolves the contingency above)?

## Deliverable
Write the ruling to `capacity_ladder_results/T1/BAR_II_ADJUDICATION.md`: verdict on each question
above, the corrected bar-(ii) text + outcome-semantics IF R1 (or the redesign spec IF R2), and an
explicit WITHIN-PLAN vs G-FORK determination. This file — not a chat message — is the deliverable.
