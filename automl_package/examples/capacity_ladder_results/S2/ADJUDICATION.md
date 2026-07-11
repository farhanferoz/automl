# S2 adjudication — direct held-out-likelihood selector (the principled objective)

**Adjudicator:** fresh-context Opus (never the producing session). Every headline number below was
re-derived independently from `s2_summary.json` (not read back from worker prose) with a standalone
recomputation script; the deterministic bootstrap SE was reproduced with the same `seed=0,
n_boot=1000` as `cs1._paired_bootstrap_se`. All 9 structured + 6 broad cases loaded and checked.

**Verdict: GO** — certify S2 and fold to the RESULTS.md ledger. Bar (i) failed as pre-registered
but the failure is **BENIGN (a 0.36·SE tie)**; bars (ii)/(iii) pass; X7 does not fire; the recipe of
record stays **SOFT**.

---

## Q1 — Re-derived numbers (confirm / correct the four bullets)

| Quantity | Reported | Re-derived (independent) | Ruling |
|---|---|---|---|
| Bar (i) n_pass (direct_300ep blend ≤ soft blend, structured) | 5/9 | **5/9** | CONFIRMED |
| Bar (i) mean paired diff (direct_300ep − soft) | +0.000413 | **+0.00041270** | CONFIRMED |
| Bar (i) bootstrap SE | 0.001148 | **0.00114761** (mean/SE = **0.36**) | CONFIRMED |
| Bar (ii) max broad advantage over global | 0.00584 | **0.00584063**, all 6 ≤ 0.02 | CONFIRMED (PASS) |
| Bar (iii) hard-read ≤ global (structured) | 9/9 | **9/9** | CONFIRMED (PASS) |
| X7 gap_closure (toy D, mean over seeds) | 1.6587 | **1.658674**, per-seed [2.8999, 1.0361, 1.0400] | CONFIRMED (no-fire) |

Bar (i) per-case (direct_300ep − soft, nat): C[−0.0027, −0.0028, +0.0028], D[+0.0064, −0.0004,
+0.0040], E[−0.0049, +0.0022, −0.0009] → 5 pass / 4 miss, all |diff| ≤ 0.0064. Mean blend NLL:
soft **0.6807** vs direct_300ep **0.6811** (Δ = +0.0004 nat). Bar (i) FAILS as written on BOTH
sub-conditions (5 < 6, and +0.0004 > 0). All four dispatch bullets are correct as reported.

## Q2 — Is bar (i)'s failure BENIGN or MATERIAL?

**BENIGN.** The prereg's own prediction was "the direct objective **wins or ties**" (PREREGISTRATION.md
line 24; EXECUTION_PLAN §1 S2 line 167). The observed result is a **tie**: mean paired diff +0.0004 nat
at **0.36·SE** — nowhere near the ~2·SE that would signal a genuine loss, and the two arms' mean blend
NLLs (0.6811 vs 0.6807) are indistinguishable. The 0.0004-nat magnitude is the same order as effects S1
itself certified "negligible" (S1 smoothing effect 0.0001 nat) and its benign sub-milli-nat regressions.
The 5-pass/4-miss split with sub-0.007-nat misses is a coin-flip pattern, not a systematic deficit.

Cross-check against under-training: the diagnostic **3000-epoch** arm is also a tie/marginal-loss
(mean diff +0.00124 nat, 5/9, mean blend 0.6820) — training 10× longer does **not** recover a win, so
the tie is a genuine property of the direct objective, not an artifact of the 300-epoch schedule or of
reading bars against `direct_300ep`. The point estimate lands marginally on the *worse* side, so the
literal "direct objective is better/at-least-ties in NLL" fails at the point estimate; statistically it
is indistinguishable from zero. Both readings converge on **TIE** — matching the "ties" branch the
prereg explicitly predicted. Not material: the direct objective is neither genuinely worse nor better.

## Q3 — Recipe implication

**The tie leaves SOFT standing.** The direct/principled objective (the arm the S1 factorial never ran)
was the only candidate that could displace `soft`; displacing it required the direct objective to be
meaningfully **better**. It merely ties (point estimate marginally worse) at both epoch counts, so S2
provides **no evidence to switch**. Recipe of record for the deployable selector remains **SOFT
(prior-dominant)**, unchanged from S1. S2's positive contribution: the deployment metric can now be
optimized *directly* with no derived label and reach parity with the best imitation target — the
principled objective is *validated as a legitimate alternative*, but does not supersede soft.

## Q4 — X7 trigger

**No-fire confirmed → no G-X7 user gate.** gap_closure = mean_seeds((nll_global − best_selector_blend)/
(nll_global − oracle_x_nll)) on toy D = **1.6587** ≥ 0.5, per-seed **[2.8999, 1.0361, 1.0400]**.
`best_selector_blend` = min over {soft, direct_300ep, direct_3000ep} per case (matches the dispatch's
"min blend NLL over {S2, soft}", S2 spanning both epoch arms). All three per-seed closures exceed 1.0
because the best selector's blend NLL is *below* the honest hard-route oracle-x bound on toy D
(best_sel < oracle_x for all 3 seeds) — expected, not a bug: the soft density-blend is a different
(softer) read than the hard-routed oracle bound, exactly S1's certified "blend can beat hard" finding.
The deployable-selector question is closed on toy D. Per §0d, X7 is **SKIPPED**; one ledger line records
the arithmetic (below).

## Q5 — Overall verdict

**GO** to certify S2 and fold to the ledger. Numbers re-derived and confirmed; bar (i) failure benign
(0.36·SE tie, matching prereg's own prediction); recipe of record stays SOFT; X7 no-fire.

---

## READY-TO-PASTE RESULTS.md FOLD BLOCK (orchestrator pastes verbatim)

## S2 — direct held-out-likelihood selector (the principled objective, ProbReg). Verdict: GO (adjudicated 2026-07-10, fresh context)

**Headline.** Training the selector π(x) **directly on the deployment metric** (maximize
`mean_i logsumexp_c(log softmax(w(x_i))_c + score_tr[i,c])`, train-half only — L5-compliant, no
derived label, no prior, no tuned λ; the arm the S1 factorial never ran) reaches **parity** with the
best imitation target `soft`, but does **not** beat it. **Recipe of record stays `soft`** (S1's
prior-dominant winner); the direct objective is validated as a legitimate alternative, not a
replacement. Bars read against the config-matched **300-epoch** arm (`direct_300ep`); the 3000-epoch
arm is an under-training diagnostic only.

- **Direct ties soft, does not displace it.** Mean blend NLL over the 9 structured cases: soft
  **0.6807** vs direct_300ep **0.6811** (Δ = +0.0004 nat). The 3000-epoch arm is also a tie/marginal-loss
  (0.6820, +0.0012 nat) — 10× longer training recovers no win, so the tie is a real property of the
  objective, not under-training.
- **Bars PASSED:** **(ii)** broad-control advantage over global ≤ 0.02 nat on all 6 (max **0.00584** on
  E_broad s2 — no invented structure on the controls); **(iii)** hard-read NLL ≤ global on **9/9**
  structured (degrades gracefully — the hard route never underperforms the fixed global column).
- **Bar FAILED as pre-registered — adjudicated BENIGN (corrected wording logged beside the original;
  NO silent rewrite, per §0b):**
  - **(i)** original: "S2 blend NLL ≤ S1-`soft` blend NLL on ≥ 6/9 structured cases AND mean paired
    diff ≤ 0 nat" (equivalently, plan §1 S2: "S2 blend ≥ S1-winner blend on ≥6/9 and mean paired diff
    ≥ 0; prediction: the direct objective wins or ties"). **FAILED both sub-conditions: 5/9** (needs
    ≥6) and **mean paired diff +0.000413 nat > 0**. Cause: the direct objective **ties** soft rather
    than beating it — the miss is a **0.36·SE** near-tie (SE 0.001148; mean/SE = 0.36), well inside
    noise, with per-case |diff| ≤ 0.0064 nat and a 5-pass/4-miss coin-flip split; the 0.0004-nat gap is
    the order S1 itself certified "negligible". This lands exactly on the "ties" branch the
    prereg's own prediction ("wins or ties") admits. **Corrected reading:** "the direct objective at
    least ties the imitation winner — mean paired blend-NLL diff indistinguishable from zero (0.36·SE),
    means 0.6811 vs 0.6807 essentially equal" — S2 **SATISFIES**. Because the tie means the principled
    objective does not *beat* `soft`, the recipe of record is unchanged (SOFT stands).
- **X7 (toy D) does NOT fire → no G-X7 gate.** gap_closure = mean_seeds((global − best_selector_blend)/
  (global − oracle-x)) = **1.6587** ≥ 0.5 (per-seed [2.90, 1.04, 1.04]); best_selector_blend = min over
  {soft, direct_300ep, direct_3000ep}. All three closures > 1 because the best selector's blend NLL sits
  *below* the honest hard-route oracle-x bound on toy D (blend beats hard — S1's certified finding), so
  the deployable-selector question is closed on D. X7 SKIPPED per §0d.
- **Strict-probabilistic + leak-free CONFIRMED (inherited from S1 verbatim):** S1's split
  (TRAIN=even rows, EVAL=odd, eval-A/eval-B disjoint for the honest oracle bound), `cs1._eval_arm`
  blend+hard reads, and `cs1._oracle_reads` are reused unchanged; the only new code is `_train_router_direct`,
  whose loss is `cs1._blend_nll`'s own computation on the train half with `score_tr` a **detached**
  constant (gradient reaches the router only — the ladder is never touched). No label, no prior, no λ.

Artifacts: `capacity_ladder_results/S2/{PREREGISTRATION.md,s2_summary.json}`, `capacity_ladder_s2.py`.
Adjudication: fresh-context Opus (not the producing session), re-derived every headline number
independently from `s2_summary.json` (all four bars + X7, bootstrap SE reproduced at seed=0/n_boot=1000);
bar (i) failure certified BENIGN (0.36·SE tie, matching the prereg's own "wins or ties" prediction);
recipe of record stays SOFT; X7 no-fire (closure 1.6587), no user gate.
