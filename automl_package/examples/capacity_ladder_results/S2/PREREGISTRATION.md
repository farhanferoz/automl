# S2 preregistration — direct held-out-likelihood selector (the principled objective)

Source: `docs/plans/perinput_selector_2026-07-10/EXECUTION_PLAN.md`, §1 WS-A, task **S2**.
Written BEFORE any real run, verbatim from the ratified plan (2026-07-10) plus the dispatch
message's arithmetic (which is more explicit than the plan text on sign convention). Do not
edit after a real run starts; outcomes are read against this text, not the reverse.

## Purpose

Train the selector DIRECTLY on the deployment objective instead of imitating any derived
label — the arm the original study never ran.

## Design

Same `_RouterMLP` w(x). Objective (train half only): maximize
`mean_i logsumexp_c(log softmax(w(x_i))_c + score_tr[i,c])` — per-input stacking; an ordinary
held-out log-score operation (L5-compliant, no labels, no λ). Adam lr 1e-2, 300 ep (match S1);
also record a 3000-ep arm to check under-training. Evaluate under the FULL S1 protocol (blend +
hard + oracle-x + broad controls), same seeds/splits, head-to-head vs the S1 winner (`soft`,
mean blend NLL 0.6807 on the structured toys — the lowest of S1's five factorial arms).

## Pre-registered bars

(i) S2 blend NLL ≤ S1-`soft` blend NLL on ≥ 6/9 structured cases AND mean paired diff ≤ 0 nat
(S2 at least ties the imitation winner — it directly optimizes the metric);
(ii) broad controls: S2 blend advantage over global ≤ 0.02 nat on all 6;
(iii) S2 HARD-read NLL ≤ global on ≥ 8/9 structured (degrades gracefully).

Bars (i)-(iii) are read against the 300-epoch arm (`direct_300ep`), the config-matched form of
S2 — see "Implementation notes" below for why.

## X7 trigger (record verdict, do NOT ask the user unless it fires)

On toy D, `gap_closure = mean_over_seeds( (nll_global − best_selector_blend) / (nll_global −
oracle_x_nll) )`, where `best_selector_blend = min` blend NLL over `{soft, direct_300ep,
direct_3000ep}` per case, and `oracle_x_nll` is S1's honest bound. All quantities are NLLs
(lower is better) — `nll_global − best_selector_blend` is the gap actually closed by the best
available selector, `nll_global − oracle_x_nll` is the achievable gap. `x7_trigger_fires =
bool(gap_closure < 0.5)`. If it fires → the deployable-selector question is still open → present
at G-X7 (§0d gate point), do NOT ask the user otherwise. If `gap_closure >= 0.5` → X7 is
SKIPPED; write one ledger line recording the arithmetic.

## Selftest

On the K6 synthetic table the direct objective must reach blend NLL ≤ the soft-imitation arm's
blend NLL + 0.005.

## Script/artifacts

`automl_package/examples/capacity_ladder_s2.py` →
`capacity_ladder_results/S2/{PREREGISTRATION.md,s2_summary.json}`.

## Run

`AUTOML_DEVICE=cpu OMP_NUM_THREADS=4 ~/dev/.venv/bin/python -u
automl_package/examples/capacity_ladder_s2.py` (selftest first with `--selftest`).

## Cost

Minutes–1 h.

## Non-goals

No joint fine-tuning of router+ladder (that is H1's arm); no new selector architecture (head
stays (32,32) — its low capacity is the safety mechanism); no ladder retraining.

---

## Implementation notes (not part of the plan text, added by the authoring worker)

- **Two S2 epoch arms, one primary for bars.** The dispatch explicitly asks for a 300-epoch arm
  (matched to S1/K6's router config) AND a 3000-epoch arm "to check under-training (report
  both)", but states bars (i)-(iii) in terms of a single "S2" without naming which epoch count
  gates them. The implementation reads bars (i)-(iii) against `direct_300ep` only (the
  apples-to-apples, config-matched comparison against S1's own 300-epoch `soft` arm), and
  reports `direct_3000ep` alongside every case purely as the under-training diagnostic. This is
  a judgment call, not a plan quote — confirm or override before the real run if bars should
  instead be read against whichever epoch arm is better per case (equivalent to folding
  `direct_3000ep` into `best_selector_blend` for bars (i)/(iii) too, not just the X7 trigger).
- **X7 `best_selector_blend` spans both S2 arms.** The dispatch's X7 formula says "min blend NLL
  over `{S2, soft}` per case"; since S2 was trained at two epoch counts, the implementation reads
  this literally as the min over `{soft, direct_300ep, direct_3000ep}` — i.e. "S2" in the X7
  formula means the best available S2 result, not specifically the 300-epoch arm used for bars
  (i)-(iii). This is consistent with X7 being a "how much of the achievable gap does the best
  selector we've built capture" question, not a bars-parity question.
- **Sign convention.** Unlike S1 (whose plan text used ambiguous "blend ≥ hard" language), the
  S2 dispatch message states its bars directly in NLL terms ("blend ≤", "advantage ≤ 0.02 nat",
  "hard-read NLL ≤ global") and gives the X7 formula already in NLL space — no sign-convention
  resolution was needed here, unlike S1's bar (iii).
