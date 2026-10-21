# Nested-vs-feedforward training-step diagnosis (Task DSEL-1)

**Status:** complete. Settled by the user 2026-07-20; this note records the ruling, it does not
re-derive it.

## 1. The two training steps, contrasted

| | Training step | Nested? | Result |
|---|---|---|---|
| certified recurrent run (`automl_package/examples/depth_selection_toy.py:475`) | mean cross-entropy over **every** depth in the ladder, each depth's target being what is actually derivable at that depth | **yes** | pass |
| failed feed-forward run (`automl_package/examples/depth_composition_toy.py:487`) | `loss = ce(net(x_tr), y_tr)` — **one** target, full depth only | **no** | overfits (per-seed numbers below) |

Per-seed result for the failed run (train_acc / val_acc), read off the landed positive-control
cells directly (each row cites only its own cell so the value-match stays scoped):
- seed 0: train_acc 0.970, val_acc 0.432 (`automl_package/examples/capacity_ladder_results/D_TOY_PROBES/f5c_poscontrol_a5_seed0.json`)
- seed 1: train_acc 1.000, val_acc 0.744 (`automl_package/examples/capacity_ladder_results/D_TOY_PROBES/f5c_poscontrol_a5_seed1.json`)

Citations re-verified against source on 2026-07-20 (not carried over from the ticket):
- `automl_package/examples/depth_selection_toy.py:475`:
  `loss = sum(ce(net(x_tr_t[:, : t * n_gen]), y_tr_by_t[t]) for t in t_ladder) / len(t_ladder)`
  inside `train_anytime` (`depth_selection_toy.py:442-488`) — a T-indexed target
  (`y_tr_by_t[t]` = the T-step prefix product, `depth_selection_toy.py:467`) is supplied at
  **every** rung of `t_ladder`, so every exit is trained against the target it can actually
  produce (module docstring, `depth_selection_toy.py:445-456`: "no impossible-target
  corruption").
- `automl_package/examples/depth_composition_toy.py:487`: `loss = ce(net(x_tr), y_tr)` inside
  `train_clf` (`depth_composition_toy.py:437-494`) — a single full-depth forward pass against a
  single target `y_tr`; no per-depth output, no per-depth target, no sampling over depth anywhere
  in the function.

## 2. The pattern across every working capacity dial

Every capacity-selection mechanism in this programme that works trains a target at more than one
resolution/depth simultaneously:
- the classifier (ProbReg) samples the number of classes per example and is scored per class;
- width (sandwich schedule) trains a head per width;
- the certified recurrent depth run (`train_anytime`, above) trains a target at every rung of the
  depth ladder.

The feed-forward arm (`train_clf`) is the only one of the group trained against a single
full-depth target with no nesting, and it is the only one that failed to generalize.

## 3. Named mechanism

**No nesting ⇒ no pressure on intermediate depth to mean anything.** `train_clf` is a plain
fixed-depth network optimized against one target at the output layer only. Nothing in the loss
constrains what any intermediate layer represents, so the optimizer is free to use the full stack
purely to fit the training set — which it does completely, on both seeds, while generalizing poorly
(the per-seed train_acc/val_acc pair in section 1 above). This needs no further diagnosis via the
escalation ladder: every rung of that ladder is an optimization remedy (LR, clipping, width, init,
…), and the optimizer here is not under-fitting — it already fits the training set. MASTER
Decision 16 ("optimization exonerated before architecture is blamed") requires ruling out
under-fitting first; by inspection, the section-1 train_acc figures show the failed arm is not
under-fit on either seed, so Decision 16's "low on both train and val ⇒ under-fit, escalate
optimization" branch does not apply here — the escalation ladder is irrelevant to this failure.

## 4. Non-goals honored

This note does not walk the escalation ladder, does not re-run the failed control (`train_clf`'s
`f5c_poscontrol_a5_seed{0,1}.json` evidence is accepted as-is and unmodified), and does not change
any training code. Building a nested feed-forward arm is a separate, later task (DSEL-1b).

## 5. Positive-control artifact schema fix (in-scope defect, same task)

The positive-control artifacts (`automl_package/examples/capacity_ladder_results/D_TOY_PROBES/f5c_poscontrol_a5_seed{0,1}.json`)
did not record training-set size, so "was there enough data?" could not be answered from the
artifact alone. Fixed in `run_positive_control`
(`automl_package/examples/depth_composition_toy.py:816-849`): the returned/landed dict now carries
`"n_train": int(data["x_tr"].shape[0])` and `"n_val": int(data["x_val"].shape[0])`, added
immediately after `"chance"` in the output dict. The already-landed `D_TOY_PROBES` evidence files
are untouched (fix is forward-only, per the non-goals above); verified with a cheap 50-epoch smoke
run into a separate directory (`automl_package/examples/capacity_ladder_results/DSEL1/schema_smoke/`),
not the 40,000-epoch re-run.
