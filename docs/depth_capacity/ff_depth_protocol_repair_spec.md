# Feedforward-depth pilot — protocol repair spec (F5c-a)

*Strand: FlexNN core (`docs/plans/capacity_programme/flexnn-core.md`, Task F5c). This spec is authored
before any build or run. It is delivered for adversarial review; the reviewer fills
`## Review verdict (pending)` at the end — the author renders no GO/PARK call here.*

*Authority: MASTER Decisions 9, 14, 15, 16, 17. Scope: repair the protocol that invalidated the F5b
battery, and make the vanishing-vs-exploding question measurable. The pre-registered bars of
`ff_depth_toy_spec.md` §6 are FROZEN and restated verbatim in §5 below.*

*Every claim about code behaviour in this document cites a `file:line` that was opened while writing
it. Claims that could not be verified are written as ASSUMPTION or OPEN, never as fact.*

---

## 0. What is established, and what is not

Two propositions must not be conflated, and the repair is designed around keeping them apart.

**ESTABLISHED (verified on disk, 2026-07-20):** the F5b battery is invalid. Its 28 runs
(`automl_package/examples/capacity_ladder_results/D_TOY_PROBES/ff_depth_pilot_a5_seed{0,1}.json`)
cannot be read against any bar, for the four reasons enumerated in `flexnn-core.md` §F5b. The
verification behind each is reproduced in §1 and §4 of this spec.

**NOT ESTABLISHED:** that untied / non-weight-shared depth fails on this substrate. Every untied arm
in F5b failed to fit its own **training** set — untied-flat `d=10` reached train accuracy 0.055
(seed 0) / 0.055 (seed 1), untied-perstep `d=10` reached 0.089 / 0.102. Under MASTER Decision 16 an
arm low on both train and val is **under-fit**: an optimization finding, never a generalization
verdict. The purpose of F5c is to run the escalation ladder that Decision 16 requires *before* any
architecture claim is permitted, so that the eventual verdict — pass or fail — is about architecture.

**ALSO NOT ESTABLISHED — an explicitly quarantined prior claim.** A narrative record
(`RESUME.md`, "### Decisions", entry `[2026-07-18] F5b divergence diagnosed as genuine substrate
finding, not a bug`) states that the `d=10` arm "suffers vanishing gradients in the unnormalized Tanh
stack" and that "neither Xavier init nor lower LR rescues it past ~0.13 val_acc". **No artifact for
this exists anywhere in the repo.** A repo-wide search for gradient-norm artifacts
(`grep -rln "grad_norm\|grad-norm\|gradient norm\|clip_grad"` over `*.py *.json *.md *.csv`) returns
only unrelated work: `automl_package/examples/sep_heads_investigation/sep_heads_grad_norms.csv`
(ProbReg separate-heads investigation) and
`automl_package/examples/report_b_results/flexnn_revalidation_grad_check.json` (M0 revalidation).
Neither concerns the A5 depth stack. Per the MASTER Rule "No claim without an artifact — prose-only
findings do not exist", that diagnosis is treated throughout this spec as an **UNVERIFIED
HYPOTHESIS**, and §3 exists to measure it.

Two parts of that same narrative entry are moreover **contradicted by the artifacts**: it says the
`d=4`/`d=7` untied-flat arms "memorize" with "train CE → 0", but the recorded train accuracies are
0.489 / 0.195 (seed 0) and 0.494 / 0.182 (seed 1) — nothing near memorization at the strand's own
0.90 fit threshold. (See §4.4 for why even those numbers do not settle the question: `train_clf`
records train accuracy only at the best-**CE** checkpoint, so whether these arms later fit the train
set is not observable from the F5b artifacts at all.)

---

## 1. Training-loop diff (MASTER Decision 15)

Decision 15: *a battery reusing a toy that produced a certified result must diff its training loop
against the loop that produced that result; every difference is justified in writing IN THE TASK
before the run.*

### 1.1 The three loops, and which certified results they carry

| Loop | File:lines | Certified result it carries | Evidence |
|---|---|---|---|
| **T** (the one F5b used) | `automl_package/examples/depth_composition_toy.py:401-443` (`train_clf`) | **NONE at A5/L=10.** No certified result on this substrate rests on this loop. | see §1.2 |
| **G** (graded battery, D1b) | `automl_package/examples/depth_graded_toy.py:134-170` (`_train_mixed`) | S5, `RecurrentComposer` shared-readout, val acc at length 10 = **0.990 / 0.998 / 0.992** on seeds 0/1/2, all `trustworthy=true` | `.../D_TOY_PROBES/depth_graded_pilot_s5_seed{0,1,2}.json`, `arms.shared_readout.by_length["10"]` |
| **S** (anytime/selection, D8b) | `automl_package/examples/depth_selection_toy.py:442-488` (`train_anytime`) | **A5, L=10**, `RecurrentComposer`, S1 per-stratum held-out acc **0.949/0.957/0.946** (seed 0) and **0.938/0.931/0.901** (seed 1), `trustworthy=true` both seeds | `.../D_TOY_PROBES/depth_selection_gradedness_seed{0,1}.json`, `bars.s1_per_stratum` |

Loop **S** is the only certified loop on **this pilot's exact substrate** (A5, L=10). Loop **G** is
the certified loop on the same architecture at the same length over S5. F5b ran loop **T**, which
carries no certified A5/L=10 result at all. That is the parity breach Decision 15 names.

### 1.2 Line-by-line diff, T vs S (and G where it disambiguates)

Each row is either **JUSTIFIED** (a difference that must be kept, with the reason) or **REMOVE**
(an unjustified divergence that the repaired protocol restores to parity).

| # | Aspect | **T** — `depth_composition_toy.py` | **S** — `depth_selection_toy.py` | **G** — `depth_graded_toy.py` | Ruling |
|---:|---|---|---|---|---|
| 1 | **Learning rate** | `LR = 1e-2` (`:93`), hard-wired into `train_clf` at `:415` (`Adam(net.parameters(), lr=LR)`); **no CLI override exists** — `main()` (`:792-857`) has no `--lr` flag | `LR_DEFAULT = 3e-3` (`:112`), used at `:470`; `--lr` flag at `:819` | `LR_DEFAULT = 3e-3` (`:74`), used at `:144`; `--lr` flag at `:328`. The in-code comment is decisive: *"canonical LR (n=10 spike: 1e-2 stalls the deep unroll, 3e-3 reaches 0.99)"* | **REMOVE.** `1e-2` is the value the repo itself documents as stalling the deep unroll at `n=10`. Both certified loops use `3e-3`; every landed A5/S5 L=10 result JSON records `"lr": 0.003`. Repaired protocol: **`3e-3`**, exposed as a `--lr` flag so it is recorded, not implicit. |
| 2 | **Gradient clipping** | **none** — `step_fn` is `zero_grad → loss → backward → step` (`:420-424`), no `clip_grad_norm_` | `torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=GRAD_CLIP_MAX_NORM)` at `:477`, before every `opt.step()`; `GRAD_CLIP_MAX_NORM = 1.0` at `:114`, commented *"anytime-net trainability fix (root-cause #1: L=10 needs clipping to stay GD-trainable)"* | none (`:147-151`) — but G runs S5 with a **summed** multi-length loss, not A5/L=10 single-exit | **REMOVE.** The only certified **A5, L=10** loop clips at 1.0, and its own comment attributes L=10 trainability to that clip. Repaired protocol: **`clip_grad_norm_(…, max_norm=1.0)`** before every step, applied identically to every arm. |
| 3 | **Convergence-gate metric** | val **cross-entropy** only (`val_fn`, `:426-436`, returns `ce(net(x_val), y_val).item()` at `:430`) | val CE (mean over the T-ladder, `:483`) | val CE (mean over lengths, `:156`) | **JUSTIFIED as-is for T's *loss* series, but INSUFFICIENT.** All three gate on CE; only T's *bars* are read on accuracy. This is the Decision-17 breach — fixed additively in §4, not by removing the CE series. |
| 4 | **Loss aggregation / supervision** | single CE on the full-word product (`:422`) | **mean** CE over `T_LADDER = (2,4,6,8,10)` against the running **prefix** product (`:475`) | **sum** of CE over lengths `(4,6,8,10)` against each length's own product (`:149`) | **JUSTIFIED — keep T's single-exit form.** The F5a reviewer ruled explicitly (`ff_depth_toy_spec.md` §4 Fix 1, §11 item 2) that Cell 4 must be trained under the *same plain single-readout protocol as Cells 1–3*, because importing a multi-exit number would confound the 2×2 grid with a training-supervision difference. Consequence to state plainly: **the repaired protocol is still not byte-identical to any certified loop** — which is exactly why F5c-b (positive control alone) exists as a gate. |
| 5 | **Optimizer** | `torch.optim.Adam` (`:415`) | `torch.optim.Adam` (`:470`) | `torch.optim.Adam` (`:144`) | **No difference.** |
| 6 | **Batching** | full-batch (whole `x_tr` tensor per step, `:411`, `:422`) | full-batch (`:465`, `:475`) | full-batch (`:149`) | **No difference.** |
| 7 | **`check_every` / `patience` / `min_delta`** | `250 / 10 / 1e-4` (`:95-97`), passed at `:437` | `250 / 10 / 1e-4` — imported *from G* at `:99`, passed at `:487` | `250 / 10 / 1e-4` (`:76-78`), passed at `:160` | **No difference** (identical constants by import). But see §4.2: with `min_delta` on CE, these constants produce a 2,750-epoch stop on 12 of the 14 F5b arms. The constants are not the defect; the **metric** they are applied to is. |
| 8 | **`max_epochs` cap** | `MAX_EPOCHS = 40000` (`:94`) | `MAX_EPOCHS_DEFAULT = 40000` (`:113`) | `MAX_EPOCHS = 40000` (`:75`) | **No difference** for the primary runs. **Must be raised explicitly for the Decision-9 confirmation runs** — see §4.5. |
| 9 | **Best-weights restore** | inherited from `fit_to_convergence` (`automl_package/utils/convergence.py:194-196`): restores the state at best **val CE** | same | same | **REMOVE (change to the bar metric).** The restore is CE-driven, so it selects the reported number, not just the flag. Measured cost: `recurrent` seed 0 reports val acc **0.758** (its best-CE checkpoint, epoch 1000) while its val-accuracy trajectory peaked at **0.830** at epoch 3000. §4.3 specifies the fix. |
| 10 | **Seeding** | `torch.manual_seed(1000*seed + d)` per ladder rung (`:514`); `1000*seed + 777` for the wide control (`:526`) | `torch.manual_seed(seed)` (`:461`) | per-arm, in `run_pilot` | **JUSTIFIED.** T's scheme gives each ladder rung an independent init, which is what a depth ladder needs; S trains one net so it needs one seed. No parity issue. |
| 11 | **CPU thread cap** | `torch.set_num_threads(TORCH_THREADS)`, `TORCH_THREADS = 2` (`:98`, called at `:814`) | absent (verified: `grep -n set_num_threads` returns no hit in `depth_selection_toy.py` / `depth_graded_toy.py`) | absent | **JUSTIFIED.** Wall-clock only; cannot change the optimization trajectory of a deterministic full-batch run. Keep. |
| 12 | **Architecture / width defaults** | `NARROW_WIDTH = 16` (`:103`) is the module default, but the F5a spec pins flat arms at width **64** | n/a | n/a | **JUSTIFIED but a latent trap.** F5b's recorded parameter counts (19,004 / 14,844) confirm width 64 was passed explicitly via `--narrow-width`. The repaired driver must **pass width explicitly and record it in the JSON**, never rely on the module default. |

### 1.3 The repaired protocol, stated once

Every arm in F5c — positive control, grid cells, and wide-shallow controls alike — is trained with:

- Adam, **`lr = 3e-3`** (diff row 1), recorded in every output JSON;
- **`clip_grad_norm_(params, max_norm=1.0)`** before every `opt.step()` (diff row 2);
- full-batch, single-exit cross-entropy on the full-word product (diff row 4 — deliberately retained);
- `check_every=250`, `patience=10`, CE `min_delta=1e-4` (diff row 7, unchanged);
- the **dual-metric convergence gate** of §4 (CE **and** val accuracy), with best weights restored on
  **val accuracy** (diff rows 3 and 9);
- widths, depths and seeds passed explicitly and echoed into the JSON (diff row 12).

**Uniformity rule (binding).** Any change adopted from the §2 ladder is applied to **every arm in the
grid**, and the positive control is re-validated at the final protocol before any bar is read.
Escalating only the failing arm would re-introduce exactly the parity breach Decision 15 was written
to stop.

---

## 2. Escalation ladder (MASTER Decision 16)

Decision 16: *no arm is recorded as an architecture failure until either it is shown to fit the
training set, or a documented escalation ladder has been run.*

### 2.1 The two failure modes are genuinely different — evidence

All numbers below are from `ff_depth_pilot_a5_seed{0,1}.json`, read directly.

**Mode A — "learns, then goes over-confident."** Val CE climbs monotonically from the first
checkpoint while val accuracy sits flat or drifts down; train accuracy is substantial.

| Arm | seed | train acc | val acc | val CE trajectory (epoch:value) |
|---|---:|---:|---:|---|
| untied-flat `d=4` | 0 | 0.489 | 0.170 | 250:2.99 → 2750:13.57 |
| untied-flat `d=4` | 1 | 0.494 | 0.161 | 250:3.01 → 2750:14.08 |
| wide `W=435` | 0 | 0.959 | 0.135 | 250:5.02 → 2750:11.50 |
| wide `W=887` | 0 | 0.969 | 0.124 | 250:4.65 → 2750:9.34 |

For the **wide-shallow controls** this is not a defect at all: `ff_depth_toy_spec.md` §6 defines the
certified width-substitution failure mode as exactly *memorization — train ≥ 0.90 while val ≤ 0.60*.
`W=435` (train 0.959 / val 0.135) and `W=887` (train 0.969 / val 0.124) **are that mode**. They were
quarantined solely by a CE-based gate reading a metric no bar uses. Their repair is §4, not §2.

**Mode B — "does not learn at all."** Val CE is flat to three decimals across every checkpoint from
epoch 250 onward; val accuracy is pinned; train accuracy is at or barely above chance
(chance = 1/60 = 0.0167).

| Arm | seed | train acc | val acc | val CE trajectory |
|---|---:|---:|---:|---|
| untied-flat `d=10` | 0 | 0.055 | 0.051 | 250:3.750 … 2750:3.750 (bit-flat, 11 checkpoints) |
| untied-flat `d=10` | 1 | 0.055 | 0.056 | 250:3.729 … 4000:3.729 (bit-flat, 16 checkpoints) |
| tied-flat `d=7` | 0 | 0.088 | 0.076 | 250:3.450 → 500:3.750, flat thereafter |
| tied-flat `d=10` | 0 | 0.055 | 0.056 | 250:3.749 … 2750:3.755 |
| untied-perstep `d=10` | 0 | 0.089 | 0.071 | 250:3.556 → 2750:3.909 (slow rise, no learning) |

**Mode C — "learns, then collapses."** The positive control, seed 0 only: val accuracy climbs
0.149 → 0.830 over epochs 250–3000, then falls to **0.097** at epoch 3250 and 0.102 at 3500; val CE
0.832 (best, epoch 1000) → 3.245 at epoch 3250. Seed 1 shows none of this (val acc 0.910 at epoch
1000 rising smoothly to 0.932; CE best 0.339 at 1250, drifting up gently to 0.423).

**Answer to the brief's question: yes, these need different remedies, and one of them is not a
"remedy" at all.**
- Mode A on the **width controls** is the intended, certified behaviour → repaired by §4 (gate on the
  bar metric), not by the ladder.
- Mode A on **untied-flat `d=4`** (train 0.49, i.e. neither fitting nor memorizing) is under-fit by
  Decision 16 → ladder applies.
- Mode B is a pure optimization failure → ladder applies, and is the ladder's primary target.
- Mode C is late-training instability → the canonical remedy is gradient clipping, already adopted as
  a **parity repair** (§1.2 row 2) before the ladder is ever entered.

### 2.2 Rung 0 — parity repair (mandatory, not optional, not a ladder rung)

Apply §1.3 in full. This is not escalation: rows 1, 2, 3, 9 and 12 of the §1.2 table are *unjustified
divergences from the certified protocol*, and Decision 15 requires their removal regardless of
whether they turn out to help. The ladder below is entered **only if Rung 0 is insufficient**.

Prediction registered before the run (so it is falsifiable, not post-hoc): Rung 0 alone is expected
to fix Mode C (clipping is the documented L=10 fix) and to materially help Mode B (`3e-3` is the
documented n=10 fix). It is **not** expected to fix Mode A on `d=4`.
*This is a prediction, not a claim — it may be wrong, and the run decides.*

### 2.3 The ladder proper — fixed order

Each rung is applied on top of Rung 0 and all preceding rungs, uniformly to every arm, and the
positive control is re-run at each adopted rung (§2.5).

| Rung | Remedy | Concretely | Why here in the order |
|---:|---|---|---|
| **L1** | **LR sweep** | `lr ∈ {3e-3, 1e-3, 3e-4}` (half-decade log grid, anchored at the certified `3e-3`). One sweep, all arms, both seeds. | Cheapest, most reversible, zero architectural change, and the repo's own evidence says LR is the live variable at this depth (`depth_graded_toy.py:74`). |
| **L2** | **LR warmup** | linear ramp `0 → lr` over the first 500 epochs (2 × `check_every`), then constant. | Addresses early-training blow-up specifically, which is where the Mode-B arms die: every Mode-B trajectory is already dead at the **first** checkpoint (epoch 250). Still zero architectural change. |
| **L3** | **Init scheme** | `xavier_normal_(W, gain=calculate_gain('tanh'))` (gain 5/3) on every `Linear`, biases zero. | First rung with a real prior behind it (§2.4). Still zero architectural change: the *function class* is untouched, only the starting point. This is the **last** architecture-preserving rung. |
| **L4** | **Normalization / residual** | `LayerNorm` between blocks, or residual `state ← state + tanh(block(state))`. | **Changes the object under test.** See the ruling in §2.6 — L4 may not be substituted for the plain arm. |

### 2.4 Why L3 has a prior, and why it is nevertheless not a diagnosis

Verified from the installed PyTorch source
(`~/dev/.venv/lib/python3.12/site-packages/torch/nn/modules/linear.py:117-128`): `nn.Linear`
initialises `weight` with `kaiming_uniform_(a=sqrt(5))`, which the code's own comment states is
identical to `uniform(-1/sqrt(fan_in), 1/sqrt(fan_in))`, and `bias` with
`uniform(-1/sqrt(fan_in), 1/sqrt(fan_in))`.

From that, `Var(W) = (2/sqrt(fan_in))^2 / 12 = 1/(3·fan_in)`. For the width-64 hidden stack of
`build_narrow_clf` (`depth_composition_toy.py:285-293`), `fan_in = fan_out = 64`, so the
**per-layer variance gain is `fan_in · Var(W) = 1/3` in the forward direction and
`fan_out · Var(W) = 1/3` in the backward direction** (tanh contributes gain ≤ 1). Over the 10-layer
stack the linearised signal scale is therefore `(1/sqrt(3))^10 ≈ 5.7e-3`.

**ASSUMPTION, flagged as such:** this is a linearised, tanh-gain-≤-1 argument at initialisation. It
predicts strong attenuation; it does **not** establish that attenuation is what stops learning.

**And here is why the vanishing-gradient story cannot be asserted even with that arithmetic in hand:**
Adam is **per-parameter scale-invariant** (the update is `m̂ / (sqrt(v̂) + eps)`). A gradient that is
uniformly 100× smaller in an early layer produces a *comparably sized* Adam step there once the
second-moment estimate has adapted. So "the gradients are small" and "the early layers are not
learning" are **logically independent** propositions, and only the second one explains a bit-flat
loss. §3 therefore logs both, plus the saturation fraction that could explain a genuinely *zero*
(not merely small) gradient.

### 2.5 Stopping rule

**Stop at the first rung at which every arm in the grid satisfies the Decision-16 exoneration
condition: `train_acc ≥ FIT_ACC (0.90)`, or is demonstrably in the *memorization* mode
(`train_acc ≥ 0.90` and `val_acc ≤ 0.60`).** Concretely:

1. Apply Rung 0. Run F5c-b (positive control alone, both seeds).
2. If the positive control fails its bar (§5), escalate one rung and repeat step 1. **Never** proceed
   to any other arm — Decision 14.
3. Once the positive control passes, run the grid at that same rung.
4. If any grid arm is `UNDER_FIT` (§4.4), escalate one rung, re-validate the positive control at the
   new rung, and re-run the **whole** grid (uniformity rule, §1.3).
5. **Ladder exhausted** = L3 applied and at least one arm still `UNDER_FIT`. That arm is then recorded
   as **UNDER-FIT / optimization-unresolved**, explicitly *not* as an architecture failure, and its
   FF-CLAIM cell is reported as **NOT EVALUABLE**. Decision 16 forbids reading it as a verdict.
6. **Budget cap (pre-registered, so escalation cannot run away):** at most **one** full grid re-run
   per rung, at most **three** rungs (L1–L3). If that budget is exhausted, stop and report; do not
   improvise a fourth remedy.

### 2.6 Ruling on L4, flagged for the reviewer

Adding LayerNorm or a residual connection to Cell 1 means the arm is **no longer "a plain deep MLP"**,
which is the precise object the FF-CLAIM bar is written about (`ff_depth_toy_spec.md` §3, §6). This
spec therefore rules:

> **L4 is not a repair of Cell 1. If it is run, it is run as a SEPARATE, LABELLED arm
> ("untied-flat + residual" / "untied-flat + LayerNorm"), reported alongside the plain arm. The
> FF-CLAIM bar is read on the plain arm only, whatever L4 shows.**

The author flags this as the one place in the ladder where the reviewer may reasonably see a
principled-design fork rather than a mechanical fix, and defers to the verdict. The conservative
default, if the reviewer prefers it, is to drop L4 entirely and let the ladder end at L3.

---

## 3. Instrumentation plan — per-layer gradient norms

**Standing constraint (MASTER Rules, "No claim without an artifact"): after F5c-c, no statement about
vanishing, exploding, saturating or starving gradients may appear in any strand, verdict, RESUME entry
or report unless it cites a file produced by this section.** The prose diagnosis quarantined in §0
does not become citable by being repeated.

### 3.1 What is logged

Per training step selected by the cadence in §3.2, for the arm under diagnosis:

| Field | Definition | Why it is needed |
|---|---|---|
| `grad_norm_pre_clip` | L2 norm of `p.grad` over the parameters of **one layer/block**, measured **after `backward()` and before `clip_grad_norm_`** | the direct vanishing/exploding measurement |
| `grad_norm_global_pre_clip` | L2 norm over **all** parameters, pre-clip | lets the clip rate be computed: `clipped = global > max_norm` |
| `update_norm` | L2 norm of the **actual parameter change** for that layer across the step (`θ_after − θ_before`) | the decisive control for §2.4: Adam's rescaling means small gradients ≠ frozen parameters. Without this the vanishing hypothesis is untestable. |
| `param_norm` | L2 norm of that layer's parameters | makes `update_norm` interpretable as a *relative* step size |
| `preact_absmean`, `preact_std` | mean `\|z\|` and std of that layer's pre-activation on a fixed 512-row probe batch from the train split | locates where the forward signal dies |
| `sat_frac` | fraction of that layer's units with `\|tanh(z)\| > 0.99` on the same probe batch | a saturated tanh gives a genuinely ~zero local gradient, which Adam cannot rescue — the one mechanism that *would* explain a bit-flat loss |
| `train_ce`, `train_acc`, `val_ce`, `val_acc` | at the same rows | ties the mechanism to the failure it is supposed to explain |

The probe batch is **fixed across the whole run and across arms** (same 512 row indices, chosen once
from the train split with the run's seed) so `preact_*` and `sat_frac` are comparable step-to-step and
arm-to-arm.

### 3.2 Cadence — driven by evidence, not by convenience

The Mode-B failure is **already complete at the first recorded checkpoint**: untied-flat `d=10`
seed 0 reads val CE `3.750` at epoch 250 and `3.750` at every one of the following 10 checkpoints.
A logger at the existing `CHECK_EVERY = 250` cadence would therefore record *nothing but the corpse*.
Cadence:

- **epochs 1–250: every epoch** (250 rows/layer — where the failure actually happens);
- **epochs 251–2,500: every 10 epochs** (225 rows/layer);
- **epochs 2,501 → end: every 250 epochs** (matches `CHECK_EVERY`, so grad rows align with the
  convergence trajectory; this window is where Mode C's collapse at epoch 3,250 lives).

Order per instrumented step: `zero_grad → forward → backward → record grad_norm_pre_clip +
grad_norm_global_pre_clip → clip → snapshot θ → step → record update_norm` .

### 3.3 Which arms

F5c-c per `flexnn-core.md`: **untied-flat `d ∈ {4, 7, 10}`**, both seeds — the three Mode-A/Mode-B
points of the headline arm. Add, at no meaningful extra cost and with a clear purpose:

- the **positive control** (`recurrent`, `d=10`), both seeds — the *reference* gradient profile. A
  vanishing/exploding claim about the untied stack is uninterpretable without the profile of an arm
  that demonstrably learns the same task. Seed 0 additionally carries the Mode-C collapse, so it is
  the only place an *exploding*-gradient signature could be caught in the act.

That is 8 instrumented runs. Everything else in the grid is diagnosed by inference from these, or not
at all.

### 3.4 Artifact files

Reusing the shape of the existing precedent
`automl_package/examples/sep_heads_investigation/sep_heads_grad_norms.csv` (columns
`epoch,head,grad_norm,strategy`) rather than inventing a new schema:

- **`automl_package/examples/capacity_ladder_results/D_TOY_PROBES/ff_depth_gradnorms_{net}_d{d}_seed{s}.csv`**
  — long format, one row per `(epoch, layer_index)`:
  `epoch,layer_index,layer_name,grad_norm_pre_clip,grad_norm_global_pre_clip,update_norm,param_norm,preact_absmean,preact_std,sat_frac,train_ce,train_acc,val_ce,val_acc,net_kind,depth,seed,lr,clip_max_norm`
- **`.../ff_depth_gradnorms_summary_seed{s}.json`** — one object per instrumented arm, carrying: the
  protocol block (`lr`, `clip_max_norm`, rung, init scheme); `first_to_last_layer_grad_ratio` at
  epochs {1, 10, 50, 250, 1000, final}; `clip_engagement_rate` (fraction of instrumented steps with
  `grad_norm_global_pre_clip > max_norm`); `max_sat_frac` per layer; and the CSV's own path, so the
  summary is never read without its source.

### 3.5 Pre-registered readings (so the diagnosis is decided by data, not by narrative)

Registered before the run. The mechanism claim that F5c-c is permitted to write is whichever of these
the artifact matches; if it matches none, the permitted claim is *"no mechanism identified"*.

| Signature in the artifact | Permitted conclusion |
|---|---|
| `grad_norm_pre_clip` decreasing roughly geometrically toward the input layer, `first_to_last_layer_grad_ratio ≪ 1` (order `(1/sqrt3)^(d-1) ≈ 5.7e-3` at `d=10`, §2.4), **and** `update_norm` correspondingly suppressed in early layers | **Vanishing gradients confirmed**, and they are load-bearing (Adam did not rescue them). L3 (init) is the indicated remedy. |
| Small `grad_norm_pre_clip` in early layers but **comparable `update_norm` across layers** | **Vanishing gradients present but NOT the cause** — Adam rescaled them. The prose diagnosis in §0 is then *refuted*, and the cause must be sought elsewhere. |
| `sat_frac → 1` in the early layers | **Saturation**, not scale-vanishing: the local derivative is ~0 and Adam cannot rescue it. Indicated remedies: L3 (smaller effective init) and/or L4-class normalization. |
| `grad_norm_global_pre_clip` spikes coincident with the CE/accuracy collapse (positive control seed 0, epochs 3,000–3,250) | **Exploding gradients** at that event; clipping (already in Rung 0) is the indicated remedy, and the artifact then *retroactively justifies* diff row 2. |
| Flat, small, uniform norms with `update_norm > 0` throughout and no saturation | **Neither vanishing nor exploding** — the arm is optimizing but on a bad objective landscape. No gradient-pathology claim may be made. |

---

## 4. Convergence gate on val ACCURACY as well as CE (MASTER Decision 17)

Decision 17: *where a pre-registered bar is read on metric M, the `trustworthy`/`diverged` flags are
computed on M (or on M and the loss, with both reported).* The bars of `ff_depth_toy_spec.md` §6 are
read on **held-out word accuracy**. F5b gated on val cross-entropy alone.

### 4.1 What the CE-only gate actually did — four measured harms

**(a) It certified dead nets as trustworthy.** untied-flat `d=10` (train 0.055, val 0.051, val CE
bit-flat at 3.750 for 11 checkpoints) carries `trustworthy=true`, `diverged=false`. The mechanism is
plain in the source: a flat series never triggers the divergence test
(`automl_package/utils/convergence.py:53-66` compares final-to-best against
`max(DIVERGENCE_ABS_EPS=0.2, 0.5·best_val)`; for a flat series that difference is 0), and
`still_improving` (`:47-50`) is likewise 0, so `trustworthy` (`:68-71`) is `True`. **A net that never
learned passes the gate, while a net that reached 0.758 fails it.**

**(b) It quarantined the valid width-substitution stalls.** All six wide-shallow controls are
`trustworthy=false` on both seeds, purely on CE explosion — including `W=435` (train 0.959 / val
0.135) and `W=887` (train 0.969 / val 0.124), which are textbook instances of the memorization mode
that `ff_depth_toy_spec.md` §6 *pre-registers as the certified failure signature*. Their val-accuracy
trajectories are flat to ±0.01 across every checkpoint; nothing about them is untrustworthy on the
metric the bar reads.

**(c) It truncated training on essentially every arm.** With CE `min_delta = 1e-4` and CE rising from
the first checkpoint, the patience clock starts at checkpoint 1 and fires
`250 + 10 × 250 = 2,750` epochs later. Measured: **12 of 14 arm-runs on seed 0** and **10 of 14 on
seed 1** stopped at exactly epoch 2,750 — against a 40,000 cap. The only certified-comparable arms in
the repo ran far longer under an equivalent gate: `depth_selection_gradedness_seed0.json` stopped at
**4,000**, and the graded S5 pilots at **8,000 / 10,500 / 7,750**
(`depth_graded_pilot_s5_seed{0,1,2}.json`).

**(d) It selected the reported number, not just the flag.** `fit_to_convergence` restores the weights
at best val **CE** (`automl_package/utils/convergence.py:194-196`), and `train_clf` computes both
accuracies *after* that restore (`depth_composition_toy.py:439-443`). Positive control seed 0: best CE
at epoch 1,000 → reported val acc **0.758**; its val-accuracy trajectory reads 0.830 at epoch 3,000.
The CE-driven restore discarded 7.2 accuracy points. In the other direction, seed 1's accuracy was
still climbing (0.910 @1,000 → 0.932 @3,750) long after its CE bottomed at epoch 1,250.

### 4.2 The repaired gate

**Reuse, do not reinvent.** `automl_package/utils/convergence.py:89-150` already provides
`ConvergenceTracker`, a standalone patience/best state machine over **one** lower-is-better series,
extracted precisely so that "the same flag semantics" can be applied to several series at once (see
its own docstring). The repair runs a **second** tracker on `−val_accuracy` — negated so the
lower-is-better contract holds unchanged and no tracker logic is duplicated.

**Constants for the accuracy series** (these are **gate** constants, not bars — see §5):

| Constant | Value | Justification |
|---|---:|---|
| `ACC_MIN_DELTA` | `1e-3` | `n_val = 10,000`, so one example is `1e-4`; `1e-3` = 10 examples — above split jitter, far below any improvement that matters at a 0.90 bar. |
| `ACC_CHECK_EVERY` | `250` | identical to the CE series (`depth_composition_toy.py:95`), so the two trajectories are index-aligned and directly comparable. |
| `ACC_PATIENCE` | `10` | identical to the CE series (`:96`), for the same reason. |
| `ACC_DIVERGENCE_ABS_EPS` | `0.05` | accuracy is bounded in [0,1], so the nats-scaled `DIVERGENCE_ABS_EPS = 0.2` (`utils/convergence.py:31`) is meaningless here. 5 pp is well above checkpoint-to-checkpoint jitter in the observed trajectories (≤0.01 on every stable arm). |
| `ACC_DIVERGENCE_REL_FACTOR` | `0.25` | mirrors the existing relative rule (`:32`) at a scale suited to a bounded metric. |

`diverged_acc := (best_acc − final_acc) > max(0.05, 0.25 · best_acc)`. Checked against the artifacts:
positive control seed 0 — best 0.830, final 0.102 → `0.728 > max(0.05, 0.2075)` → **diverged ✓**
(correctly catches Mode C, which the CE flag also caught, but now on the bar's own metric). Positive
control seed 1 — best 0.932, final 0.932 → `0 >` nothing → **not diverged ✓**. untied-flat `d=10` —
best 0.051, final 0.051 → **not diverged**, which is correct: it did not diverge, it never learned.
That case is caught by §4.4, not by the divergence flag.

> **ASSUMPTION requiring reviewer sign-off:** the five constants above are new. They are *gate*
> parameters (how a run is judged converged), not *bars* (what threshold a result must clear), and no
> value in `ff_depth_toy_spec.md` §6 is touched by them. The author flags them explicitly because
> Decision 9 forbids moving bars after a run, and wants the boundary between the two audited rather
> than assumed.

**Stopping rule:** the loop stops when **both** series have flattened (logical AND), or `max_epochs`
is hit. AND, not OR, because an OR would leave the CE clock firing at 2,750 on arms whose accuracy is
still climbing — harm (c).

**Best-weights restore:** on the **bar metric**, val accuracy — harm (d). The CE series continues to
be tracked, trajectory-logged and reported in full; it simply no longer selects the weights.

**Reported flags — both metrics, side by side, in every JSON:**
`converged_ce`, `diverged_ce`, `still_improving_ce`, `trustworthy_ce`, `stop_epoch_ce`,
`best_val_ce`, `best_epoch_ce`, `trajectory_ce`, and the same seven for `_acc`, plus `hit_cap`.
**A cell is readable against a bar iff `trustworthy_acc` is true**; `trustworthy_ce` is reported for
transparency and is not a gate on an accuracy-read bar. Any cell where the two disagree is called out
explicitly in the verdict rather than silently resolved.

### 4.3 Trajectories that must be recorded

`train_clf` already records a val-accuracy trajectory (`depth_composition_toy.py:417`, `:434`,
attached at `:438`) — that plumbing is reused, not rebuilt. Two additions are required:

1. **A train-accuracy trajectory at the same cadence.** This is the single most consequential gap in
   the F5b artifacts: `train_acc` is computed once, after the best-**CE** weights are restored
   (`:439-443`), so it reports the train accuracy at (typically) epoch 250 — **not** at the end of
   training. Decision 16's exoneration test is literally *"is it shown to fit the training set"*, and
   the F5b artifacts cannot answer it. Every claim in §0 about whether the `d=4`/`d=7` arms memorized
   is unanswerable for exactly this reason.
2. **A train-CE trajectory at the same cadence**, so train/val CE can be compared and the
   overfit-vs-underfit distinction read off the curves rather than inferred.

### 4.4 Mechanizing Decision 16 in the gate — the `fit_status` field

Every arm's JSON carries a `fit_status` computed from the trajectories, using **only the already-frozen
thresholds** `FIT_ACC = 0.90` and `STALL_ACC = 0.60` (`depth_composition_toy.py:110-111`) — no new
threshold is introduced:

| `fit_status` | Condition (at the reported, accuracy-best checkpoint) | Consequence |
|---|---|---|
| `GENERALIZED` | `val_acc ≥ 0.90` | eligible for a FIT read |
| `MEMORIZED` | `train_acc ≥ 0.90` and `val_acc ≤ 0.60` | the certified width-substitution stall (`ff_depth_toy_spec.md` §6) — a **valid** stall |
| `UNDER_FIT` | `max over the trajectory of train_acc < 0.90` | **Decision 16: optimization finding only.** No architecture verdict. Escalate the ladder (§2.5) or report NOT EVALUABLE. |
| `INTERMEDIATE` | none of the above | reported as-is; no bar read |

`UNDER_FIT` is deliberately evaluated on the **maximum over the train-accuracy trajectory**, not on
the endpoint — so an arm that fits the train set and later destabilizes is not misfiled as never
having fit it.

### 4.5 Satisfying MASTER Decision 9 (early-stop-OFF confirmation at ≥ 4× budget)

Decision 9: *load-bearing verdicts get an early-stop-OFF confirmation run at ≥ 4× the self-terminated
budget.* F5b never ran one (`flexnn-core.md` §F5b ground 4).

**Which cells are load-bearing — pre-registered here, before any run, so the set cannot be chosen to
suit an outcome:**
1. the **positive control**, both seeds (F5c-b) — it gates the whole battery under Decision 14;
2. **untied-flat at every `d ∈ {4,7,10}`**, both seeds — it carries FF-CLAIM, and its *failure* is
   the headline if it fails, so a failure needs the same confirmation a pass would;
3. **any cell reaching `val_acc ≥ 0.90`**;
4. **the wide-shallow control param-matched to any cell reaching ≥ 0.90** — the falsifier side of
   FF-CLAIM (`ff_depth_toy_spec.md` §5 reading rule).

**Protocol.** Let `E_stop` be the primary run's accuracy-gated stop epoch. The confirmation run
repeats that arm at the identical protocol, seed and init, with **early stopping disabled** and
`max_epochs = 4 · E_stop`, rounded up to a multiple of `CHECK_EVERY = 250`, and with **`max_epochs`
passed explicitly**.

> **Concrete trap, stated because it is easy to walk into:** the module cap is `MAX_EPOCHS = 40000`
> (`depth_composition_toy.py:94`), and `train_clf`'s `max_epochs` argument defaults to it (`:401`).
> Certified comparable stop epochs are 4,000 (`depth_selection_gradedness_seed0.json`) and
> 7,750–10,500 (`depth_graded_pilot_s5_seed{0,1,2}.json`); 4× those is 16,000–42,000. A confirmation
> run left on the default cap would hit it, and `hit_cap=True` forces `trustworthy=False`
> (`automl_package/utils/convergence.py:68-71`) — the confirmation would invalidate itself. The
> driver must accept and record an explicit `--max-epochs`, and the run is invalid if
> `4 · E_stop > max_epochs`.

**How it is read.** The bar is read from the **primary** run's accuracy-best weights. The
confirmation run's job is strictly to falsify: (i) no arm that was recorded as failing crosses its bar
later; (ii) no arm that was recorded as passing collapses later; (iii) `diverged_acc` on the full
extended trajectory. Both full trajectories (CE and accuracy) are stored in the confirmation JSON. Any
crossing found is not silently absorbed — it is escalated to the adjudicator per the unattended-run
contract rule 4.

**Output.** `.../D_TOY_PROBES/ff_depth_confirm_{net}_d{d}_seed{s}.json`, carrying `E_stop`,
`max_epochs`, `early_stop=false`, both trajectories, both flag sets, `fit_status`, and the path of the
primary run it confirms.

---

## 5. Bars — RESTATED UNCHANGED AND FROZEN

**No bar in this spec is new, moved, relaxed or reinterpreted.** MASTER Decision 9 forbids adjusting a
bar after a run, and F5b was a run. The following is a verbatim restatement of
`docs/depth_capacity/ff_depth_toy_spec.md` §6.

- **FIT threshold = 0.90.** An arm "fits" iff held-out accuracy ≥ **0.90**.
- **STALL threshold = 0.60.** A width control "stalls" iff held-out accuracy ≤ **0.60**.
- **Chance = 0.0167** (1/60).
- **Metric: held-out (val) word accuracy**, cross-entropy classification.
- **FF-CLAIM (the headline) PASSES iff both hold:** (1) the untied-flat arm (Cell 1) reaches held-out
  **≥ 0.90 at some `d ≤ 10`**, **and** (2) the wide-shallow control **param-matched to that winning
  depth** (188 / 311 / 435 for `d = 4 / 7 / 10`) stays **≤ 0.60**.
- **Non-substitutability falsifier:** FF-CLAIM is falsified if the param-matched width control also
  reaches ≥ 0.90. A width control low on *both* train and val is under-fit, not a clean
  width-substitution failure, and must be flagged rather than counted as a stall.
- **Seeds: ≥ 2.** A bar is read as passed only if it holds on ≥ 2 **trustworthy** seeds.
- **Convergence-gated (hard gate).** Every reported number comes from a trustworthy, non-diverged
  trajectory; no number is read from an endpoint alone.

These values are the certified graded pilot's own constants, `FIT_ACC = 0.90` and `STALL_ACC = 0.60`
at `automl_package/examples/depth_composition_toy.py:110-111` (identically
`depth_graded_toy.py:80-81`), and they remain the module constants the repaired driver reads. **The
matched-control widths {188, 311, 435, 146, 123, 887} of `ff_depth_toy_spec.md` §5 are likewise
frozen and unchanged.** The 2×2 architecture grid of §3/§4 of that spec is unchanged.

**What §4 of *this* spec changes and what it does not.** It changes *which trajectory the gate reads
to decide a run is finished and trustworthy*, and *which checkpoint's weights are reported*. It does
not change any threshold a result must clear, nor the metric the bars are read on — that metric was
always held-out accuracy; F5b simply gated on a different one.

**Bars added by this spec: none.** The stage gate of F5c-b (positive control ≥ 0.90 and
`trustworthy` on both seeds) is not a new bar — it is `ff_depth_toy_spec.md` §6's FIT threshold and
2-seed rule applied to the certified arm, as `flexnn-core.md` §F5c-b already specifies.

---

## 6. Open questions and assumptions (for the reviewer)

1. **§2.6 — L4 (normalization / residual) is ruled to be a separate labelled arm, never a repair of
   Cell 1.** This is the one ladder decision the author judges could be seen as a design fork rather
   than a mechanical fix. Conservative alternative: end the ladder at L3.
2. **§4.2 — five new gate constants** (`ACC_MIN_DELTA`, `ACC_CHECK_EVERY`, `ACC_PATIENCE`,
   `ACC_DIVERGENCE_ABS_EPS`, `ACC_DIVERGENCE_REL_FACTOR`). Each is justified against the observed
   trajectories, none is a bar, but they are pre-registered numbers and the author wants them audited
   as such.
3. **§1.2 row 4 — the repaired protocol is deliberately NOT byte-identical to any certified loop.**
   The single-exit supervision is retained because the F5a reviewer required it
   (`ff_depth_toy_spec.md` §4 Fix 1) to keep the 2×2 grid free of a supervision confound. The residual
   risk is therefore real and named: plain single-readout `RecurrentComposer` on A5/L=10 has **never**
   been certified, at any LR, in this repo. F5c-b exists to settle that, and its failure would be a
   substantive finding about the *substrate*, not merely a protocol problem.
4. **§2.4 — the init-scale arithmetic is a linearised argument** from the verified PyTorch default
   init, not a measurement. It motivates rung L3's position in the order; it establishes nothing about
   cause. §3 measures.
5. **OPEN — how the ladder interacts with the frozen matched-control widths.** The widths in
   `ff_depth_toy_spec.md` §5 are matched to arm parameter counts. Rung L4, if ever run, would change
   an arm's parameter count (LayerNorm adds parameters), so its matched control would have to be
   recomputed. The author's reading is that this is a further reason L4 belongs in a separate labelled
   arm with its own separately-computed control (§2.6) rather than inside Cell 1; the reviewer should
   confirm. Rungs L1–L3 change no parameter count, so the frozen widths stand unchanged for them.
6. **OPEN — a gate-baseline bookkeeping item, not a design question.**
   `docs/plans/capacity_programme/gates_baseline.txt` lists this spec's own path as a not-yet-existing
   future artifact. Now that the file exists, `test_baseline_shrink_only` in
   `docs/plans/capacity_programme/test_plan_gates.py` fails until that one line is removed. The plan
   directory is orchestrator-owned (MASTER Rules, "Single writer"), so this spec does not touch it;
   the removal is flagged here for the orchestrator.

---

## Review verdict — SOUND-WITH-FIXES ⇒ **GO for F5c-b** (adjudicator, 2026-07-20)

**Verdict: SOUND-WITH-FIXES.** Seven fixes are required (M1–M7 below). Every one is mechanical —
plumbing, a missing pre-registered constant, a missing selection rule, a missing terminal branch. None
is a principal-investigator design decision, so per `flexnn-core.md` §F5a the fixes are folded and
F5c-b proceeds. Nothing in this spec is PARKED.

Every number below was re-derived from the artifacts and the source while writing this section; the
author's own numbers were checked, not accepted.

### Findings on the five contract items

**§1 training-loop diff (Decision 15) — CONFIRMED.** Spot-checked every cited line. `LR = 1e-2`
(`automl_package/examples/depth_composition_toy.py:93`); `MAX_EPOCHS/CHECK_EVERY/PATIENCE/MIN_DELTA =
40000/250/10/1e-4` (`:94-97`); `NARROW_WIDTH = 16` (`:103`); `FIT_ACC/STALL_ACC = 0.90/0.60`
(`:110-111`). `LR_DEFAULT = 3e-3` with the decisive comment at
`automl_package/examples/depth_graded_toy.py:74`; `GRAD_CLIP_MAX_NORM = 1.0` with its L=10
trainability comment at `automl_package/examples/depth_selection_toy.py:114`, applied at `:477`.
Certified JSONs record `"lr": 0.003` (`depth_selection_gradedness_seed0.json`,
`depth_graded_pilot_s5_seed0.json`); the F5b JSONs record no `lr` key at all. The certified numbers
the diff cites are exact: S1 per-stratum `{6: 0.9487, 8: 0.9573, 10: 0.946}` seed 0; S5
shared-readout at length 10 = `0.9903 / 0.9985 / 0.9923` with stop epochs `8000 / 10500 / 7750`, all
`trustworthy=true`. Rulings JUSTIFIED/REMOVE are correct as written.

**§2 escalation ladder (Decision 16) — CONFIRMED, with M4 and M7.** The three-mode taxonomy is
verified against the artifacts: `mlp` train acc `0.489/0.195/0.055` (seed 0) and `0.494/0.182/0.055`
(seed 1) at `d = 4/7/10`; `untied_perstep` `0.089/0.102`; `wide_w435` `0.959/0.947`, `wide_w887`
`0.969/0.972`, all with val ≤ 0.135 — i.e. exactly the memorization mode `ff_depth_toy_spec.md` §6
pre-registers. Order is right: MASTER Decision 16's canonical ladder is *LR sweep → clipping → warmup
→ init → normalization*, and the spec preserves that relative order (LR correction and clipping in
Rung 0, then L1 LR sweep, L2 warmup, L3 init, L4 normalization). Bundling the LR correction with
clipping in Rung 0 is **not** a shortcut: Decision 15 requires both as parity restorations regardless
of whether they help, so they are not ladder rungs and no ordering claim is being made about them.
The cost is attribution (which of the two rescued the run) — recoverable cheaply via R1.
The two failure modes do have different mechanisms, but a **single uniform ladder is the correct
response**: per-arm remedies would re-create the parity breach the whole repair exists to close, and
the ladder's rungs cover both modes (clipping for instability, LR/warmup/init for no-learning).

**§3 instrumentation — CONFIRMED.** The plan measures what §0 quarantines. `update_norm` is the
decisive field: §2.4's observation that Adam's per-parameter rescaling makes "small gradients" and
"frozen early layers" logically independent is correct, and without `update_norm` the vanishing
hypothesis is untestable. The cadence argument is verified — `mlp d=10` seed 0 reads val CE `3.750`
at epoch 250 and `3.750` at all 10 following checkpoints, so a 250-epoch-cadence logger would record
only the corpse. §3.5's pre-registered readings are falsifiable and include a row that *refutes* the
quarantined prose diagnosis; that is the right shape.

**§4 dual-metric gate (Decision 17) — SOUND IN DESIGN, DEFECTIVE IN PLUMBING → M1, M2, M3.** The four
measured harms in §4.1 all reproduce (see the constants audit below), including the stop-epoch count:
**12 of 14** seed-0 arm-runs and **10 of 14** seed-1 arm-runs stopped at exactly epoch 2,750.

**§5 bars — CONFIRMED FROZEN. No bar has moved.** §5 was compared clause-by-clause against
`docs/depth_capacity/ff_depth_toy_spec.md` §6: FIT 0.90, STALL 0.60, chance 0.0167, metric =
held-out (val) word accuracy, FF-CLAIM's two conditions with 188/311/435 for `d = 4/7/10`, the
non-substitutability falsifier, ≥ 2 trustworthy seeds, convergence-gated. All identical. The six
matched-control widths {188, 311, 435, 146, 123, 887} are those of `ff_depth_toy_spec.md` §5 and are
unchanged; the corresponding arm parameter counts (19,004 / 31,484 / 43,964 / 14,844 / 12,476 /
89,660) are likewise untouched. **Changing which metric the convergence gate reads is not a bar
move** — it is mandated by MASTER Decision 17, which was written from the F5b case itself and
post-dates the toy spec. §5's own "what this changes and what it does not" paragraph states the
boundary correctly.

### Ruling on the five new gate constants (brief item 2) — GATE, not bars; but the set is incomplete

Reproduced against `ff_depth_pilot_a5_seed{0,1}.json` by replaying every accuracy trajectory through
`cvg.ConvergenceTracker(patience=10, min_delta=1e-3)`:

- positive control seed 0 — best `0.830` @3000, final `0.102` → drop `0.728 > max(0.05, 0.2075)` →
  **diverged ✓**, exactly as claimed;
- positive control seed 1 — best `0.932` @3750, final `0.932` → **not diverged ✓**;
- `mlp d=10` — best `0.051`, final `0.051` → **not diverged**, correctly left to `fit_status` ✓.

`ACC_MIN_DELTA`, `ACC_CHECK_EVERY` and `ACC_PATIENCE` are pure stopping-rule parameters and are
**gate constants, not bars** — confirmed. `ACC_DIVERGENCE_*` decide only whether a cell is *readable*,
not what it must clear, so they too are gate constants — confirmed. But three defects follow.

**M1 (required). The accuracy flags must be computed explicitly; they cannot be read off
`ConvergenceResult`.** `diverged`, `still_improving` and `trustworthy` are properties hard-wired to
module globals — `_STILL_IMPROVING_EPS = 5e-3` (`automl_package/utils/convergence.py:28`),
`DIVERGENCE_ABS_EPS = 0.2` (`:31`), `DIVERGENCE_REL_FACTOR = 0.5` (`:32`), consumed at `:45-71` — with
no per-instance override. On a negated (`−accuracy`) series `best_val` is **negative**, so
`max(0.2, 0.5·best_val)` collapses to the constant `0.2` and *both* declared `ACC_DIVERGENCE_*`
constants are silently discarded. §4.2's "reuse, do not reinvent" is right about `ConvergenceTracker`
(the patience/best state machine *is* parameterized) and wrong about the flag layer. The driver must
compute `diverged_acc`, `still_improving_acc` and `trustworthy_acc` itself from the tracker's
trajectory using the pre-registered constants, and must never read `result_acc.trustworthy`.

**M2 (required). A sixth constant is missing: `ACC_STILL_IMPROVING_EPS`.** `trustworthy` also depends
on `still_improving`, which the spec's table does not pre-register. Measured: replaying the F5b
accuracy trajectories through the tracker, the nats-scaled `5e-3` fires on **8 of 28** arm-runs —
including the **positive control seed 1** (`recent_improvement = +0.0067`), the one clean seed, and
`wide_w435` seed 1 (`+0.0095`). Those series were truncated at 2,750 by the CE gate and would run
longer under the repaired AND rule, so this is a hazard rather than a certainty; the binding point is
that an unpre-registered, wrong-scale constant participates in `trustworthy_acc`, which is the flag
that decides readability. Pre-register it with a justification in the §4.2 table.

**M3 (required). Drop `ACC_DIVERGENCE_REL_FACTOR`; use the plain absolute `0.05`.** For a bounded
metric the relative term can only *loosen* the flag, and it loosens most where the bar is read: at
`best_acc = 0.93` it tolerates a 23-point collapse before flagging, versus 5 points from the absolute
rule. Checked against all 28 F5b arm-runs: the plain `0.05` rule flags exactly one run (positive
control seed 0, drop `0.728`) and zero others — the next-largest drop anywhere is `0.025`
(`tied_flat d=7` seed 0). Identical verdicts on this data, strictly more sensitive at high accuracy,
zero observed false positives. This tightens the gate, so it is not a relaxation.

### Ruling on L4 (brief item 4) — the author's call is CORRECT and is not an overreach

**Affirmed as written: L4 is a separate, labelled arm, never a repair of Cell 1.** Two independent
grounds, both verified:

1. **The bar is written about a specific object.** `ff_depth_toy_spec.md` §6's FF-CLAIM is read on
   "the untied-flat arm (Cell 1)", which §3 defines as the plain deep MLP built by
   `build_narrow_clf` (`depth_composition_toy.py:285-293`) — `Linear→Tanh` stacked, nothing else.
   LayerNorm or a residual changes the function class, so an L4 arm is not the object the bar
   describes.
2. **It breaks the frozen param match.** The matched widths are pinned to exact counts with a ±0.3%
   tolerance (`ff_depth_toy_spec.md` §5: 19,004→188, 31,484→311, 43,964→435). LayerNorm adds
   parameters, so an L4 Cell-1 arm has no valid frozen control and could not be read at all.

This is therefore the **bar-preserving** ruling, not a design fork — the alternative (silently
repairing Cell 1 with L4) is what would have required a PI decision. **L4 is permitted as an optional
labelled arm with its own separately computed matched control; dropping it is not required.** This
also settles §6 item 5 in the affirmative: rungs L1–L3 change no parameter count, so the frozen widths
stand unchanged for the entire ladder as run.

### Ruling on the Decision-9 trap (brief item 6) — mitigation is PARTIAL → M5

The spec correctly identifies the cap-magnitude trap (`MAX_EPOCHS = 40000` at `:94`, also
`train_clf`'s default at `:401`; 4× certified stops = 16k–42k). Its mitigation — explicit
`--max-epochs`, run invalid if `4·E_stop > max_epochs` — handles magnitude but not **flag semantics**,
which is the sharper edge of the same trap.

**M5 (required).** `ConvergenceTracker.result()` sets `hit_cap = not converged`, and `trustworthy`
requires `not hit_cap` (`convergence.py:68-71`). If "early stopping disabled" is implemented by
inflating `patience`, the tracker never converges, `hit_cap` is True by construction, and the
confirmation run self-invalidates *exactly as §4.5 warns* — at any cap value. Specify instead: early
stop OFF = **suppress the loop break only**; both trackers keep their full patience bookkeeping so
`converged` and `stop_epoch` remain meaningful, and `hit_cap` is True only if `max_epochs` is reached
with the tracker still not converged. Second clarification in the same fix: define
**`E_stop` := the primary run's actual final epoch**, not "the accuracy-gated stop epoch". Under the
§4.2 AND rule the loop stops at `max(stop_ce, stop_acc)`, and `stop_acc` can be the smaller of the
two — in which case `4·E_stop` could be shorter than the primary run itself.

### Ruling on "is F5c-b a reproduction?" (brief item 7) — the author is RIGHT, and the gate still works

**Confirmed, from three directions.** (i) `ff_depth_toy_spec.md:175-183` already records the verified
finding: "no plain-`RecurrentComposer`-A5-L10 JSON exists in `D_TOY_PROBES/`, only `depth_selection_*`
files" — and Fix 1 (`:422-429`) upgraded Cell 4 to a mandatory confirm run for precisely that reason.
(ii) Directory check: the only A5/L=10 recurrent artifacts are
`depth_comp_pilot_a5_recurrent_n10_seed{0,1}.json` (2026-07-18 12:16) and
`ff_depth_pilot_a5_seed{0,1}.json` (13:06); they carry the *same* numbers (val `0.7581`/`0.9257`,
stop 3500/3750, best CE epoch 1000/1250), i.e. both are loop-T / `lr = 1e-2` runs of the same failed
protocol. Neither is certified. (iii) The certified A5/L=10 result
(`depth_selection_gradedness_seed0.json`, `lr 0.003`, per-stratum `0.9487/0.9573/0.946`) comes from
the multi-exit prefix-supervised loop, and the certified L=10 shared-readout result
(`depth_graded_pilot_s5_*`) is on S5 with summed multi-length CE. So Decision 14's "reproduce its
certified result" has, literally, no certified result to reproduce.

**Does the gate still work? Yes in the pass direction, no in the fail direction — and that is
sufficient for a gate.** A pass is a *sufficient* protocol validation: if the plain single-exit
protocol drives the certified architecture to ≥ 0.90 on both seeds, the protocol is demonstrably
trainable on this substrate and the grid may proceed. A failure, however, cannot separate "protocol
still broken" from "single-exit supervision cannot reach 0.90 on A5/L=10".

**Bar: unchanged — held-out ≥ 0.90 and trustworthy on both seeds** (`flexnn-core.md` §F5c-b; =
`ff_depth_toy_spec.md` §6 FIT + the 2-seed rule). No new bar, and none is needed. What *is* needed is
the discriminating experiment for the fail branch:

**M6 (required, conditional). Pre-register the discriminator now.** If and only if the positive
control is still failing at ladder exhaustion (L3), re-run `depth_selection_toy.py`'s certified
anytime configuration at its own settings and check it still reproduces its ≥ 0.90 per-stratum
numbers. Reproduced ⇒ data pipeline and substrate are intact and the failure is attributable to the
single-exit supervision (a substantive substrate finding). Not reproduced ⇒ the defect is
environmental or a regression, and nothing about supervision may be claimed. This costs nothing
unless the branch is taken, and registering it now stops it being improvised later.

**M7 (required). §2.5 has no terminal branch for the positive control.** Step 5 defines "ladder
exhausted" only for grid arms. `flexnn-core.md` §F5c-b says "FAIL ⇒ HALT". State it in the spec:
a positive control still failing after L3 **HALTS F5c-c and F5c-d**, runs M6's discriminator, and is
escalated to the user/adjudicator — it may not be silently converted into a finding or worked around,
because the consequence (Cell 4 of the 2×2 grid unreadable under the shared protocol) is a PI-level
call about whether the grid is still worth running. *Contingent only — this does not park the spec.*

### Additional required fix, outside the brief's seven items

**M4 (required). §2.3 rung L1 has no selection rule and an inconsistent scope.** L1 is a three-point
sweep `lr ∈ {3e-3, 1e-3, 3e-4}` over "all arms, both seeds", but §2.5 step 3 says the grid runs "at
that same rung" and step 6 caps escalation at **one** full grid re-run per rung — a 3× sweep across
every arm satisfies neither, and no criterion says which LR becomes *the* protocol. Choosing it after
seeing grid results is the Decision-9 hazard in a new place. Fix, mechanically: **sweep the positive
control only**; adopt the **largest** `lr` at which it is `trustworthy_acc` and ≥ 0.90 on **both**
seeds; run the grid once at the adopted `lr`. Selection then depends only on the certified arm, so no
grid outcome can influence the protocol. This also protects the Mode-A arms — an LR reduction that
rescues `d=10` could push `d=4` (train 0.489) further into `UNDER_FIT`, and "largest passing" is the
rule that resists that.

**Implementation constraint attached to M4/M1: do not mutate `train_clf` in place.**
`depth_selection_toy.py:98` imports it as `_train_clf_generic` and calls it at `:570` for the S3
surface probe, whose landed artifacts are `depth_selection_surface_seed{0,1}.json`. Editing the LR,
the clip or the gate inside `train_clf` silently re-protocols another strand's trainer. Parameterize
(`lr`, `clip_max_norm`, gate mode) with the current behaviour as the default, or audit that call site
explicitly and record the decision in the task.

### Non-blocking recommendations

**R1.** Apply §3's logging to the **F5c-b** positive-control runs, not only to F5c-c. Rung 0 bundles
the LR correction and clipping by design, so `clip_engagement_rate` is the only cheap way to know
which of the two was load-bearing — and it is worth having at the moment the gate passes, not three
stages later. Two runs' worth of logging.

**R2.** §4.5 pre-registers eight or more confirmation runs at ≥ 4× budget (16k–42k epochs each) with
no wall-clock estimate. State one, plus the concurrency plan, against the MASTER environment rule
(`AUTOML_DEVICE=cpu`, ≤ 4 concurrent heavy, detached).

**R3.** The reported `val_acc` is the tracker's best, which can lag the trajectory maximum by up to
`ACC_MIN_DELTA` (1e-3). Immaterial except for a cell landing in `[0.899, 0.901]`; if one does, read
the trajectory maximum and say so explicitly.

**R4. §6 item 6 is already resolved — no action.** `docs/plans/capacity_programme/gates_baseline.txt`
no longer lists this spec's path (the entry was removed in the working tree), and
`~/dev/.venv/bin/python -m pytest docs/plans/capacity_programme/test_plan_gates.py -q` returns
**6 passed**. The item can be struck.

### Coverage note (what this review can and cannot certify)

Adjudication removes false positives; it cannot recover what the spec never considered. This review
actively hunted for omissions and found four (M2, M4, M5's flag-semantics half, M7) plus the
cross-caller hazard on `train_clf` — none of which were errors in what the spec says, all of which
were gaps in what it covers. That hit rate is a caution: the §3 instrumentation plan in particular is
a *design* whose adequacy can only be judged once one instrumented run exists. Re-read §3.5's
permitted-conclusion table against the first landed CSV before any mechanism claim is written.

**Disposition: GO for F5c-b once M1–M7 are folded. Bars unchanged. Nothing parked for the user.**
