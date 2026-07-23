# WSEL-23 candidate 1 — the derived observation-model loss weighting: the written spec

**Scope of this document.** Authoring only, per the standing delegation this programme uses for
every candidate in an evidence-gated ladder (precedent: `shared/wsel21-escalation.md` rung (i)).
No code, no runs, no commits. This is candidate 1 of `docs/plans/capacity_programme/width.md`'s
`### WSEL-23` block (lines 1459-1546) — the PRIMARY item on the improvement ladder MASTER Decision
34 (`MASTER.md:615-631`) requires exhausted before WSEL-8's both-halves-FAIL verdict
(`width.md:1914-1995`) may be reported final. It covers, in the order the ratified text requires
they run: **Part 1**, the zero/near-zero-training gradient-attribution diagnostic
(`width.md:1488-1507`, MASTER Decision 35(iii) `MASTER.md:643-648`); **Part 2**, the derived
per-width weighting itself (`width.md:1468-1487`); **Part 3**, the training arms, primary and
conditional companion (`width.md:1508-1517`). Every binding clause from those blocks, §3.7
(`width.md:407-484`), §3.10 (`width.md:659-677`), and §3.8's canonical toy suite (`width.md:486-601`)
is implemented below, never re-derived differently.

**Every architecture claim in this document was verified at the constructor line before being
used**, per this programme's own case law (`project_width_kdropout_arch_error` — a wrong-class
headline once stood 3 days): `automl_package/models/flexnn/width/architectures.py:41-143`
(`NestedWidthNet`, the single-head arm), `:214-356` (`SharedTrunkPerWidthHeadNet`, the multi-head
arm, `G-WIDTH`-certified — `docs/plans/capacity_programme/width-cert.md:318-319`). The canonical
training driver is `automl_package/examples/kdropout_converged_width_experiment.py`
(`Arch`/`LossType` enums `:100-113`; the ALL-schedule unweighted sum this candidate targets at
`:293-303`); its dedicated-net anchor is `automl_package/examples/width_wsel8.py`'s
`--arm w_sweep`, already landed at
`automl_package/examples/capacity_ladder_results/WSEL8/hetero_{seed}_{width}_w_sweep.json`
(seeds 0/1/2, widths 1-12, 36 files, verified present on disk).

---

## 1. Grounding — what is fixed before any design choice below

- **§3.7 (`width.md:407-484`), binding throughout:** every variance in this document is FIXED at a
  declared value, never learned. `σ² = HETERO_NOISE_SIGMA² = 0.0025`
  <!-- numcheck-ignore: 0.0025 = 0.05^2, HETERO_NOISE_SIGMA (`automl_package/examples/nested_width_net.py:93`) squared; a derived constant, not a ledger leaf --> on the tier-1 `hetero`
  toy (`automl_package/examples/nested_width_net.py:93,144`). `a²(w)`, candidate 1's approximation-
  deficit law, is likewise DECLARED and FROZEN once pinned (§3 below) — never fit per run, never a
  per-width free constant (§3.10, `width.md:663-667`).
- **§3.8 (`width.md:486-601`):** the canonical cell is tier 1 (`--toy hetero --n-train 1500
  --n-test 500 --sigma 0.05`), seeds 0/1/2, `w_max=12`. Decision 31(a) (`MASTER.md:562-564`) makes
  the ALL schedule (every width, every step) the default training schedule for any new width run;
  `--fused-heads` stays default OFF (Decision 31 does not flip it — WSEL-18's own non-goal,
  `width.md:3114`, re-verified against the driver's live `--fused-heads` help text,
  `kdropout_converged_width_experiment.py:783-789`). **Footgun on record:** the driver's own
  `--schedule` CLI default is still
  `sandwich` (`kdropout_converged_width_experiment.py:770`) — Decision 31 is a research ruling, not
  a re-defaulted flag (the same shape as the WD6 loss-default leak, §3.7). Every command in this
  document passes `--schedule all` explicitly; an un-flagged invocation trains the wrong thing.
- **No cached MID-TRAINING state of the canonical joint nets exists to source the diagnostic's
  second checkpoint from — ⚖️ factual claim CORRECTED at the root's adversarial read (2026-07-23;
  the draft asserted "no `.pt` weight file" under these dirs, which is FALSE as written):**
  `WSEL8/_cache/` holds 36 `.pt` state dicts (`sweep_tier1_seed{0-2}_w{1-12}.pt` — the DEDICATED
  single-width sweep nets, plus 2 capped originals under `capped_at_6000/`), and `WSEL16/` holds
  75 `.pt` state dicts (`state_a_{cascade,corrective,staged,...}_tier1_seed*.pt` — the residual
  recipe study's nets). **The sourcing CONCLUSION survives the correction:** every one of those is
  a FINAL state of a different model kind or recipe lineage — none is a mid-training state of the
  canonical UNWEIGHTED joint nets (`SharedTrunkPerWidthHeadNet`/`NestedWidthNet` under
  `--schedule all --loss mse`) this diagnostic must probe. **Consequence, stated plainly in §2.3:
  the diagnostic's init half is genuinely zero-training; its mid-training half is not, and its
  cost is charged, not hidden. Root requirement added: the 6 partial runs' mid-training snapshots
  are SAVED to disk (git-ignored, beside the run's cells) so the diagnostic is re-runnable without
  retraining.**
- **G-WIDTH's two clauses**, needed verbatim for the recipe-survival gate (§5.3), from
  `width-cert.md:308-311,318-326`: **(a) noisy-easy** — on `hetero3` (tier 2), the dial's picked
  width for the noisy-easy region must stay small (`noisy_easy_pass=true`) on ≥2 trustworthy seeds;
  **(b) dial-sep + fit** — the dial-network's quality must separate across width in ≥3 of 4 WP-4
  tier-3 corners, and the widest head must reach the noise floor (`fit_bar` pass) at the
  discriminating σ=0.05 corner on ≥2 trustworthy seeds.

---

## 2. Part 1 — the gradient-attribution diagnostic (runs FIRST, before any weighted training)

### 2.1 What counts as a "shared parameter", per architecture (verified at the constructor)

| Architecture | Class | Shared params measured | Excluded |
|---|---|---|---|
| Multi-head (certified) | `SharedTrunkPerWidthHeadNet` (`architectures.py:214-356`) | `trunk.weight`, `trunk.bias` (`:282`) | `mean_heads[k].{weight,bias}` — per-width OWN heads, not shared (`:297`) |
| Single-head | `NestedWidthNet` (`architectures.py:41-143`) | `trunk.weight`, `trunk.bias` (`:65`), `mean_head.weight`, `mean_head.bias` (`:67`) | `logvar_head` — unused dummy under MSE, never in the optimiser's parameter list (§3.7 architecture note, `width.md:481-484`) |

This matches the ratified text exactly ("trunk for the multi-head; trunk + output weights for the
single-head", `width.md:1493-1494`) and reuses this programme's existing filter-by-name-substring
convention (`param_count`'s `path_filter` argument, `automl_package/utils/capacity_accounting.
py:124`, and `LOGVAR_HEAD_PATH_SUBSTRING` already imported for exactly this exclusion at
`kdropout_converged_width_experiment.py:65`) rather than inventing a new accounting mechanism.

**Structural fact, verified by reading the masking code, not assumed:** `forward_width(x, k)`
zeroes `h[:, k:]` identically before either readout (`architectures.py:331-333` unfused;
`:74-87` `NestedWidthNet`), so width-`k`'s loss term has **exactly zero gradient** on `trunk`
rows `>= k` — width 1's term only ever touches trunk row 0; width 12's touches all twelve. The
"outsized share" the pre-registered expectation names is therefore about gradient MAGNITUDE on the
rows a narrow width DOES touch (chiefly the earliest, most-shared rows), not about touching more
rows — stated here so the measurement below is read correctly and not miscalibrated against a
"share should equal `k/w_max`" strawman.

### 2.2 Measurement protocol

For each width `k = 1..w_max`, using the CURRENT unweighted per-width loss exactly as trained
(`_width_loss(LossType.MSE, net, k, x, y)`, `kdropout_converged_width_experiment.py:142,145-146`,
which is `((mean - y)**2).mean()` off `forward_width`), compute

```
g_k = ∇_{θ_shared} L_k(θ)          (flatten + concatenate the shared-parameter grads above)
share_k = ||g_k||_2 / Σ_{j=1}^{w_max} ||g_j||_2
```

via `w_max` independent forward+backward passes (recompute `forward_width` per `k`, `zero_grad()`
between each — deliberately NOT `sampled_widths_forward`'s shared-trunk-reuse trick, since that
computes one autograd graph across widths and this measurement wants each width's contribution
ISOLATED, not summed). `share_k` is a plain L2-norm share (not squared/"energy") — literal reading
of "share of gradient magnitude" (`width.md:1493`); flagged in §7 as a considered, not incidental,
choice. Reported per width, per architecture, per seed, at both checkpoints below.

**Checkable numeric reference point.** If gradient share were proportionate to width count, widths
1-3 would carry `3/12 = 0.25` of the total. "Outsized" is defined against this baseline.

### 2.3 Checkpoint sourcing — the decision this brief asked for, made explicit

- **Init**: genuinely zero training. Construct `net` under a fixed seed (0/1/2), measure `share_k`
  before any optimiser step. Free.
- **Mid-training**: **no existing cached state serves this** (§1). One bounded, cheap PARTIAL
  training run per (architecture, seed) is required, and its cost is charged in §6, not concealed
  under the ratified block's "zero training" phrasing (flagged for the adversarial read as
  Open Question 1, §7). The run trains the canonical cell (tier 1, `--schedule all`, `--loss mse`,
  the CURRENT unweighted objective — this measurement is about the problem, not the candidate fix)
  and stops the FIRST epoch at which any one width's `ConvergenceTracker.converged` flips `True`
  (`automl_package/utils/convergence.py:107,121-123`) while at least one other width's has not —
  i.e. the first genuinely "mid" state, some widths done, others still moving, detected with the
  EXISTING per-width tracker machinery already in `_train_kdropout_to_convergence`
  (`kdropout_converged_width_experiment.py:279`), not a new training loop. State snapshot
  (`net.state_dict()`) taken at that epoch for the gradient measurement; the run may then be
  discarded (it is not the candidate's own reference number — Part 3 trains its own arms).
  **Cost precedent, cited so this is not a guess:** under the SANDWICH schedule (not this run's ALL,
  but the same canonical cell otherwise), width 1 — historically the fastest to converge — stopped
  at epoch 6500/6500/7000 across seeds 0/1/2 while the whole run continued to 24000/16500/27500
  <!-- source: `automl_package/examples/capacity_ladder_results/W_KDROPOUT_CONVERGED/w_kdropout_converged_summary_shared_trunk_mse.json` `per_case[].convergence."1".stop_epoch` and the per-seed max over `convergence.*.stop_epoch` -->,
  i.e. the "first width converged" checkpoint historically lands at roughly a quarter to a third of
  the full run's eventual length. ALL trains width 1 every step (same as SANDWICH does for widths
  `{1, w_max}`), so a similar or shorter fraction is expected; the driver task confirms it empirically
  rather than assuming the SANDWICH number transfers exactly.

### 2.4 Seeds, pre-registered expectation, branch consequences

3 seeds (0, 1, 2 — §3.8's fixed set). **Confirmed** iff, at BOTH checkpoints independently, `Σ_{k=1}^{3} share_k`
exceeds the `0.25` uniform baseline by more than 2×SE of that sum across the 3 seeds — the
noise-aware rule WSEL-20 adopted and made strand-wide binding (`width.md:1242-1244`: "any future
plateau/invariance decision uses the noise-aware rule; no new flat-percentage bar without a stated
noise argument"), reused here for a different decision rather than re-litigated. **If only one of
the two checkpoints shows the pattern beyond noise, that is a MIXED finding, recorded as such, and
NOT treated as a clean confirmation — ⚖️ root ruling (2026-07-23): a MIXED finding does NOT
authorize the companion arm (§4.2 is conditional on CONFIRMING, and mixed ≠ confirmed); the
primary proceeds either way per the Open Question 3 resolution below.**

- **Confirmed** → weighted training (Part 3) proceeds with direct evidence that the current
  unweighted sum lets narrow widths' large irreducible error dominate the shared units' gradient,
  per the mechanism paragraph already on record (`width.md:1500-1507`); the single-head companion
  (§4.2) is authorized.
- **Refuted** (neither checkpoint shows the pattern beyond noise) → recorded against the candidate;
  the companion arm is DROPPED (ratified, `width.md:1517`, MASTER Decision 35(iii)). **The primary
  weighted-training arm is NOT explicitly gated by this outcome in the ratified text** — only the
  companion carries an explicit "if (i) refutes, drop" clause. This document's reading: the primary
  proceeds regardless, because the MLE derivation (§3.1) is valid independent of the gradient-
  attribution story; a refutation is recorded as a weakened MOTIVATING NARRATIVE beside whatever
  the primary run shows, never suppressed. Flagged as Open Question 3 for the root's confirmation.

---

## 3. Part 2 — the derived per-width weighting

### 3.1 Observation model and its MLE — full derivation

Posit, for each width `k` and each training point `(x_i, y_i)`, a per-channel Gaussian observation
model (the standard heteroscedastic multi-task-uncertainty construction — each output "channel" is
allowed its own noise scale, here indexed by width rather than by task):

```
y_i | x_i, k  ~  N( f_k(x_i; θ),  σ² + a²(k) )
```

`σ²` is the toy's TRUE, declared noise variance (§3.7 — irreducible, identical across every width,
since it is a property of the DATA). `a²(k)` is the DECLARED approximation-deficit law: the extra
effective variance width `k`'s function class carries even at its best achievable fit, because a
narrow head cannot represent the toy's width-hungry component at all well. Modelling an irreducible
BIAS floor as an ADDITIONAL variance term is the same move `width_candidates.weighted_squared_error`
already makes for per-point noise (`width_candidates.py:323-345`) — this is that same move, indexed
by width instead of by point.

Assuming conditional independence across points and across width-channels given `θ` (the standard
assumption already implicit in `_sampled_widths_total_loss`'s per-width sum,
`kdropout_converged_width_experiment.py:150-175`), the joint negative log-likelihood over one batch is

```
NLL(θ) = Σ_i Σ_k [ (y_i − f_k(x_i;θ))² / (2(σ² + a²(k)))  +  ½ log(2π(σ² + a²(k))) ]
```

**The second term is a CONSTANT in θ**, because `σ²` and `a²(k)` are FIXED/declared, never learned
(§3.7's binding rule — this is exactly why the rule is safe to lean on here: if either were being
fit, this term would carry a gradient and the derivation below would not hold). Dropping it,
minimizing `NLL(θ)` is therefore EXACTLY equivalent to minimizing

```
L(θ) = Σ_k  w_k · MSE_k(θ),      w_k := 1 / (σ² + a²(k)),      MSE_k(θ) = (1/N) Σ_i (f_k(x_i;θ) − y_i)²
```

**`w_k` falls out of the model as an MLE consequence — it is not an appended penalty.** This is the
programme's own "no arbitrary penalties" test (`width.md:1473`), passed by construction: nothing is
added to the objective that the observation model does not already imply.

**Rescaling lemma (used in §3.4).** Multiplying every `w_k` by the SAME positive constant leaves
`argmin_θ L(θ)` unchanged (it globally rescales the objective, not its shape), so any positive
rescaling of the weight table is a free choice with no effect on the population optimum — though it
DOES change Adam's per-step effective update magnitude (gradient scale interacts with Adam's own
adaptive normalisation), which is exactly why §3.4 chooses a specific, stated rescaling rather than
leaving the raw `1/(σ²+a²(k))` values unmodified in the loss — those raw values span roughly an
order of magnitude across widths (preview: ~11x from `w=1` to `w=12`, §3.3's preview `A`) rather
than the mild, widest-head-anchored range §3.4's normalization produces.

### 3.2 Proposed parametric form for `a²(w)`

**Declared form:** `a²(w) = A · w^(−p)`, with the EXPONENT `p = 2` FIXED by an approximation-
theoretic anchor, not freely fit, and exactly ONE free scale parameter `A` — satisfying the
generalization clause's "parametric law... instantiated once and FROZEN — never per-width
constants" (`width.md:1477-1478`) as tightly as a one-parameter family can.

**Why `p = 2`.** The toy's target is smooth (a bounded-second-derivative easy-line-plus-sine,
`nested_width_net.py:187-188`) — classical interpolation theory gives `O(N^{-2})` sup-norm error
for optimal `N`-node approximation of a twice-continuously-differentiable function (the textbook
node-spacing argument: linear-interpolation error on an interval scales as `h²·|f''|/8`, so `N`
nodes give `O(N^{-2})`); squaring an `O(N^{-1})` amplitude rate for a smoother function class gives
the same `O(N^{-2})` shape for the SQUARED error this candidate's `a²(w)` denotes. This is offered
as a classical-approximation-theory justification for the FUNCTIONAL FORM, not a claim about a
specific named theorem for THIS toy — the fit below is the empirical check that the form is not
merely plausible but actually close.

**Empirical support — ⚖️ ROOT-CORRECTED (adversarial read, 2026-07-23; the draft's preview was
MISLABELED and the honest picture is domain-sensitive).** The draft claimed a free log-log fit
"over widths 8-12" gives `p ≈ 1.9885` <!-- numcheck-ignore: the draft's mislabeled preview exponent, quoted for the correction record; root-replicated as the w4-12 fit, see the ignore note below -->; the root's replication from the same 36 cells shows that
number actually comes from fitting widths **4-12**, while the draft's stated 8-12 domain gives a
NEGATIVE exponent (`p̂ ≈ −0.56`) — because widths 8-12 sit at the convergence FLOOR (raw deficit
flat at ≈0.0002-0.0004, no decay; see §3.3's corrected domain rationale), and the genuine decay
regime 4-7 alone gives `p̂ ≈ 3.80`
<!-- numcheck-ignore: root-replicated free-fit exponents (−0.5601 over w8-12, 1.9885 over w4-12, 3.7966 over w4-7), log-log OLS on the 36 cited WSEL8 cells; domain-sensitivity illustration, not a ledger leaf; the pinning script recomputes -->.
**The honest status of `p = 2` is therefore: a DECLARED theory anchor whose adequacy is validated
by the pre-registered held-out check in §3.5 — NOT a data-confirmed exponent.** The free fit is
too domain-sensitive on 12 points spanning two regimes and a floor to confirm any exponent; the
factor-of-3 held-out validation (which the corrected-domain law passes in preview, §3.5) is the
load-bearing empirical check, and a held-out failure there falsifies the declared form.

### 3.3 Pinning procedure — from the existing dedicated-net curves, zero new compute

**Source (already on disk, verified present):**
`automl_package/examples/capacity_ladder_results/WSEL8/hetero_{seed}_{width}_w_sweep.json`,
`seed ∈ {0,1,2}`, `width ∈ {1..12}` — 36 files, each a `held_out_mse` from an independently-trained
DEDICATED net at that exact width (`--arm w_sweep`, WSEL-4's ported protocol, `width_wsel8.py`'s own
module docstring). This is the "existing dedicated-net curves" the circularity guard names as
legitimate for pinning (`width.md:1539-1540`).

**Procedure:**
1. For each width `w`, `mean_mse(w) := mean over seeds {0,1,2} of held_out_mse(w)`.
2. `raw_a2(w) := max(mean_mse(w) − σ², 0)` (floor at 0 — a seed-noise safety clamp; not exercised in
   the 12 widths on record, all of which land strictly positive
   <!-- numcheck-ignore: per-width mean-mse-minus-sigma2 preview values (e.g. 0.067083 at w=1 down to 0.000305 at w=12) computed by this document's author directly from the 36 cited WSEL8 cells; illustrative only, the pinning script recomputes -->).
3. **Fit domain: `w ∈ {4, 5, 6, 7}` — ⚖️ ROOT-AMENDED (adversarial read, 2026-07-23; the draft's
   `{8..12}` domain is REVERSED into the held-out role).** The root's replication showed widths
   8-12 sit at the convergence FLOOR (raw deficit flat at ≈0.000234-0.000383 with NO decay — the
   free-fit exponent there is negative, §3.2): a floor region cannot identify a decay constant,
   and pinning `A` there anchors the law to floor residue amplified by `w²`, not to approximation
   decay. Widths 4-7 are the verified smooth-decay regime (the knee sits at w=4, §3.3's own
   regime argument), so the fit and held-out roles are SWAPPED from the draft: fit on the decay
   regime, hold out the floor. Fit `A` by ordinary least squares on the DECLARED form with `p`
   fixed at 2: `A* = ( Σ_w raw_a2(w)/w² ) / ( Σ_w 1/w⁴ )`, sum over `w ∈ {4..7}`.
   Corrected preview: `A* ≈ 0.040647`
   <!-- numcheck-ignore: root-replicated preview via the stated closed-form OLS over the corrected w4-7 domain on the cited WSEL8 cells; not a ledger leaf, the pinning script recomputes and freezes the authoritative value -->.
   (The draft's `A* ≈ 0.027076` under the floor-domain fit is retained here only as the record of
   what the amendment changed; it is not the law. Weight-table consequence of the correction is
   immaterial to the mechanism: the w=1 discount moves ≈91% → ≈93.5%, same shape, same
   monotonicity.)
4. `σ²` itself is READ from `HETERO_NOISE_SIGMA` (`nested_width_net.py:93`), never re-estimated.

**Why this fit domain, not "all 12 widths".** Widths 1-3 sit in a QUALITATIVELY different regime —
the toy's sine component is not resolvable there AT ALL (a representational floor, not a smooth
decay: `mean_mse` drops from ≈0.05-0.07 at `w≤3` to ≈0.003-0.005 at `w=4` in one step
<!-- numcheck-ignore: illustrative preview of the same 36-cell table already cited above; the knee is visible without decimal precision mattering -->,
consistent with WSEL-8's own recorded mechanism, `width.md:1927-1930`: "the dial network's shared
trunk cannot serve the middle heads... premium peaks at width 4"). Including them in the SAME
least-squares fit as the smooth-decay widths would let three outlier-shaped points dominate the fit
of a curve meant to describe a different mechanism. Excluding them is also SAFE for the weighting's
purpose (§3.4): widths 1-3 already sit at dial/sweep PARITY in WSEL-8 — width 1 ratio 1.081, width 2 ratio 1.159, width 3 ratio 0.997, all `mean_ratio_shared_over_sweep` in `automl_package/examples/capacity_ladder_results/WSEL8/frozen.json` `quality_at_matched_width` (quoted at `width.md:1925`) — nothing there needs correcting, so a law that (as shown in §3.5) UNDER-predicts
their true deficit merely leaves them slightly LESS discounted than an ideal law would, which is the
conservative, non-disruptive direction.

### 3.4 Freeze rule and normalization

`(A, p=2, σ²)` are computed ONCE by the procedure above and FROZEN into a small config artifact
(bare filename `a2_law.json`, written under the candidate's own results directory, §6) that every
training arm in §4 reads — never refit per run, per architecture, or per seed (this is the literal
"instantiated once and FROZEN" clause, `width.md:1477-1478`).

**Normalization, a deliberate choice (Open Question 5, §7):** rescale so the WIDEST head keeps its
old (unweighted) relative scale —
`weight(w) := (σ² + a²(w_max)) / (σ² + a²(w))`, so `weight(w_max) = 1` and every narrower width's
weight is `< 1`, a pure DISCOUNT relative to today's unweighted sum rather than an inflation of the
wide heads. By the rescaling lemma (§3.1) this changes nothing about the MLE optimum, only the
per-step gradient scale the existing Adam/convergence-gate machinery (tuned at the unweighted scale)
sees — keeping `min_delta=0.002` and friends meaningful without re-tuning them.

Preview table (illustrative, from the frozen `A*≈0.027076` above; the driver recomputes and freezes
the authoritative version)
<!-- numcheck-ignore: full preview weight table (w=1..12, weight ranging 0.0909 to 1.0000) computed by this document's author from the frozen preview A above; not a ledger leaf -->:
narrow widths are discounted smoothly and substantially (roughly a 91% discount at w=1, down to no
discount at w=12), monotonically — exactly the shape the mechanism paragraph predicts is needed.

### 3.5 Generalization / `≥2 w_max` validation — no new sweep

**Held-out-widths validation (zero new compute; ⚖️ ROOT-AMENDED domains, and DEMOTED from
satisfying the `≥2 w_max` clause — see below):** the frozen law, fit ONLY on the decay regime
`w ∈ {4..7}` (§3.3 as amended), is evaluated at the held-out floor widths `w ∈ {8..12}`, using
cells ALREADY on disk.

**Pre-registered profile response, three parts:**
1. **Shape sanity** — the predicted curve is monotone-decreasing in `w` by construction (a trivial
   property of any `A·w^-p` form with `A,p>0`); reported as a pass/fail check anyway, since a bug in
   the pinning code could still violate it.
2. **Held-out match, `w ∈ {8..12}` (never used in the fit):** predicted `a²(w)` within a factor
   of 3 of the observed value (a DECLARED tolerance, not the 2×SE rule — reasoned explicitly:
   the only SEs available come from 3-seed means, which are far tighter than the genuine systematic
   gap this extrapolation can carry, so a 2×SE bar would be uninformatively strict; factor-of-3 is
   the stated, checkable bar instead) for **at least 4 of the 5** held-out widths — mirroring
   G-WIDTH's own "≥3/4 corners" bar shape (§1). Root-replicated preview at the corrected domains,
   three spot-checked widths: obs/pred ≈ 0.37 / 0.73 / 1.08 at w = 8/10/12 — within factor 3 at
   every spot-checked width; the driver recomputes all five
   <!-- numcheck-ignore: root-replicated preview ratios from the cited WSEL8 cells at the corrected fit domain (pred 0.000635/0.000406/0.000282 vs obs 0.000234/0.000298/0.000305 at w=8/10/12); illustrative, the driver recomputes all five -->.
   A held-out failure here FALSIFIES the declared form (it is the load-bearing empirical check,
   §3.2 as corrected).
3. **Out-of-regime miss, `w ∈ {1,2,3}`:** EXPECTED and REGISTERED AS ACCEPTABLE, not a falsifier —
   a qualitatively different (representational-floor, not smooth-decay) regime the declared law was
   never meant to cover, and one where under-prediction is the safe direction (§3.3). Reported
   honestly (an order-of-magnitude miss is expected here, not concealed).

**`≥2 w_max` validation — ⚖️ ROOT RULING (adversarial read, 2026-07-23; resolves Open Question
2 AGAINST the draft's recommendation): the width-truncation reading does NOT satisfy the clause's
letter.** "Validated at ≥2 values of `w_max` (the `w_max=12` toy plus at least one other)"
(`width.md:1477-1480`) names two LADDER SCALES, and slicing one `w_max=12` sweep's widths is one
ladder read twice — the held-out-widths check above validates the functional form's extrapolation,
not the law's transfer across problem scale. **The draft's secondary path is PROMOTED to
REQUIRED, extended with its missing control arm:** train the PRIMARY weighted recipe (§4.1) at
`--w-max 6` (`hetero`, tier 1, seeds 0/1/2) with the frozen `(A, p=2, σ²)` sliced to `w=1..6` —
NO refitting — **plus 3 UNWEIGHTED `--w-max 6` control cells (same cell, `--loss mse`), without
which no premium-shrink comparison exists at the second scale** (the dedicated-net comparators
are already on disk and are `w_max`-independent — the SAME WSEL8 cells at `w ∈ {1..6}` serve).
**Pre-registered profile response at `w_max'=6`:** the weighted arm's mid-width matched-width
ratios move in the SAME direction vs its unweighted control as at `w_max=12` (the premium
shrinks), under the per-width noise-aware readout of §5.1 — the law transfers across ladder
scale, or the candidate's generalization clause fails and that failure is recorded. This is
"a declared cheap probe" in the circularity guard's own words (`width.md:1541-1542`): 6 small
training cells, no new sweep, the law pinned before any of them runs.

### 3.6 Circularity guard — compliance statement

Per `width.md:1539-1542`: the pin (§3.3) uses ONLY already-landed WSEL8 cells (legitimate for the
mechanism test); the `≥2-w_max` validation (§3.5, primary) uses the SAME already-landed cells with
zero new training — strictly cheaper than "a declared cheap probe", so the law does not inherit the
sweep's cost and the replacement economics stay non-circular.

---

## 4. Part 3 — training arms

### 4.1 Multi-head weighted (PRIMARY)

`SharedTrunkPerWidthHeadNet`, canonical cell (`--arch shared_trunk --toy hetero --schedule all`,
tier 1: `n_train=1500 n_test=500 sigma=0.05`), **3 seeds (0,1,2)** — justified directly from §3.8's
fixed seed set and the WSEL-8/WSEL-16 canonical-cell precedent (`width.md:2295`: "root re-runs one
canonical cell... seeds 0/1/2"), never re-derived per task. New `LossType.WEIGHTED` member
(extending, not replacing, the existing `NLL`/`MSE` closed set at
`kdropout_converged_width_experiment.py:109-113`), reading the frozen per-width weight table from
`a2_law.json` (§3.4) and multiplying each width's MSE term by its weight inside
`_sampled_widths_total_loss` (`:150-175`) before summing — the exact pattern
`weighted_squared_error` already uses for per-point weights (`width_candidates.py:323-345`), here
indexed by width instead of by point; an optional `weight: float = 1.0` parameter threaded through
`_loss_from_readout`/`_width_loss` keeps every existing NLL/MSE call site byte-identical (default
weight 1.0 is a no-op).

MASTER Decision 9 (trajectory discipline, `MASTER.md:174-177`) and Decision 16 (optimization
exonerated before architecture is blamed, `MASTER.md:208-213`) both bind: full held-out trajectory,
`hit_cap=False` required, and an under-converged run goes through the LR/clip/warmup/init/normalize
escalation ladder before being recorded as an architecture (candidate) finding.

### 4.2 Single-head conditional companion

**Runs ONLY if §2.4's diagnostic confirms.** `NestedWidthNet`, same canonical cell, same frozen
weights, `Arch.NESTED`, 3 seeds — a mechanism DISCRIMINATOR (`width.md:1508-1517`), not a
resurrection attempt: because `NestedWidthNet` shares its readout too (trunk AND the one `mean_head`,
vs the multi-head's trunk-only sharing, §2.1), the SAME weighting should help its mid-widths MORE in
relative terms if the gradient-attribution mechanism is real. **Expectation on record, verbatim**:
it stays uncompetitive at full width; its existing retirement (ordering-success / accuracy-failure)
stands unless it closes the primary bar (§5.1), in which case the existing promotion path applies.
If §2.4 refutes, this arm is DROPPED and the drop recorded — no training, no cells.

---

## 5. Pre-registered bars

### 5.1 Matched-width ratio bar (noise-aware, WSEL-20-style)

**Affected band:** `w ∈ {4..11}` — the widths where WSEL-8 recorded the largest premium, from width 11's 1.571 up to width 4's 7.207 (`mean_ratio_shared_over_sweep`, `automl_package/examples/capacity_ladder_results/WSEL8/frozen.json` `quality_at_matched_width`, quoted at `width.md:1925`); widths 1-3 (parity already, §3.3) and width 12 (smallest premium, 1.486, same file) are excluded from the CLOSURE bar but still reported.

**Closure rule — ⚖️ ROOT-AMENDED (adversarial read, 2026-07-23; the draft's pooled-24-cell
aggregate is REJECTED):** pooling all `(width, seed)` cells mixes the SYSTEMATIC width-to-width
premium variation (1.5× to 7.2× across the band) into the "noise" SE — inflating the bar into
near-uncheckability — and lets one badly-open width hide under seven closed ones (the exact
pooled-hides-split flaw the root amended out of two other specs this same day). **Closure is
PER-WIDTH: for EACH width in the affected band `{4..11}`, the 3-seed mean matched-width ratio's
gap to `1.0` must not exceed 2×SE of that mean across its own 3 seeds; the candidate CLOSES the
bar iff EVERY band width individually closes.** With 3-seed SEs this is generous per width, so
the ALL-widths conjunction is not an unreasonably strict compound bar. The pooled mean ± SE is
reported descriptively beside the per-width table, never as the gate.

If closed: WSEL-8's protocol (`width_wsel8.py`) re-runs at the winning recipe (threading
`--loss weighted` through its own `--arm w_shared` training call) and the both-halves-FAIL verdict
re-grades. If not closed: recorded as CLOSED-UNFIRED under trajectory-verified convergence (Decision
9) — the (ii) branch of WSEL-23's end-state (`width.md:1532`) — and the ladder proceeds to candidate 2.

### 5.2 Selection corollary (verbatim, `width.md:1481-1487`)

If the candidate shrinks the mid-width premium, the end-state WSEL-8 re-run's dial pick must move
NARROWER (today: 9/11/11 vs the sweep's 7/8/6, `width.md:1934-1938`) and pick-agreement with the
sweep must rise from `0/3`. Both directions are read off the SAME WSEL-8 re-run in §5.1 — no
separate compute.

### 5.3 Recipe-survival gate (G-WIDTH's two clauses re-read)

A winning recipe (§5.1 closed) must ALSO re-verify G-WIDTH's two clauses (§1) on the RETRAINED
(weighted) network: **(a)** re-run the noisy-easy check on `hetero3` (tier 2, 3 seeds) with the
weighted objective's per-width weight table sliced appropriately (tier 2 additionally carries the
per-POINT `weighted_squared_error` from §3.7 — the two weightings compose, per-point sigma AND
per-width `a²`, since they answer different questions and neither subsumes the other); **(b)**
re-run the ≥3/4 tier-3 corner check and the σ=0.05 fit-bar check. The EXACT corner definitions are
pulled from `width-cert.md`'s own W2/W4 tasks at execution time — not re-derived in this document
(non-goal, §9) — this is a pointer, not a gap. Closing §5.1 while failing either G-WIDTH clause is
recorded as "closes matched-width, breaks selection" — a genuine possible outcome, reported either
way, never suppressed.

---

## 6. Driver contract (built later, by a different task — this document specs it, never runs it)

**Per-cell CLI** (extends the existing driver, `kdropout_converged_width_experiment.py`, per §3.9's
"ONE home" discipline — no new driver file for the training arms themselves):
`--arch {shared_trunk,nested} --loss weighted --schedule all --toy hetero --w-max {12,6}
--tag wsel23c1 --results-dir automl_package/examples/capacity_ladder_results/WSEL23/`. A NEW,
SEPARATE small script (bare filename `width_wsel23_candidate1_pin.py`, living beside the other
`width_wsel23_*` drivers under `automl_package/examples/`) owns: (a) the pinning procedure (§3.3),
reading the 36 WSEL8 cells and writing the frozen `a2_law.json`; (b) the §3.5 primary (zero-cost)
validation report; (c) the diagnostic (§2), including the bounded mid-training partial run and its
early-stop-on-first-convergence instrumentation. One JSON per cell, written to disk the MOMENT it is
produced (standing clause) — never held for a final aggregate write. `--summarize` mode aggregates
already-landed per-cell JSONs into the report tables of §5 without retraining. `--selftest` mode
proves: the weighted-loss reduces to plain MSE when every weight is forced to 1.0 (a byte-identical
regression guard, the same shape as `sw.DELTA_TIE`'s regression guard,
`kdropout_converged_width_experiment.py:91-92`); the `a2_law.json` schema round-trips; the
early-stop-on-first-convergence instrumentation actually stops before `max_epochs` on a tiny smoke
config. **The ROOT runs every grid cell — no worker ever owns execution** (standing clause,
`~/.claude/model-routing-policy.md`).

---

## 7. Compute estimate

| Item | Cells | Basis |
|---|---:|---|
| Diagnostic, init (§2.3) | 6 net constructions (2 arch × 3 seeds), 0 training | free |
| Diagnostic, mid-training (§2.3) | 6 partial runs (2 arch × 3 seeds), each stopped at first-width-convergence | cheap — historically ~25-30% of a full run's epoch budget on the SANDWICH precedent (§2.3); ALL-schedule's own fraction confirmed empirically, not assumed |
| Pinning (§3.3) | 0 | reuses 36 already-landed WSEL8 cells |
| Primary `≥2-w_max` validation (§3.5) | 0 | reuses the same 36 cells, different width slice |
| `≥2-w_max` validation (§3.5, REQUIRED per root ruling) | 6 (weighted + unweighted control, 3 seeds each, `w_max=6`) | small toy, ≤ the canonical cell's own cost per run |
| Multi-head weighted, primary (§4.1) | 3 (1 arch × 3 seeds) | canonical cell |
| Single-head companion (§4.2, conditional) | 3 (1 arch × 3 seeds) | canonical cell, only if §2.4 confirms |
| WSEL-8 re-run at winning recipe (§5.1, conditional) | 36 sweep + 3 dial (WSEL-8's own count, `width.md:1916`) | only if §5.1 closes |
| Recipe-survival gate (§5.3, conditional) | ~3 (tier 2) + up to ~8-12 (tier-3 corners) | only if §5.1 closes; exact corner count pulled from `width-cert.md` at execution time |

**Total unconditional cost: 15 real training runs (6 diagnostic partials + 3 primary + 6
`w_max=6` validation cells, weighted + control) plus 36 free reads.** The companion (3) and every
WSEL-8-re-run-onward line are conditional on earlier branches, never charged twice.

---

## 8. Failure branches (checkable end-states, none invented)

- Diagnostic refuted at both checkpoints → companion dropped; primary proceeds, motivating story
  recorded as weakened (§2.4).
- Primary under-converges (`hit_cap=True` or still-improving) → Decision 16's escalation ladder
  runs before any architecture/candidate verdict is recorded (§4.1).
- Primary converges cleanly but §5.1 does not close → CLOSED-UNFIRED, ladder proceeds to candidate 2
  (deployment-prior mixture training, `width.md:1518-1521`).
- §5.1 closes but §5.3 (recipe-survival) fails → recorded as "closes matched-width, breaks
  selection", reported prominently, NOT treated as an unqualified win (§5.3).
- §5.1 and §5.3 both close → WSEL-8's both-halves-FAIL verdict re-grades at the winning recipe;
  the selection corollary (§5.2) is read off the same re-run.

Any of these outcomes is entered into WSEL-23's exhaustion ledger (`width.md:1529-1534`) — the
programme's checkable "either every candidate closes its bar, or fails it with trajectory-verified
convergence" end-state. Candidate 1 does not need to WIN for WSEL-23 to make progress; it needs to
be MEASURED.

---

## 9. Non-goals

No depth-plan edits — the transfer note (`width.md:1543-1545`, candidate 1's derivation applying
verbatim to per-depth heads) is recorded already and restated here only as this same pointer, not
expanded. No candidates 2-4 content beyond the sequencing already fixed by `width.md`'s own ladder
order (deployment-prior mixture → per-width private capacity → self-distillation, unchanged here).
No learned variances anywhere (§3.7 binding). No per-width free constants — `a²(w)` is the ONE
parametric law, `A` the ONE free scalar (§3.2). Neither this document's author nor the driver
task's author runs any training, grid, or commit — the root does (standing clause).

---

## 10. Open questions for the root's adversarial read — ✅ ALL RESOLVED (root, 2026-07-23; verdict in §11)

1. **"Zero training" vs the verified absence of any cached checkpoint (§2.3).** The ratified text
   calls the diagnostic zero-training; this document shows the init half genuinely is, but the
   mid-training half requires 6 bounded partial runs, charged in §6. Needs ratification that this is
   the correct reading, not a violation of the ratified design.
2. **Whether the zero-cost width-truncation check (§3.5) satisfies "`w_max=12` plus at least one
   other" on its own**, or whether the optional secondary (a genuinely different, real `w_max'=6`
   toy) is required. Recommendation: the zero-cost check suffices; the secondary is a cheap hedge.
3. **Whether diagnostic-refutation halts only the companion or also the primary (§2.4).** The
   ratified text is explicit only about the companion. This document assumes the primary proceeds
   regardless, since the MLE derivation does not depend on the gradient-attribution story being
   correct — flagged for confirmation.
4. **The gradient-share metric's normalization (§2.2)** — plain L2-norm share, not a squared-norm
   "energy" share. Literal reading of "share of gradient magnitude"; flagged in case the root reads
   it differently.
5. **The weight-table normalization convention (§3.4)** — rescaling so the widest head's weight is
   1.0. A considered default under the rescaling lemma, not dictated by the ratified text.
6. **The matched-width closure bar's exact aggregate (§5.1)** — mean ratio over the 4-11 band times
   3 seeds (24 cells), 2×SE gate. The ratified text names "the noise-aware rule" without specifying
   which aggregate; this is this document's proposed instantiation.
7. **The `p=2` exponent and the `w∈{8..12}` fit domain / `w∈{4..7}` held-out split (§3.2-§3.3).**
   Theory-anchored and empirically checked (§3.2, §3.5), but the specific domain split is this
   document's design choice, not dictated verbatim upstream.

**Resolutions (root, each applied in place above):** (1) RATIFIED — the partial runs are the
minimum faithful implementation given the (corrected) absence of any mid-training joint-net
state; "zero training" in the ratified block described the measurement, not checkpoint
production; snapshots must be SAVED (§1 as amended). (2) REJECTED as proposed — width truncation
is one ladder read twice; the `w_max'=6` probe is REQUIRED, extended with its missing unweighted
control arm (§3.5 as amended). (3) CONFIRMED — the primary proceeds on refutation (the MLE
derivation stands independently; exhaust-before-negative wants the candidate MEASURED); mixed →
companion not authorized (§2.4 as amended). (4) CONFIRMED — plain L2 share, pre-registered.
(5) CONFIRMED — widest-head-anchored normalization under the rescaling lemma. (6) REJECTED as
proposed — per-width closure, all band widths, pooled number descriptive only (§5.1 as amended).
(7) DOMAINS SWAPPED — fit on the decay regime {4..7}, hold out the floor {8..12}; `p=2` survives
as a theory anchor validated by the held-out check, NOT as data-confirmed (§3.2/§3.3/§3.5 as
amended; the draft's p̂ preview was mislabeled — root-replicated).

## 11. Adversarial-read verdict (root, 2026-07-23) — **GO, with the root amendments applied above**

Load-bearing claims re-verified at source before this verdict: both constructors and the
masking structure (`architectures.py` — trunk/heads/mask lines as cited), the driver's enums,
MSE loss line, and the `--schedule sandwich` CLI-default footgun
(`kdropout_converged_width_experiment.py:100-113`, `:142-146`, `:769-771`), the 36 WSEL8
dedicated cells on disk, the matched-width ratios cited in §3.3/§5.1 (1.081/1.159/0.997 at w1-3 — root-verified against `automl_package/examples/capacity_ladder_results/WSEL8/frozen.json` `quality_at_matched_width`,
and 7.207 at w4, 1.571/1.486 at w11/12 — same leaves, `automl_package/examples/capacity_ladder_results/WSEL8/frozen.json`),
and the full pinning arithmetic REPLICATED independently from the raw cells (the draft's `A*`
reproduced exactly under its stated formula; its free-exponent preview did NOT reproduce under
its stated domain, which is what exposed the floor-region problem the amendments fix). Root
catches, beyond the §10 resolutions: the false "no `.pt` files" claim corrected (§1); the
fit-domain semantics inverted (§3.3); the mislabeled p̂ preview replaced with the honest
domain-sensitivity statement (§3.2). The MLE derivation itself (§3.1), the rescaling lemma, the
shared-parameter accounting, and the driver contract are sound as drafted. Next step per the
wave line: the driver task builds the `LossType.WEIGHTED` extension + the pinning/diagnostic
script against §6; the ROOT runs all 15 unconditional cells backgrounded, diagnostic first.
