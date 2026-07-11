# NOV-1 — novelty & positioning of the capacity-ladder program

Working positioning note for the program report (REPORT-2). States what is new here relative
to the closest published work, grounded in the three completed lanes (WS1 ProbReg-k, WS2
FlexNN-depth, WS3 variance). Literature anchors are those already used in the program's own
design docs (`STACKING_NOTE_2026-07-05.md`, `docs/research_plan.md`); the one external claim
asserted below (hierarchical stacking = input-dependent weights) was checked against the source.

## Thesis (one line)

One instrument — a **prefix-nested capacity ladder trained once and read post-hoc by held-out
stacking / a held-out-NLL knee** — selects model capacity across three different axes: mixture
**component count** (WS1), network **depth** (WS2), and **variance-function class** (WS3). The
program's contribution is (a) that *unification*, and (b) two empirical results about the *per-input*
read that qualify the naive picture: a **faithfulness** result and a **power** result.

## The closest prior art, and where we differ

1. **Bayesian stacking — global (Yao, Vehtari, Simpson, Gelman 2018).** Fits one weight vector over a
   set of models by maximizing held-out predictive log-score. We use exactly this for the *global*
   read (WS1/WS2 π̂, the global knee). Not new; we adopt it as the aggregate instrument, and confirm
   its Occam behaviour (overfit rungs get near-zero weight; the global knee abstains on flat-capacity
   controls: WS2 G_flat r\*=0, WS3 homoscedastic twin v0 3/3).

2. **Bayesian hierarchical stacking — input-dependent (Yao, Pirš, Vehtari, Gelman 2021,
   arXiv 2101.08954).** Generalizes stacking so the model weights **vary as a function of the input**,
   partially pooled across bins/values. Their headline — *stacking helps most when predictive
   performance is heterogeneous in the input* — is the same phenomenon our per-bin read targets.
   **This is the nearest neighbour to our per-input read.** Two differences:
   - **Averaging vs capacity.** Hierarchical stacking produces input-dependent *averaging weights over
     a fixed set of K models*. We read an input-dependent *capacity/count over a **nested** ladder*
     (prefix sub-models of one trained network), where "which rung" is a statement about the
     complexity a given input needs, not just which of several models to average. The ladder is
     ordered by construction (nested dropout / masked-prefix training), so the read is a monotone
     capacity index, not a categorical model-average.
   - **A faithfulness caveat they do not report.** See below.

3. **Nested / compute-adaptive training (Rippel & Adams 2014 nested dropout; slimmable networks;
   early-exit: BranchyNet, PABEE, PonderNet [Banino 2021], CALM [Schuster 2022]).** These make every
   prefix capacity *available* and are used at inference to save compute. They do **not** provide a
   held-out, distribution-level *readout of how much capacity each input needs*, nor validate that
   readout against a known ground-truth per-input count. Our Step-2/3/4 (score table → global stack →
   per-bin stack / arbiter) is the missing readout layer on top of that training.

4. **Faithful heteroscedastic regression (Stirn et al. 2023).** Motivates WS3's angle — in-sample
   Gaussian-NLL variance heads misbehave. We add: the collapse is **finite-sample** (gone by N≈1000),
   and — the new faithfulness point — the **held-out-NLL knee is a *coarse* σ-selector**: it detects
   heteroscedasticity but cannot resolve the correct smooth σ-function class (v1 vs v2), which the
   direct σ-recovery metric can. This persists at N=4000, so it is intrinsic, not just small-N noise.

## The two new results that qualify the per-input picture

- **Faithfulness.** The naive per-input readout — the per-example knee on held-out NLL — is
  **unfaithful**: it under-reads and is seed-noisy. The faithful fine readout is a *different*
  instrument in each lane: WS1 the neighbour-averaged **arbiter** (top-vs-bottom rung advantage), WS3
  the **σ-ratio-error**. WS1's K6 router corroborates this downstream (a router trained on the smooth
  responsibilities beats one trained on the noisy knee labels on 7/9 cases). This is a caveat to the
  imported "per-bin stacking beats global exactly where truth varies" slogan: it beats global *in
  aggregate*, but the raw per-input knee is not the object that recovers the per-input count.

- **Power.** Per-input reads are **held-out-sample-limited**; the aggregate/global read is robust. WS2:
  the global knee cleanly separates varying-capacity toys (G r\*=3, H r\*=2) from the flat control
  (G_flat r\*=0), but per-bin-beats-global clears >2 SE on only 1/3 seeds at N_TEST=500 and is lost at
  finer (sextile) bins — power, not absence, with the flat-control tie as the no-false-positive guard.
  WS1: the fixed-mode per-input count recovers (D staircase, all seeds) but the moving-mode case (E)
  stays seed-fragile. The honest scope is: **capacity selection is reliable in aggregate; per-input
  capacity is recoverable but power-limited and needs the faithful (non-knee) readout.**

## The negative-existence claim (stated as our search, not as fact)

We are **not aware of a published validation** of a *prefix-nested capacity ladder read by held-out
stacking / a knee for per-input capacity **count** on probabilistic **regression*** (mixture-component
count or variance-function class). The pieces exist separately — nested training (compute-adaptive
inference), stacking (Bayesian model averaging), hierarchical stacking (input-dependent averaging
weights) — but not this combination validated against a known per-input ground truth, together with
the faithfulness/power characterization. This is the differentiator to lead with; it should be stated
as "we found no prior work doing X", not as an absolute non-existence.

## How this feeds the two planned papers

- **Paper A (ProbReg):** WS1 supplies the per-input component-count read + the arbiter-vs-knee
  faithfulness result as the "automatic bin complexity" evidence.
- **Paper B (FlexNN):** WS2 supplies the input-dependent depth read; the honest framing is
  global-depth-selection-is-robust, per-input-depth-is-power-limited (do not over-claim per-input at
  N_TEST=500).
- **Cross-cutting (this program / a possible methods note):** the *unification* — one held-out
  capacity-ladder instrument across count, depth, and variance — plus the faithfulness/power results,
  is the stand-alone methodological contribution, positioned directly against hierarchical stacking.

## Sources checked
- Yao, Pirš, Vehtari, Gelman (2021), *Bayesian hierarchical stacking: Some models are (somewhere)
  useful*, arXiv 2101.08954 / Bayesian Analysis — input-dependent, partially-pooled stacking weights.
- Yao, Vehtari, Simpson, Gelman (2018), *Using stacking to average Bayesian predictive distributions*,
  Bayesian Analysis — global stacking (via `STACKING_NOTE_2026-07-05.md`).
- Rippel & Adams (2014) nested dropout; Stirn et al. (2023) faithful heteroscedastic regression;
  early-exit line (BranchyNet, PABEE, PonderNet, CALM) — via `docs/research_plan.md`.
