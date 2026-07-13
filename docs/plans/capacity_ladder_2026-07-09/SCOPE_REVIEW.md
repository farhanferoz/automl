# Critical review of the flexible-architecture scope document (2026-07-09)

**Reviewed document:** `docs/plans/capacity_ladder_2026-07-09/INPUT_SCOPE_DOC.md`
(the "Flexible Architecture Scope Review" provided 2026-07-09 as input to this
planning round).
**Review sources:** the autocast-private rank-ladder program (design
`~/dev/turing/autocast-private/notes/design/rank_ladder/rank_ladder_design_2026-07-03.md`,
execution plan + results in the same folder, the evidence/ARD discussion
`~/dev/turing/autocast-private/notes/design/lowrank_evidence_discussion_2026-07-03.md`,
canon graveyard `~/dev/turing/autocast-private/notes/research/uncertainty/lowrank_nll_program_2026-07-02.md` §9),
this repo's June arc (`docs/kselection_variational_em_2026-06-13/kselection_variational_em.md`,
`REVIEW_HANDOVER_2026-07-03.md`, `STACKING_NOTE_2026-07-05.md`), a code survey of this
repo, and a focused literature check (§5).

**Overall verdict: the document's central recommendation is CORRECT and is now
backed by measured evidence from two projects — but it under-specifies the two
hardest transfers (ProbReg class-prefix validity; per-input routing safety), omits
the governance that made the autocast program trustworthy, and one of its
architectural claims (the stopping-rule router "preserves the prefix
interpretation") is vacuous as stated.** The corrected design is folded into
`EXECUTION_PLAN.md` in this folder.

---

## 1. Answers to the document's five review questions

**Q1 — Does nested-width training reliably produce an ordered family of
sub-networks?** YES, with two qualifications. Measured (autocast, covariance-rank
setting): the nested ladder passed every tier — Tier-0 knee r\*=5/5/6 vs planted 5;
the conditioned control was retired because nesting cost nothing (bar B3:
per-rung NLL within 2·SE of the unconstrained per-rung model, and the trained
nested fit matched the closed-form maximum-likelihood optimum digit-for-digit);
training cost is FLAT in the ladder cap (measured 77.8/74.8/82.6 s at cap 4/8/16 —
unused rungs are masked, not paid for). Literature agrees for neural widths
(nested dropout; slimmable networks; Matryoshka representation learning — §5).
Qualifications: (i) the draw distribution over capacities is a SCHEDULE, not a
prior — uniform draws are the measured-safe default; geometric/harmonic draws were
rejected in autocast because tail rungs starve exactly where the knee is read;
(ii) ordering must be VERIFIED per architecture (across-seed stability of what the
early units/classes learn), not assumed — the plan pre-registers that check.

**Q2 — Is global post-hoc selection sufficient, or does per-input routing
materially help?** Problem-dependent, and THIS REPO IS THE CASE WHERE PER-INPUT
GENUINELY HELPS — that is its measured, headline result (comparison job bpk6u9xjz:
fixed-count mixtures tie the oracle on aggregate fit but the best global k is
seed-incoherent [6,3,4] on staircase data whose true count varies 1→2→3; the
per-input held-out readout recovers the varying count on D and E). In autocast the
same question got the opposite answer (stationary process noise ⇒ global rank;
per-input rank PARKED as degenerate there). The document's framing ("is there
clear evidence...") is answered: yes here, no there — the plan's toys C/D/E are
exactly the discriminating instruments because their true per-input count is
known analytically.

**Q3 — Which routing targets are most stable?** Measured evidence says: NOT hard
per-example oracle labels. A single held-out example gives one noisy NLL vector —
the June arc needed neighbour-averaging to make per-input reads stable, and the
global best-k was seed-incoherent even with full validation sets. The stable
targets, in order of preference: (a) per-BIN stacking weights (a handful of
parameters fitted on held-out data by the concave stacking objective — the
low capacity is the safety mechanism); (b) neighbour-averaged arbiter reads
distilled into a classifier ("arbiter distillation", REVIEW_HANDOVER idea 3 —
ordinary classification likelihood on measured counts, strictly probabilistic);
(c) soft responsibilities q_i(r) as targets — acceptable smoothing of (b);
(d) hard argmin_r labels — REJECTED as the primary route (label noise;
brittleness the document itself half-acknowledges). Honesty note from the
literature check: the specific claim "oracle per-example labels are brittle" is
NOT directly settled in the literature — it is corroborated indirectly
(SkipNet gate death, MoE hard-routing instability) and directly by OUR local
measurements (the [6,3,4] incoherence; the arbiter's need for neighbour
averaging). The plan therefore keeps a hard-label arm as a registered PILOT
inside K6 rather than asserting its failure from citations. Hierarchical
stacking (Yao et al. 2021) is the principled generalization of (a) when a
smooth weight function is wanted (§5).

**Q4 — Does a prefix-preserving stopping-rule router beat unconstrained gating?**
The question is MISPOSED, and the document's claim for it is vacuous. The prefix
property lives in the LADDER (sub-model r is a prefix of sub-model r+1 by
construction); ANY router that outputs a distribution over r — softmax, stopping
rule, anything — routes among prefix models. The stopping-rule parameterization
q(r|x)=(∏_{k≤r} c_k(x))(1−c_{r+1}(x)) is a bijective reparameterization of a
generic distribution over {0..n}; it adds no constraint and preserves nothing the
plain softmax doesn't. What DOES matter (measured, autocast §5b): free per-unit
gates g_k multiplying units are VACUOUS when the units are also free (pure
reparameterization, unidentifiable) — the useful "switch" is evaluation-time
truncation at the arbiter's knee, never a learned gate. The plan drops the
stopping-rule router as a distinct arm and keeps: masked evaluation of prefixes +
a plain softmax router distilled from held-out reads.

**Q5 — Adaptive compute vs genuine specialization?** For this repo the value is
NOT compute (toy scales are trivial; ProbReg evaluates all k in one forward pass
anyway) — it is STATISTICAL: (i) per-input count recovery is the scientific
claim itself (k as an SNR/difficulty-adaptive resolution dial); (ii) one nested
model replaces K separately-trained fixed-k baselines, removing the seed
incoherence that made fixed-k comparisons unstable; (iii) a coherent truncation
ladder gives the arbiter a full per-input advantage-vs-k CURVE (knee = count)
instead of the current single mixture-vs-Gaussian advantage. The document's
compute/latency metrics (its Phase 3) are the wrong primary metrics here; the
plan's bars are count-recovery vs the analytic ceiling and held-out NLL vs
oracle. (For FlexibleNN, compute saving at inference is a real but secondary
benefit — recorded as a reported metric, not a bar.)

## 2. Corrections (the "no passes" items)

**C1 — The width→ProbReg mapping is not a straight prefix, and the document never
notices.** For hidden units (exchangeable, sum into a readout), truncation is
trivially valid. For ProbReg's classes it is NOT: the classes tile the OUTPUT
range (percentile bins), so "the first r classes" of a mean-ordered set does not
cover the target range — a prefix would amputate the upper tail. Prefix validity
requires (a) mixture scoring (renormalized masked softmax over active components
is then a valid k-component conditional mixture), and (b) component means free to
move (the adaptive-heads regime — which the June arc showed is also the regime
where eff#-style in-fit counts over-count and only the held-out arbiter stays
faithful), and (c) importance-ordering, which CONFLICTS with the repo's new
mean-ordering identifiability penalty (commit 50b1f62) — you cannot order
components by location and by importance simultaneously. The plan resolves this
explicitly (nested arm trains with the ordering penalty OFF or restricted to the
active prefix; the conflict is a pre-registered ablation, not a silent choice).
This is the single largest under-specification in the document.

**C2 — The document's Phase-3 router violates the measured safety line.** Its
two-stage hard routing trains a router on per-example oracle widths; the June
[6,3,4] instability and the arbiter's need for neighbour averaging show
per-example labels are noise. The measured-safe route is per-bin stacking (~15
numbers) or arbiter distillation on neighbour-averaged reads. The document's own
risk table names "label noise in oracle widths" but proposes it anyway as the
primary route; the plan inverts that ordering.

**C3 — The knee rule is imported without its guardrails.** The autocast program
learned, at measured cost, that a bare knee rule is not deployable: it needs
(i) block-bootstrap standard errors (the knee moves with readout size, SE ∝ 1/√N —
measured T\*=3–59 trajectories); (ii) abstain semantics (r\*=0 = NO-READ, never
"capacity 0 confirmed"); (iii) the CAP-SATURATION rule (r\*=r_max ⇒ invalid read,
double the cap and rerun); (iv) per-regime/per-bin cells as a MANDATORY backstop
(the T1c demoR confound {7,4,7} fired while every global gauge stayed clean);
(v) a pooling/locality guard (neighbour width can MANUFACTURE multimodality that
exists at no single input — the E6 lesson, flagged for this repo in
REVIEW_HANDOVER's pooling-artifact check). All five are standing rules in the
plan.

**C4 — The mixture objective needs its likelihood footing stated.** The document
writes exp(−ℓ_r) for a generic "loss"; that quantity is a likelihood only when
ℓ_r is a negative log-likelihood. This repo's standing ruling (strictly
probabilistic, mixture scoring leaning) makes it exact here — but it also means
the whole program is CONDITIONAL on mixture scoring for ProbReg (REVIEW_HANDOVER
FRAMING 5): under blend scoring, k is nearly invisible to the likelihood and the
score table is mush. The plan makes mixture scoring an explicit precondition,
with the blend-scored model kept only as a deployment-summary comparison.

**C5 — "Evaluate every width post hoc is costly" — false for both our models.**
One forward pass yields all per-class (mean, log-variance, logit) triples for
ProbReg — the full score[i,k] table for all k follows analytically from cached
per-component quantities (same trick as autocast's nll_mat, F×(r_max+1), which
made their entire STACK battery pure post-hoc analysis). FlexibleNN evaluates all
depths in one pass through the layer stack (each depth is a readout off a shared
trunk). The document's "coarse-to-fine screening" mitigation solves a non-problem;
the plan mandates saving the full score table as the primary artifact of every
run (their `_nllmat.pt` convention), which is also what makes every selection
method above a cheap, re-runnable post-hoc analysis.

**C6 — The fully-trained mixture deserves a harder verdict than "expensive".**
Measured: 6.4× the nested arm's cost at cap 8 (linear in rung count vs flat);
in-fit π rails toward the cap under any persistent structure and its railing
mode misleads every scalar summary; its occasional predictive wins were traced
to scale-absorption (a mechanism) and recovered honestly by per-bin scale
recalibration on held-out data. In this repo the same fact appears as the April
finding (un-regularized selection cap-tracks; E[k|nb] follows (k_max+2)/2). The
plan does not carry a trained-mixture arm at all; the mixture PREDICTIVE is
obtained by stacking on the nested ladder at ~zero cost.

**C7 — In-sample capacity charges: the document under-states what is known.**
"Mixture overfitting" is listed as a risk; it is a measured structural fact with
arithmetic: the in-sample pull is ~½ nat of training log-likelihood per spurious
parameter, while any fixed prior charge is O(10) nats total at every admissible
concentration — the prior loses at every knob setting (autocast E4b: α₀ 0.1 vs
1.0 identical to 3 decimals; this repo's prior ablation: prior-ON ≈ prior-OFF ≈
plain mixture under adaptive heads). Consequence for the plan: NO in-fit
complexity device is load-bearing anywhere — not as a fallback, not "for
stability". The complexity charge is levied by fresh data only. (K_PENALTY-class
terms remain banned by standing ruling regardless.)

**C8 — What the document omits entirely.**
(i) The scoring-rule precondition (C4). (ii) Pre-registration discipline — bars
before runs, 3 seeds, no verdict off a single reading, STOP-on-failed-bar —
without which the June/autocast history says results do not survive review.
(iii) The identifiability caveat: held-out selection defeats memorized SAMPLING
noise, not persistent MODEL-BIAS structure (autocast STACK-5: held-out π̂ rails
under bias exactly like in-fit π; their preflight gauges exist for this). In
this repo's supervised setting the analogue is conditional-MEAN misfit
masquerading as extra components — the plan imports a misfit tell
(shrink-the-neighbourhood stability + the fixed-k-sweep sanity check) rather
than assuming immunity. (iv) The variance workstream (below) — the same
disease in a third organ, absent from the document. (v) Toys with analytic
ceilings already exist here (C/D/E with known per-input counts) — the document
proposes generic toys and loses the ability to grade reads against truth.

## 3. What the document gets right (kept as-is)

The primary recommendation (nested ladder as the trained object; selection
separated from representation learning; post-hoc knee + post-hoc stacking as the
two global readers; conditioned arm as a control only; trained mixture rejected
as the operational method; the four-module decomposition of the problem). All of
this now has measured support from two projects and the plan adopts it without
modification. The risk table's "weak self-ordering" and "label noise" rows
correctly anticipate the two real failure modes (addressed by C1's ordering
check and C2's routing inversion).

## 4. The variance extension (not in the document; scoped on request)

The same failure mechanism — in-sample fit gain ~N per parameter always beats any
fixed in-sample charge — afflicts jointly-fitted heteroscedastic Gaussian NLL:
the mean head's in-sample residuals shrink as it overfits, dragging the fitted
σ̂(x) below the true noise (and the NLL's gradient geometry adds its own
pathologies — the Seitzer 2022 line). The autocast reconciliation supplies the
correct target semantics: a variance fitted as a nuisance legitimately absorbs
model error, and for CALIBRATION that is exactly what is wanted — but then the
residuals it absorbs must be HELD-OUT residuals (true noise + honest
generalization error), never in-sample ones (shrunken by overfitting). MacKay's
evidence framework (and Minka's dimensionality selector — the "evidence
framework" family referenced in discussion) is the principled IN-SAMPLE route
for GLOBAL, low-dimensional nuisances on (near-)well-specified models — exact
for linear-Gaussian regression, Laplace-approximate for small networks — and it
was measured in autocast to break exactly where persistent bias enters (khat
rails; the Occam race). Consequently the plan's variance workstream is a ladder
of mechanisms matched to capacity: evidence for global σ²/weight-decay on the
linear model (exact, cheap, classical); held-out / cross-fitted residual fitting
for σ(x) heads; per-bin scale recalibration (the measured STACK-2b remedy) as
the low-capacity heteroscedastic default; β-NLL and joint NLL as the baselines
they must beat. Full detail: `EXECUTION_PLAN.md` WS3.

## 5. Literature cross-check (focused web verification, 2026-07-09)

Headline: nothing found contradicts the core architecture; three sub-claims are
weaker than "settled literature" and are carried as REGISTERED ASSUMPTIONS with
fallback arms in the plan, not as citations.

- **Nested dropout** (Rippel, Gelbart & Adams 2014, ICML): the PCA-equivalence
  ordering guarantee is proved for semi-linear autoencoders only. The
  supervised follow-up (nested dropout for compact CNNs, arXiv:1412.7155)
  reports that vanilla truncation draws leave late units learning very slowly
  and needed a "unit sweeping" freeze schedule to converge. ⇒ Plan consequence:
  K4/F2 carry a registered ordering bar (B-order) with pre-authorized fallback
  arms (freeze schedule; sandwich draws; boosted smallest-prefix weighting) —
  self-ordering is an assumption to verify, not a given. (Autocast's clean
  ordering was measured in a closed-form-solvable covariance setting; it does
  not automatically transfer to NN heads.)
- **Slimmable / universally slimmable / once-for-all** (Yu et al. 2018; Yu &
  Huang 2019; Cai et al. 2020): sub-models match individually-trained nets at
  scale; the load-bearing tricks are switchable BatchNorm (moot if our ladder
  arms run BN-free — and FlexNN's blocks CAN contain BN, so ladder arms pin BN
  off or use per-depth statistics), the sandwich rule, and in-place
  distillation. No direct evidence for small BN-free tabular MLPs — a real
  pilot risk, not an established result.
- **Matryoshka representation learning** (Kusupati et al. 2022): strong
  large-scale confirmation for embeddings/classification/retrieval, INCLUDING
  the caveat that uniform per-prefix loss under-serves the smallest prefix
  (their boosted variant recovers ~3%). NO published validation for regression
  heads — a genuine literature gap; this is simultaneously a risk (registered)
  and a novelty opportunity (recorded in NOV-1).
- **Bayesian stacking** (Yao, Vehtari, Simpson & Gelman 2018) and
  **hierarchical stacking** (Yao, Pirš, Vehtari & Gelman 2021): confirmed as
  the accepted method for held-out-fitted global and INPUT-DEPENDENT model
  weights with partial pooling as the overfitting safeguard — the reference
  method behind K2/B5. Concavity of the global objective needs no citation:
  Σ_i log Σ_c π_c L_ic is concave in π (log of a linear function of π), hence
  the EM fit is exact.
- **Heteroscedastic pathologies**: Seitzer et al. 2022 (β-NLL; explicitly NOT
  a proper distributional fix) and Stirn et al. 2023 (faithful heteroscedastic:
  stop-gradient decoupling — still in-sample variance) confirm the disease and
  the two in-training fix families. Skafte et al. 2019 name a post-hoc
  mean-then-variance heuristic (mechanism unverifiable from the abstract).
  Directly on point: "Practical Deep Heteroskedastic Regression"
  (arXiv:2603.01750, 2026) — fit the mean, freeze, fit variance on HELD-OUT
  data; reports it competitive with or better than joint NLL, β-NLL and
  faithful-HR. Brand new, so V2 treats it as directional confirmation to be
  reproduced, not consensus to lean on.
- **Evidence framework**: MacKay 1992 + the modern Laplace instantiation
  (Immer et al. 2021, scalable marginal likelihood; Immer et al. 2023 for
  heteroscedastic regression) are real and practical. Minka 2000's correctness
  guarantee is under its own well-specified Gaussian assumptions; its
  misspecification behaviour is NOT documented in the literature — our
  program's own measurements (autocast: khat fires on bias-free hetero-D and
  heavy-tail data) are currently the best evidence on that point and V-tasks
  inherit the burden of the discriminating experiment (V0/V1).
- **Process caution carried into governance**: one fetched "supporting quote"
  was caught as a fabrication during verification (JEI-DNN, oracle-label
  claim). All load-bearing citations in RESULTS.md must quote text verified
  against the actual source, per the report's standard.

## 6. Annex — the Mixture-of-Experts arbitrariness audit (Q9, verified 2026-07-09)

Feeds the plan's NOTE-MOE task. **Verdict: the user's claim is SUPPORTED —
mainstream LLM sparse-MoE training departs from strict probabilistic inference
in identifiable, citable ways** — with one nuance the note must carry: every
"principled" alternative found either RELOCATES the arbitrary choice or CHANGES
the problem, none removes it for sparse discrete routing at scale.

**The non-likelihood ingredients (each a citable heuristic):**
- Load-balancing auxiliary losses: Shazeer et al. 2017 (arXiv:1701.06538,
  importance + load losses, coefficient-of-variation based); Switch
  Transformer (Fedus, Zoph, Shazeer 2021) simplifies to one loss with a
  hand-tuned coefficient α=0.01 chosen empirically to trade utilization
  variance against perplexity; GShard (Lepikhin et al. 2020) adds a random
  second-expert routing rule justified as an engineering approximation.
- Router z-loss: ST-MoE (Zoph et al. 2022, arXiv:2202.08906) — a numerical-
  stability patch on unbounded router logits, weight ~0.001, no likelihood
  derivation; adopted by essentially all later large-scale MoE recipes.
- Capacity factor + token DROPPING (GShard, Switch): a throughput constraint
  whose overflow the auxiliary loss exists to suppress — one heuristic
  patching another.
- Top-k hard gating: discrete forward, soft/straight-through gradients,
  discrete-only inference — different computational graphs at train vs test.
- Noisy gating (Shazeer 2017): exploration noise justified by its balancing
  effect.

**The failure without them:** routing collapse / rich-get-richer (Wang et al.
2024, arXiv:2408.15664 — the DeepSeek aux-loss-free method paper): preferred
experts get more and cleaner gradient, unpreferred ones decay, the router
learns to avoid them harder. The paper itself notes the standard fix's tension:
too large an auxiliary loss impairs the model. **This is structurally the same
mechanism as our measured B1 pathologies** (April cap-tracking; autocast in-fit
π railing; the June per-input weight freeze): an in-fit gate trained with no
honest complexity/usage charge, patched by a hand-weighted term — the exact
device banned here as K_PENALTY-class, whose principled counterpart in this
program is the aggregate Dirichlet usage prior (one prior, charged once, on
dataset-average usage).

**The alternatives, and what each actually does:** Jacobs–Jordan 1991 / Jordan
& Jacobs 1994 (EM latent-variable MoE) — fully principled but dense and never
scaled to sparse activation; BASE layers (Lewis et al. 2021, arXiv:2103.16716)
— optimal-transport balanced assignment, removes the aux loss but installs
uniform load as an underived objective; Expert Choice (Zhou et al. 2022,
arXiv:2202.09368) — balance by construction, at the cost of violating the
causal constraint (its own patch required); Soft MoE (Puigcerver et al. 2023,
arXiv:2308.00951) — genuinely aux-loss-free but no longer discrete sparse
routing (a different computational object); DeepSeek-V3's aux-loss-free bias
(arXiv:2412.19437; theoretical framing arXiv:2512.03915 as dual ascent) —
relocates the free choice from loss coefficient α to bias step-size γ.

**Verification caveats the NOTE-MOE author MUST resolve against primary
sources before quoting:** (i) Shazeer et al. 2017's own framing of their
losses (the "heuristic" characterization currently rests on secondary
sources — the primary PDF did not parse); (ii) the "relocates rather than
resolves" reading of the dual-ascent framing (PARTLY verified, from a fetch
summary). Two strongest verified citations for the verdict: Switch
Transformer (hand-tuned α + capacity factor/token dropping) and ST-MoE
(z-loss as an explicit stability patch).
