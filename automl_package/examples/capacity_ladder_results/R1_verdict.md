# R1 interpretation checkpoint — K1/K2/K3 GO/NO-GO for K4

**Verdict: GO for K4, with amendments.** The two pre-registered STOP triggers that
halted the WS1 lane are an **artifact of an invalid de-risk instrument**
(prefix-masking the `aggregate_sparsity` surrogate), not a failure of the
per-input-count method. K4's nested training is precisely the construction that
makes the ladder valid. The `mdn_sweep` valid-ladder evidence is far cleaner and
does not refute the method; its residual noise is the expected seed-incoherence of
separately-trained fixed-k models — the exact problem K4 exists to remove.

Every load-bearing claim below is re-derived from the artifacts; the probe script
that reproduces the mechanism is
`/tmp/.../scratchpad/r1_probe.py` (CPU, `AUTOML_DEVICE=cpu`).

---

## 1. Mechanism — WHY the `aggregate_sparsity` columns pair up and swing

**Finding: prefix-masking `AggregateSparsityKSelector` is not a valid k-component
mixture ladder.** Confirmed by direct probe on toy D seed 0 (exact K0 hyperparameters,
`adaptive_bin_means=True`).

The invalidity has three compounding causes, all verified:

1. **No importance ordering.** The `k_max=6` bin heads are seeded at percentiles of
   `y` (14.3 … 85.7th → centroids `[-0.98,-0.51,-0.16,0.08,0.38,0.81]`) — ordered by
   **location**, never by importance. Under `adaptive_bin_means=True` the heads then
   move freely; converged means are `[0.74, 1.30, 0.48, -0.22, -0.50, 1.02]` — even the
   seeded location order is scrambled. "The first c components" is therefore an
   arbitrary subset in every sense.
2. **The aggregate-sparsity prior kills an arbitrary subset of bins.** Dataset-average
   usage `ū_c = mean_x softmax(weight_net)[c]` came out
   `[0.00, 0.00, 0.328, 0.148, 0.00, 0.076 | bypass 0.448]` — bins 0, 1, 4 are dead;
   only bins 2, 3, 5 are alive; the **bypass carries the most weight (0.448)**. The
   prior does its job at the aggregate level, but it empties bins at arbitrary
   positions, including the first two the prefix walks through.
3. **The model was trained as a full mixture with no prefix/nesting pressure**, and the
   K0 adapter (`_AggregateSparsityNested`) **excludes the bypass** from the ladder — so
   even the top rung (c=6) is the full model *minus its dominant component*.

The renormalized prefix weights (softmax over the first-c bin logits, averaged over x)
make the pathology mechanical:

| prefix c | renorm weights (bins 0..c-1) | col-mean nats |
|---|---|---|
| 1 | [1.000] — all on **dead** bin 0 | −96.8 |
| 2 | [0.998, 0.002] — still all on dead bin 0 | −38.0 |
| 3 | [0.000, 0.000, **1.000**] — jumps to first **live** bin 2 | −3.13 |
| 4 | [0,0,0.699,0.301] — adds live bin 3 | −2.02 |
| 5 | [0,0,0.699,0.301, **0**] — adds **dead** bin 4 → **identical column** | −2.02 |
| 6 | [0,0,0.611,0.256,0,0.132] — adds live bin 5 | −2.03 |

Increments `[+58.9, +34.8, +1.1, 0.0, −0.006]`: the huge jumps are the prefix wasting
its first slots on dead/mislocated bins and "activating" only when it reaches a live
bin; the identical pair (c4=c5) is an interior dead bin; the small terminal drop is
renormalization handing weight to a mislocated component. A valid ladder needs
prefix c = the best c-component mixture, strictly nested and non-decreasing on held-out
data. This construction satisfies none of that.

**Re-derived across all 18 K0 tables** (`score_mat` column means, from the `.pt`, not
the JSON):

- `aggregate_sparsity`: adjacent-identical column pairs up to **3 of 5**; max
  non-monotone decrease down to **−5.84 nats** (toy E seed 1 — the cited smoking gun);
  column swings up to ~94 nats (toy D seed 0: −96.8 → −2.0).
- `mdn_sweep`: **0** identical pairs on every table; max decrease between −0.004 and
  −0.067 nats; all six columns within ~0.09 nats and near-monotone.

The **cleanest proof** is on the single-mode broad twins, where a valid ladder must
abstain: the prefix ladder reports capacity advantages of **+24 nats (C_broad seed 1)**
and **+229 nats (E_broad seed 2)** on data with no structure. That is physically
impossible for a real capacity gain — it is pure renormalization artifact.

The `score_table` nested branch itself (`_capacity_ladder.py:112-128`) is **correct** —
a properly renormalized c-component Gaussian-mixture log-likelihood. The invalidity is
entirely in the *input* (percentile-tiled, sparsity-killed, un-ordered components). The
reader is fine; it needs a validly-trained nested ladder as input, which is K4.

**June never did this.** June's arbiter (`kselection_variational_em.md` §11, §14) reads
k from the held-out advantage of the *full mixture over the single best Gaussian*,
neighbour-averaged — never by prefix-masking. §4 states it explicitly: "the surviving
classes need not be a prefix … the inference keeps whichever classes the data use." §
Notation: counting "over-counts when the classes are free to move," which is exactly the
`adaptive_bin_means=True` regime. Prefix-masking is a new construction introduced at K0
that June's validated result never covered.

## 2. Attribution — genuine method failure vs invalid-instrument artifact

| STOP / failed bar | where it fired | verdict |
|---|---|---|
| **K1 P1 rail** π̂→c6 on bias-free data | `aggregate_sparsity` C s0 (π₆=0.50), E s0 (π₆=0.90) | **artifact** — the only "good" prefix is the full one, so stacking rails to 6 |
| K1 P1 rail | `mdn_sweep` D s0 (π₆=0.86) | **not artifact, not refutation** — valid columns within 0.01 nats; stacking concentrates on a near-tie where a flexible k=6 MDN's extra components are ~free on held-out. The documented [6,3,4] seed-incoherence; K4 removes it. |
| **K1 P2** pooled knee out of band 5/6 | `aggregate_sparsity` C/D/E (r*=6/5/0) | **artifact** (garbage columns) |
| K1 P2 | `mdn_sweep` C r*=3, E r*=3 (band [1,2]) | **mild over-read (off by one)** on the leaner N=2500/75-epoch de-risk table; band is "ambiguous by design" (plan K1 P2). Not a refutation. `mdn_sweep` D pooled r*=2 **in band**. |
| **K1 P3** broad-twin over-chop r*=6 | `aggregate_sparsity` C/E_broad | **artifact** — +24/+229-nat fabricated advantage on single-mode data |
| **K2 P1** per-bin beats seed-fragile (~1/3) | both methods, D/E | mixed: aggregate rows read off garbage; `mdn_sweep` rows are the real seed-fragility of separately-trained fixed-k models |
| **K3** per-input D staircase 1/3, E hump 0/3 | **100 % `aggregate_sparsity`** (`run_k3` reads only those tables) | **artifact** — the entire per-input read is off the invalid ladder |
| K3 [6,3,4] coherence: argmax {5,6}, knee {2,3} | `mdn_sweep` D | **not artifact** — residual fixed-k seed-incoherence; K4's motivation |

Verified specifics: K3 D modal knee by third — seed0 [6,5,6], seed1 [1,2,2]
(directionally right, right-shifted), seed2 [3,3,1] (truth 1/2/3). K3 E hump: 0/3
transitions near the analytic crossings [0.23, 0.77]. K2 G5 tercile-vs-sextile
verdict disagreement: **3/12** cells (present on both the invalid aggregate ladder,
D s0, and the valid `mdn_sweep`, D s0 / E s1) — a real locality instability, though I
measure 3/12, not the 6/12 in the halt summary.

**Every catastrophic signal is on the invalid `aggregate_sparsity` prefix instrument.
The residual issues on the valid `mdn_sweep` ladder are the known seed-incoherence and
leaner-table noise of separately-trained fixed-k models — neither an artifact nor a
refutation of the per-input-count claim.**

## 3. GO / NO-GO for K4 — GO, with amendments

**GO.** K4 (B3) trains one model with per-sample k~Uniform{1..k_max} and loss = NLL of
the renormalized masked mixture over components 1..k. That puts **every prefix directly
in the training objective**, forcing "the first c components" to be a valid c-component
mixture for every c — i.e. it *creates* the importance-ordered nesting whose absence is
the entire root cause diagnosed in §1. Its B-coh bar (per-k held-out NLL within 2·SE of
the separately-trained fixed-k models) checks exactly this. The de-risk did not refute
the method; it de-risked the plumbing and surfaced an instrument bug.

The positive existence proof that the claim is alive: June's arbiter on this same class
of surrogate already recovered the D and E per-input counts (§14: staircase −0.009 /
+0.147 / +0.201 at true count 1/2/3; hump −0.018 / +0.149 / −0.026 at 1/2/1). The
de-risk failed only because it used prefix-masking instead of the arbiter.

**Amendments to the K4 / WS1 spec:**

- **A. Drop the `aggregate_sparsity` prefix-masking as a de-risk/comparison instrument.**
  It is an invalid capacity ladder; it must not be a baseline, nor the "valid-ladder"
  reference for B-coh. Record the K1/K2/K3 `aggregate_sparsity` rows as
  *instrument-invalid, not a method read*.
- **B. Fix the B-coh reference explicitly to `mdn_sweep`.** The plan's B-coh already
  says "the SEPARATELY-trained fixed-k models from K0's table" — pin that to the
  `mdn_sweep` per-k columns (valid), never the aggregate prefix table.
- **C. Re-instate the June arbiter as the per-input reference reader.** K3's stated
  approach was "B5 **vs the validated reference reader**," but the K3 script never runs
  the arbiter (mixture-vs-single-Gaussian NLL, neighbour-averaged); it only prefix-reads
  the invalid ladder. K5's per-input read on the nested ladder must be cross-checked
  against the June arbiter on the same model — the cross-check the de-risk promised and
  dropped.
- **D. Align the k=1 rung with the bypass (already in the K4 spec) and treat B-order as
  load-bearing, not a formality.** The K0 adapter made bin 0 the k=1 rung and excluded
  the bypass — backwards from K4's design (k=1 = direct/bypass single Gaussian). Because
  the whole failure is *absence of ordering*, K4 must verify prefix content is
  importance-ordered and stable across seeds (B-order), with the registered fallback arms
  (sandwich draws → boosted smallest-prefix → freeze schedule) on standby; the
  nested-dropout literature warns self-ordering of NN heads is not guaranteed.

## 4. Honest limits

- **The cheap post-hoc de-risk did not certify the scientific claim; it certified the
  plumbing and caught the instrument bug.** On the valid `mdn_sweep` ladder the global
  reads are noisy (C/E pooled knee off by one; D s0 marginal rail) and per-bin gains are
  G5-fragile. This is **expected** for separately-trained fixed-k models — it is the
  compounded independent-fit noise of 18 separate MDN fits and is the documented [6,3,4]
  incoherence itself, not a deeper problem for the per-input claim. The claim now rests
  entirely on **K4 + K5**: they are its first real test.
- **Conditional risk to carry into R2.** The above is only reassuring *if* K4's single
  coherent ladder actually removes the seed-incoherence (B-knee: global knee reproduces
  across seeds) and its per-input reads survive G5. If the *nested valid* ladder also
  shows seed-incoherent global knees and G5-fragile per-bin gains, that WOULD be a
  genuine problem for the per-input-count claim — so B-knee and the G5 guard on the K4/K5
  reads are the decisive checks, not B-coh alone.
- **Not independently re-verified here** (accepted from the halt summary, non-load-bearing
  to this verdict): the broad-twin generators' <5% correctness, and the exact epoch/N
  sensitivity of the `mdn_sweep` off-by-one knee. Neither changes the attribution: the
  broad-twin failures are on the invalid prefix ladder regardless of generator fidelity.

## 5. Recommendation

**Ship K4 (GO)** with amendments A–D. Do **not** carry the `aggregate_sparsity`
prefix-masking forward in any form. Lean the WS1 de-risk narrative on `mdn_sweep`
(valid per-k ladder) + the June arbiter; report K1/K2/K3's `aggregate_sparsity` results
as the discovery of an invalid instrument, not as reads of the method.
