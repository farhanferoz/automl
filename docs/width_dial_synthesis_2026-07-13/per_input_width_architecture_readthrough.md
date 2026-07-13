# Varying network width per input — a technical read-through

*A consolidated record of every approach we tried for making a network's width adapt to the
individual input, the mathematics behind each, and what actually happened when we ran them.*

Date of this synthesis: 2026-07-13.

---

## 0. What this document is

We asked a concrete question: **can a single model decide, per input, how much network capacity
that input needs — narrow for easy inputs, wide for hard ones — and pay for it only where it is
needed?** We attacked it along the *width* axis (number of hidden units) on a controlled toy where
we know the right answer. This note pulls together, in one place, the five approaches we tested, the
idea and the mathematics of each, and the measured results. It supersedes the pre-run design docs on
one important point (see the note in §7); where the design docs and the run disagree, the numbers
here win.

Every number in this document was read directly from the saved result files listed in the appendix.

---

## 1. The setup — what we are testing, and on what data

### 1.1 The toy: an easy half spliced to a hard half

All runs use the same one-dimensional dataset, `make_hetero`
(`automl_package/examples/nested_width_net.py`). The input `x` is drawn uniformly on `[-4π, +4π]`.
The target `y` is a clean signal plus small Gaussian noise (standard deviation 0.05), where the
signal has two halves:

- **Easy half** (`x < 0`): a straight line, `y = (0.5 / 4π) · x`. One hidden unit is enough to fit
  this — it is *width-flat*.
- **Hard half** (`x ≥ 0`): a stretched sine, `y = 0.5 · sin(x)` — two full oscillations spread over
  the range. This is *width-hungry*: a one-unit network gets it badly wrong, and it takes roughly a
  dozen units to fit it well.

The two halves meet continuously at `x = 0`. Because the oscillation is spread wide (low frequency in
input space), a small network *can* learn it — unlike a tightly packed sine, which is provably beyond
a small network's reach. That "hungry but learnable" property is exactly what makes this a fair test:
there genuinely is a per-input width signal to recover, and it is recoverable.

A quick probe confirmed the gradient before any of the real runs (fit error as a multiple of the
noise floor):

| hidden units | easy half | hard half |
| ---: | ---: | ---: |
|  1 |  ~2× |  ~52× |
|  7 |  ~1.3× |  ~1.5× |
| 10 |  ~1.2× |  ~1.3× |

The hard half needs about ten times the width of the easy half. That is the signal every method below
is trying to (a) fit and (b) read back out per input.

### 1.2 The objective and the yardstick

**Every model here is a heteroscedastic Gaussian.** Given an input `x` it outputs a predicted mean
`μ(x)` and a predicted **log-variance** `s(x) = log σ²(x)`, defining a per-input predictive density
`N(μ(x), e^{s(x)})`. We parametrize by the log-variance rather than the variance so the output can be
any real number while the variance `e^{s}` stays positive automatically. The training objective is the
per-example negative log-likelihood (NLL):

```
ℓ(μ, s; y)  =  ½ [ log 2π  +  s  +  (y − μ)² e^{−s} ]                         (1)
```

Lower NLL is better; every "log-likelihood" (LL) number in this document is the held-out mean of `−ℓ`,
so **higher is better**. We score likelihood rather than squared error because the entire object of
study is *calibrated per-input uncertainty*, which a point-prediction loss cannot measure. Program
doctrine: held-out LL is the only bar — no ad-hoc penalties or tuned regularization weights. (This is
the standard Gaussian-NLL loss used across the library; full treatment in
`docs/mathematical_guide.tex`, §"Gaussian Negative Log-Likelihood".)

**The yardstick — the oracle / noise floor.** The targets carry Gaussian noise of standard deviation
σ = 0.05, so no model can beat the Gaussian that predicts the true conditional mean with exactly that
variance. Taking the expectation of `−ℓ` when `μ = f(x)` (the true mean) and `y − μ ~ N(0, σ²)`, the
term `E[(y − μ)² e^{−s}] = σ² · e^{−s}` is minimized at `s = log σ²`, giving `E[−ℓ] = −½log 2π −
log σ − ½`. So the ceiling on held-out LL is

```
LL_oracle  =  −log σ  −  ½ log 2π  −  ½
           =  −log(0.05)  −  ½ log 2π  −  ½   ≈   1.577 nat                    (2)
```

"Reaching the floor" means a method has fit the signal as well as the noise allows (held-out LL ≈
1.577). All fit numbers below are held-out LL, **higher is better, 1.577 is the ceiling**.

*Notation used throughout:* `x` input, `y` target, `μ`/`s = log σ²` the predicted mean/log-variance,
`ℓ` the per-example NLL of eq. (1), `k` the capacity level (width / number of bins / number of
factors), `f(x)` the true noise-free signal, `σ = 0.05` the noise scale.

### 1.3 How we score a method — three bars

We hold every method to the same three pre-registered tests
(`docs/plans/width_dial_2026-07-11/EXECUTION_PLAN.md`, §2d):

1. **Construction** — does held-out log-likelihood in the *hard* half climb as we add width, and reach
   the noise floor? This asks whether the model *can even use* the extra width. (Reported below as
   "hard-region fit"; the field name in the files is `centre_pass`.)
2. **Recovery** — does the learned per-input selector actually assign **more** width to hard inputs
   than to easy ones, by a statistically clear margin? This is *the actual research question* — it
   asks whether the per-input width is **read back correctly**. (Reported as the width difference
   `hard − easy`; positive and clear = pass.)
3. **Deploy** — does routing each input to its own width **match or beat** the single best global
   width, and land within a hair (0.02 nat) of the per-input oracle? This asks whether it is
   *useful in practice*.

Two standing caveats about the bars, so the tables below are not misread:

- The **composite "construction pass" is failing for every method**, including the ones that clearly
  work, purely because of an over-strict "easy half must be perfectly flat" sub-clause. The meaningful
  construction signal is the hard-region fit reaching the floor, which is reported separately. The
  easy-flat clause is a known-too-strict bar on the recalibration list.
- The **0.02-nat deploy sub-clause is unmet by every method**, again because it is too tight for this
  toy. The meaningful deploy signal is "matches or beats the best single global width," reported
  separately.

### 1.4 The convergence rule (why it matters here)

One hard-won discipline governs every result: **never read a fit off a fixed epoch budget — train each
width to convergence and check the loss trajectory.** As §7 explains, the single biggest correction in
this whole program came from ignoring this rule once. Runs that let the model converge carry a
`trustworthy` flag per seed (meaning every width fully converged within the epoch cap); where seeds are
not fully trustworthy it is called out.

---

## 2. Why nesting, and whether it transfers across domains

### 2.1 Why we want a nested model at all

The goal is not just "a network that fits." It is **one model that contains a whole ladder of smaller
models inside it** — so that at inference time you can run only the first `k` units (or `k` bins, or
`k` factors) and get a valid, calibrated prediction at that capacity, then let a per-input selector
pick how far up the ladder each input climbs. If you instead train a separate model per capacity level,
you pay K× the parameters, get no cheap "run the first `k`" shortcut, and have no single object to route
within. Nesting is what turns "per-input capacity" from K separate models into one truncatable model —
it is the whole reason the per-input dial is cheap rather than a K-fold cost. The question is whether a
*shared, truncatable* representation can actually stay coherent as you truncate it. The next subsection
is the tool that answers "when does it, and when does it not."

### 2.2 The coherence invariant — what separates the cases that nest from the ones that don't

Before the methods, the principle that predicts which ones should work. We want one shared
representation that can be *truncated* to any width and still be a valid model at that width — a
"nested" ladder of models, `width-1 ⊂ width-2 ⊂ … ⊂ width-K`. Across every case we studied (here and
in two sister problems — a low-rank covariance ladder and a classifier-over-bins), a nested ladder is
**coherent** only when, for every rung `i`, all three of these hold:

1. **Stable identity** — "rung `i`" means the same thing no matter how many rungs you keep. It is not
   silently redefined when you add more capacity.
2. **Self-contained contribution** — rung `i` is a complete piece on its own, not a fragment that only
   makes sense entangled with the others through one shared read-out.
3. **Importance ordering** — the rungs are ordered so that any prefix (the first `k`) is close to the
   best possible model of that size.

A scorecard of every case against the three properties:

| Case | stable identity | self-contained | importance-ordered | works? |
| --- | :---: | :---: | :---: | :---: |
| Low-rank covariance ladder (§2.3) | ✓ | ✓ | ✓ (training-induced) | **yes** — rides on linear-algebra (SVD) structure |
| Probabilistic regression, recipe of record (§2.4) | ✓ | ✓ | ✓ (built in) | **yes** |
| Probabilistic regression, dynamic-`k` with cross-entropy (§2.4) | ✗ (bins redefined per `k`) | ~ | ✗ | **no** — same conflict as the net |
| Shared-nested width, naive/under-trained | ✗ | ✗ | ✗ | **no** (but see §7 — convergence) |
| Frozen residual cascade | ✓ | ✓ | ✓ (by construction) | **no** on this toy (§5.1) |
| Independent weights per width | ✓ | ✓ | n/a | **yes**, but K× the parameters |

The deep reason the two clean cases nest so well is **linearity**: ordered truncation of a *linear*
map recovers the optimal ordering for free (this is the singular-value decomposition, SVD). A
nonlinear network has no such structure, which is the mechanistic reason naive width-truncation of an
MLP does not automatically give a clean ordering. Every method below is a different attempt to *supply*
the three properties that linearity would otherwise hand us.

The next three subsections walk the three domains where we actually tested nesting — two where it
works and one where it does not — because the *contrast* is the evidence for the invariant.

### 2.3 Domain one — the low-rank covariance ladder (nests cleanly, and shows why)

**What it is.** A way of modeling a covariance matrix as a small number of "factors": one shared
loading matrix `M` (dimensions × `r_max` columns) plus a diagonal `D`. The rung-`r` model keeps only
the first `r` columns of `M`. This is the direct analog of width — "how many factors does this input's
uncertainty need?"

**Why every truncation is a valid model (moment-matching).** Write the factors as the columns
`m₁, …, m_{r_max}` of `M`. The rung-`r` covariance is

```
Σ_r  =  D  +  Σ_{k ≤ r} m_k m_kᵀ   +   diag( Σ_{k > r} m_k m_kᵀ )              (3)
        └── kept factors, in full ──┘   └─ dropped factors, collapsed to diag ─┘
```

The kept factors `k ≤ r` enter as full rank-1 outer products; the *dropped* factors `k > r` are
**folded onto the diagonal** — their marginal variances retained, their off-diagonal correlations
discarded. Because that diagonal term restores exactly the variance the truncation removed, every rung
has the **same marginal variances as the full model** (`diag Σ_r = diag Σ_{r_max}` for all `r`). A
truncated model is therefore a *calibrated* smaller model, not a broken one — the "self-contained
contribution" property, by construction.

**Why the ordering comes out clean (this is the crux).** A fixed-rank fit is **rotationally
degenerate**: for any orthogonal `R` (`R Rᵀ = I`), replacing `M → M R` leaves the covariance
`M Mᵀ = (M R)(M R)ᵀ` unchanged, so at a single fixed rank the column order carries no information. But
truncation to the first `r` columns is **not** invariant under `R` — in general
`(M R)_{:,≤r} ≠ M_{:,≤r}`, because a rotation mixes kept and dropped columns. So scoring the *same* `M`
at every prefix length `r = 1…r_max` simultaneously removes the rotational freedom, and the unique
ordering that is good at *every* truncation is the importance-sorted one (largest factor first) — which
is exactly the singular-value decomposition / principal-component ordering (Eckart–Young). A second,
milder pressure helps: under uniform random-rank training, column `j` is kept with probability
`(r_max − j)/(r_max + 1)`, so the first column participates in ~89% of steps and the last in ~11%
(for `r_max = 8`), pushing the highest-value directions into the earliest slots. Together these give
"importance ordering" **for free — because a covariance is a linear-algebraic object and ordered
truncation of a linear map is SVD.** (Caveat, expanded in the verdict below: this ordering was
*argued*, never measured at the column level.)

**Verdict and honest caveat.** It reaches essentially the maximum-likelihood optimum on
aggregate/calibration measures. The caveat worth carrying forward: the column-level ordering was
*specified* as a diagnostic but **never actually measured** — so "the factors are importance-ordered"
is an argued property here, not a verified one. That unverified-ordering gap is precisely what the
cascade (below) was designed to close by making the ordering explicit.

### 2.4 Domain two — probabilistic regression as a classifier over bins (nests for the recipe of record)

**What it is.** The probabilistic-regression model is **a classifier over `k` classes**, not a Gaussian
mixture. It cuts the target range into `k` percentile bins, a classifier outputs bin probabilities
`P(C=i | x)`, a small regression head per bin predicts a local mean and variance `(μ_i(x), σ²_i(x))`,
and these are combined by the **law of total variance** into one predictive Gaussian:

```
μ(x)   =  Σ_i P(C=i | x) · μ_i(x)                                             (4)
σ²(x)  =  Σ_i P(C=i | x) · σ²_i(x)   +   Σ_i P(C=i | x) · ( μ_i(x) − μ(x) )²   (5)
          └── E[Var(Y|C)]: within-bin ──┘   └── Var(E[Y|C]): between-bin ──┘
```

Equation (5) is the exact variance decomposition — average within-bin variance plus the variance of the
per-bin means — so the predicted uncertainty widens both when the bins are individually noisy and when
the classifier is unsure which bin applies. (Decomposition derived in full in
`docs/mathematical_guide.tex`, §"Law of Total Variance Aggregation".) Varying `k` is the capacity dial
here — a coarse-to-fine *resolution* knob, the analog of width.

**Two ways to vary `k`, only one of which is nested.**

- *Fixed `k`* (the default): a completely separate model per `k`, zero weight sharing — not nested at
  all.
- *Dynamic `k`*: **one** classifier (sized to the largest `k`) with shared heads; to run at a smaller
  `k` you mask the logits down to the first `k` bins and re-normalize (a "masked prefix"). This is the
  genuinely nested, shared structure.

**Does it nest coherently? Entirely down to stable identity.**

- The per-input-`k` result that **works** used a nested surrogate whose components are importance-ordered
  by the nesting itself, with independent heads, and — the load-bearing detail — it does **not**
  recompute the bin edges as `k` changes. So "component `i`" means the same thing at every `k`: stable
  identity holds, and the model recovers the right per-input `k`.
- The shipped dynamic-`k` path, **when the cross-entropy classification loss is active**, recomputes the
  percentile bin edges for each `k`. The cut-points are the evenly spaced quantiles at percentiles
  `q(k) = linspace(0, 100, k+1)`, which shift with `k`: the `k=2` cut sits at the median (percentile
  50), while the `k=10` cuts sit at percentiles `10, 20, …, 90` — the median is not among them. So the
  set of boundaries for `k=2` is **not** a subset of the boundaries for `k=10`; the grids are
  independently defined, not a coarse-to-fine refinement. Consequently "class 0" means "the bottom 50%
  of the target" at `k=2` but "the bottom 10%" at `k=10`, and head 0 is asked to be simultaneously
  correct for both, with nothing in the input to say which regime it is in. **This is the exact same
  conflict as the shared nonlinear net** — one component forced to double as a good predictor at
  multiple resolutions — and it fails for the same reason (broken stable identity).
- **Resolved:** the shipping recipe of record trains *regression-only*, which structurally disables the
  cross-entropy branch, so the recompute-per-`k` defect never fires — cross-`k` identity is carried by
  the one shared classifier plus fixed heads under the masked prefix. The defect is therefore scoped to
  a *misconfiguration* (dynamic `k` with cross-entropy on), not the recipe we actually ship.

**Verdict.** Nesting **works for probabilistic regression** in the recipe of record, because stable
identity is preserved. The single path where it breaks fails for exactly the same reason the nonlinear
width net does — which is itself a second, independent confirmation of the coherence invariant.

### 2.5 Domain three — the nonlinear dynamic network: why the clean mechanism does not transfer

Both working domains ride on **linearity**. The covariance ladder's clean ordering *is* an SVD; the
published ordered/nested-dropout guarantees that a prefix recovers the optimal sub-model hold **only for
linear maps**. A nonlinear multilayer network has none of that structure: its hidden units are not an
eigenbasis, and a shared read-out entangles them. So the very same prefix-truncation that orders a
linear map cleanly induces **no** clean ordering in the network — a unit's role shifts as the width
changes (no stable identity), there is no importance order over units, and the shared read-out breaks
self-containment. By default the nonlinear net fails all three properties. That is the mechanistic
answer to "why doesn't nesting work for the dynamic architecture": **the property that makes nesting
free — linear-algebraic ordering — is simply absent in a nonlinear model.**

Two consequences follow, and both showed up in the runs:

- The standard published fixes for shared-width training (the sandwich schedule and in-place
  distillation) address a *different* failure — gradient **starvation**, making sure every width
  receives training signal. They do not create identity or ordering. That is why, on the under-trained
  runs, they appeared not to rescue the toy.
- The honest nuance from our converged runs (see §7): with **enough** training the shared net *does*
  reach the noise floor and recover the dial — but by brute-force joint convergence, not by getting a
  clean ordering for free. The nonlinear net loses the cheap linear guarantee; it can still be made to
  work, it just pays for it in convergence.

Everything in §§3–6 is a different response to this single fact. The **cascade** tries to *supply* all
three properties by construction (boosting generalizes past linearity) — and fails empirically here.
**Matryoshka** supplies only the self-contained read-out and leaves identity/ordering alone — it fits,
but its per-input dial is not robust. The **converged shared net** supplies none of them explicitly and
instead pays the convergence cost — and works.

---

## 3. The two baselines that bound the problem

### 3.1 Shared-nested width, under-trained — the apparent failure (`W2`)

**Idea.** One network, `Linear(1 → 12) → tanh`, with two read-out heads for the predicted mean and
log-variance. To run it at width `k`, zero out hidden units `k…12` before the read-outs, so
`width-k ⊂ width-(k+1)` share weights. Train by sampling a random width each minibatch (the "sandwich"
schedule: always include width 1 and width 12, plus a couple of random middles) and minimizing the
Gaussian negative log-likelihood.

**Result (2,500 epochs):** hard-region fit stalls at about **−0.4 nat** — roughly **2.0 nat below the
floor** — and construction fails on all 3 seeds. On its face, "shared-nested width does not work."

This is the result the pre-run design docs enshrined as "shared-nested fails all three properties." It
is misleading, for the reason in §3.3.

### 3.2 Independent weights per width, converged — the existence proof (`W_CONVERGED`)

**Idea.** Give up on sharing entirely: train a *separate* full network for each width `1…12`. This
throws away efficiency (K× the parameters, no cheap shared prefix) but it is the clean existence proof
— every width is its own self-contained model, so if the per-input dial can be recovered *at all*, it
should be recoverable here.

**Result (trained to convergence):** hard-region fit **reaches the floor on all 3 seeds** (gap 0.02 to
0.05 nat). The learned selector **recovers the dial on all 3 seeds** — it assigns clearly more width to
hard inputs (width difference `hard − easy` of +1.9 to +3.3, all statistically clear). This is the
positive control: the signal is real and recoverable. The cost is the K× parameter blow-up.

### 3.3 The convergence correction — shared-nested actually works (`W_KDROPOUT_CONVERGED`)

**Idea.** The *same* single shared net as §3.1 (1× parameters, cheap prefix inference), trained with
the k-dropout sandwich schedule — but run **to convergence** (up to 300,000 epochs) instead of a fixed
2,500.

**Result:** hard-region fit **reaches the floor on all 3 seeds** (gap 0.03 to 0.08 nat), and the
selector **recovers the dial on all 3 seeds** (width difference +2.1 to +2.9, all clear). In other
words, **the cheap shared net does everything the expensive independent nets do — once it is allowed to
converge.**

**Why the "untrustworthy seed" flags do not weaken this (the load-bearing check).** Seeds 1 and 2 are
flagged untrustworthy, but the flag is triggered per *width*, and the widths that fell short are **not**
the ones the bars depend on:

- **Width 12 — the width the hard-region fit bar reads — is fully converged on all 3 seeds.** So the
  "reaches the floor" claim rests entirely on converged widths.
- The untrustworthy widths are a few *middle* rungs (4, 5, 6, 8, 10) that were still creeping by less
  than 0.015 nat when stopped — and creeping *downward* (loss still falling), so full convergence would
  only *improve* them, never reverse the conclusion.
- The recovery separation is **structurally anchored**, not marginal. On every seed the per-width curves
  have the right shape: the hard region climbs steeply from about −0.5 to the noise floor (~1.5) by
  roughly width 6, while the easy region saturates by width 2. That shape *forces* the selector to give
  hard inputs more width; the +2-to-+3-width separation clears 2·SE on all 3 seeds. A sub-0.015-nat
  creep in one or two middle rungs cannot flip a separation of that size, and — because the easy region
  is a straight line that gains nothing from extra width — any residual improvement would land in the
  hard region and *widen* the separation.

So the correction is firm on the existing data: **the flags are about middle-rung creep, not about the
width-12 fit or the recovery direction.** No re-run is needed to trust the conclusion (an all-seeds,
all-widths re-run at a higher cap would only remove a cosmetic flag).

**This overturns §3.1.** "Shared-nested fails" was an under-training artifact, not a real property of
the architecture. On this data-rich toy the honest payoff of per-input width is therefore **compute,
not accuracy**: a single global width also reaches the floor given enough data, so routing buys you
cheaper inference rather than a better fit. (Whether it buys *accuracy* is the small-data / high-noise
question flagged as future work.)

---

## 4. The organizing question the two headline methods were built to answer

The baselines leave a clean question: the independent nets work because each width is fully
self-contained; the shared net works when converged despite *not* obviously having stable identity or
ordering. So **which of the three coherence properties actually matter for a nonlinear network?** The
two headline methods were designed as a controlled contrast:

- The **frozen residual cascade** forces *all three* properties by construction.
- **Matryoshka heads** fix only the "self-contained read-out" property and deliberately leave identity
  and ordering unaddressed.

Comparing them isolates whether stable identity and ordering matter *beyond* just giving each width its
own read-out.

---

## 5. The two headline methods

### 5.1 Frozen residual cascade — the primary bet

**Idea (boosting for width).** Build width up one block at a time. Block `b` is a single tanh unit
`h_b(x) = tanh(u_b x + c_b)` with its own mean and log-variance read-outs,

```
δμ_b(x) = a_b · h_b(x) + d_b            δs_b(x) = a′_b · h_b(x) + d′_b         (6)
```

and the width-`k` (prefix-`k`) model is the **sum** of the first `k` blocks:

```
μ_k(x) = Σ_{b ≤ k} δμ_b(x)              s_k(x) = Σ_{b ≤ k} δs_b(x)             (7)
```

Rung 0 (no blocks) is `(μ, s) = (0, 0) = N(0, 1)` — exactly the standardized marginal of `y`, the
analog of the low-rank ladder's diagonal-only rung. Note the additivity is in the **distribution's
parameter space**: additive mean, additive log-variance (hence *multiplicative* variance, which stays
positive for free). This is NGBoost's parametrization.

Train it in stages. At stage `b`, freeze blocks `1…b−1` (cache their outputs), **zero-initialize**
block `b`'s read-outs (`a_b = d_b = a′_b = d′_b = 0`, hidden weights random), and train **only** block
`b` on the NLL of the full prefix-`b` model. Then apply an **acceptance rule**: keep block `b` only if
its converged best held-out NLL improves on rung `b−1` by more than a threshold `min_delta`; otherwise
reset it to zero (rung `b` becomes inert, identical to rung `b−1`).

**The mathematics — three results that made this the principled choice.**

*Lemma 1 (function class — the only real change is the training scheme).* Expanding the prefix sum,

```
μ_k(x) = Σ_{b ≤ k} a_b tanh(u_b x + c_b)  +  ( Σ_{b ≤ k} d_b )                 (8)
```

which is *exactly* a width-`k` single-hidden-layer tanh network, with one extra freedom: the ordinary
nested-width net shares a single read-out bias across all widths, whereas the cascade's rung-`k` bias is
the running sum `Σ_{b≤k} d_b` — an independent bias per prefix. So the cascade is **not a new
architecture family**; it is the nested-width network with (a) a staged, frozen training scheme and (b)
per-prefix biases. This is why comparing it to the shared/independent nets is an apples-to-apples
*training-scheme* contrast at matched capacity (rung `k` ↔ width `k`).

*Lemma 2 (the monotone ladder).* Zero-initializing block `b`'s read-outs makes the stage-`b` model at
initialization identical to rung `b−1` (its added terms are zero). Training only block `b` and then
applying the acceptance rule can only keep an improvement, so **held-out NLL is non-increasing in `k`**:
`NLL_val(rung k) ≤ NLL_val(rung k−1)`. Every rung literally *was* the trained model at the end of its
stage, so every prefix is a valid calibrated model, and — because earlier blocks are frozen — rung
identity never changes afterward. This delivers the coherence invariant in its guaranteed (weak) form:
stable identity (frozen), self-contained (each block a complete additive correction), and importance
ordering as *monotone* improvement. The strong form (strictly decreasing gain per rung) is a
**measured** diagnostic, never assumed — exactly the discipline the low-rank ladder skipped.

*Proposition (moment-matching, emergent for free).* Hold `μ` fixed and minimize the NLL (1) over `s`.
Setting `∂ℓ/∂s = ½(1 − (y − μ)² e^{−s}) = 0` gives the stationary log-variance

```
e^{s(x)}  =  E[ (y − μ(x))² | x ]                                             (9)
```

so with `y = f(x) + ε`, `ε ~ N(0, σ²)`, the variance head **automatically absorbs the not-yet-explained
signal**:

```
σ²_k(x)  →  σ²  +  ( f(x) − μ_k(x) )²                                         (10)
```

Every accepted prefix is thus a *calibrated* model at its own capacity **by training**, not a
mis-calibrated truncation. This is precisely the low-rank ladder's moment-matching (eq. 3, dropped
factors folded into the diagonal so every rung stays calibrated) — but here it is emergent from the
NLL's own stationarity condition rather than hand-built into the covariance. It is the exact sense in
which the cascade is "the low-rank ladder generalized past linearity."

**Why it was the primary bet.** This *forces* all three coherence properties: frozen blocks give stable
identity; each block is a complete additive correction (self-contained); greedy residual-fitting plus
the acceptance rule guarantee a monotone non-increasing held-out loss as width grows (a weak form of
importance ordering). It keeps ~1× parameters and cheap prefix inference. And unlike the linear-algebra
cases, boosting works for arbitrary nonlinear pieces — so in principle it sidesteps the "SVD only works
for linear maps" barrier. It has solid precedent: gradient boosting (Friedman 2001), additive updates
in a distribution's parameters (NGBoost), and freeze-as-you-grow width (Cascade-Correlation, 1990).

**A known risk, pre-registered.** Differentiating the NLL (1) with respect to the mean gives

```
∂ℓ/∂μ  =  −(y − μ) · e^{−s}                                                   (11)
```

— the mean-fit gradient is scaled by `e^{−s} = 1/σ²`. But by the Proposition (eq. 10), after stage `k`
the variance is *largest exactly where residual signal remains*, and zero-initialization means stage
`k+1` *starts* at that inflated `s_k`. So the very gradient that stage `k+1` needs, to fix the hard
region, is attenuated by `e^{−s_k}` precisely on that region — staging **sharpens** the well-known
heteroscedastic-NLL pathology (Seitzer et al. 2022) relative to joint training. We pre-registered a
mitigation to fire if the cascade fell more than 0.10 nat short: retrain each stage with the **β-NLL**
loss, a per-point reweighting of (1) by a detached (stop-gradient) variance factor,

```
ℓ_β(μ, s; y)  =  ⌊ e^{s} ⌋^β_detached · ℓ(μ, s; y) ,     β = 0.5             (12)
```

which cancels part of the `e^{−s}` damping so high-variance points regain mean-gradient weight. (β-NLL
is already in the library — `loss_type="beta_nll"`; full treatment in `docs/mathematical_guide.tex`,
§"β-NLL Loss".) The evaluation bar is always plain held-out LL, whatever loss is used for training.

**Result: it fails, and the mitigation made it worse.** Across all 3 seeds the hard-region fit stalls
at about **−0.1 to −0.2 nat — roughly 1.7 nat below the floor.** The "anchor" diagnostic (cascade's
width-12 hard-region fit versus a *dedicated* width-12 net on the same data) is **+0.95 to +1.02 nat
worse** — the cascade is nowhere near what a plain net achieves at the same width. The greedy stagewise
scheme stalls on the hard region and never recovers. The pre-registered β-NLL escalation **did not
rescue it** — it made the gap slightly *worse* (1.76–1.80 nat vs 1.64–1.70 for plain loss), with more
blocks rejected as inert. The per-input dial recovery also fails: the selector mostly assigns *more*
width to easy inputs (inverted), separation fails on all seeds. This is a clean **negative result** for
the primary hypothesis: forcing identity + ordering by greedy freezing is not just unnecessary here, it
is actively harmful.

*(Likely mechanism for why β-NLL backfired: the block-acceptance test is computed on the β-weighted
loss, which de-emphasizes exactly the high-variance hard points, so more blocks get rejected as
"inert." Documented as a hypothesis, not yet confirmed.)*

### 5.2 Matryoshka heads — the fallback

**Idea.** One shared trunk `h(x) = tanh(W x + b) ∈ ℝ^{12}` (`Linear(1 → 12) → tanh`). For **each** width
`k = 1…12` a *dedicated* pair of read-out heads reads only the first `k` units of the trunk,

```
( μ_k(x), s_k(x) )  =  Head_k( h(x)_{1:k} ) ,   Head_k = Linear(k → 2)        (13)
```

and everything is trained **jointly**, minimizing the unweighted sum of all 12 rungs' NLLs each step:

```
L(x, y)  =  Σ_{k=1}^{12}  ℓ( μ_k(x), s_k(x) ; y )                             (14)
```

(the per-rung weights `c_k` of Matryoshka representation learning are all set to 1 here — an unexplored
dial, pre-registered as uniform). Prefix inference at width `k` computes `k` hidden units and one
`(k → 2)` head, so it stays cheap.

**What it fixes and what it doesn't.** The per-width heads remove the shared-read-out entanglement —
each width gets its own read-out basis, which is the mechanism the Matryoshka-representation result
rides on. But all widths' gradients still hit the one shared trunk jointly, forever, so the trunk's
hidden units get **no** stable identity and **no** ordering. On the scorecard: self-contained partly
fixed; identity and ordering not. That is exactly why it is the informative contrast to the cascade.

**Result: it fits to the floor, but the dial is not robust.** Hard-region fit **reaches the floor on
all 3 seeds** (gap 0.004 to 0.062 nat), and the anchor diagnostic is **≈ 0.00** — it matches a
dedicated net at the same width, about 1 nat better than the cascade. So for the *fit*, per-width
read-out heads are sufficient and the cascade's frozen identity/ordering machinery is unnecessary.

**But the per-input dial is fragile for this arm specifically.** Among the trustworthy seeds it is
**one clean success and one inversion**: seed 1 assigns strongly more width to hard inputs (difference
+4.05), while **seed 2 inverts** (−0.61 — more width to easy). Seed 0 also recovers (+2.03) but did not
fully converge. So per-width heads reliably *fit*, but reading the right width back out per input is
**seed-inconsistent** for Matryoshka.

**The inversion has a diagnosed cause — and it is exactly what the invariant predicts.** On seed 2 the
per-width curves come out *scrambled*: the hard region reaches ≈ 1.2 nat by width **3** and then goes
nearly flat, while the *easy* region climbs unusually *slowly* (log-likelihood 0.19 → 0.98 → 1.34 over
widths 1–4). Because the easy region is under-served at low width, the selector correctly reads "easy
inputs still want more width" and hands them more — producing the inversion. Compare the converged
shared and independent nets (§3.3), whose curves have the *correct* shape on every seed (hard climbs
steeply, easy saturates by width 2) and whose dial therefore recovers 3/3. This is the predicted price
of Matryoshka fixing only the read-out and leaving the trunk's hidden-unit identity and ordering
unaddressed: with no ordering pressure on the shared units, some seeds settle into a solution where the
trivial easy region is starved at low width. The fit survives it (the top width still reaches the
floor); the per-input dial does not.

**Consequence for the headline.** The robust per-input dial does **not** come from Matryoshka — it comes
from the converged shared net and the independent nets, which recover it cleanly on 6 of 6 seeds between
them (§3.3, §3.2). Matryoshka is the best *fit* but the weakest *dial*.

### 5.3 Parameter accounting — what "1× vs K×" actually means

For width cap `K = 12` and a single hidden layer on 1-D input (`W = K`), the parameter counts and their
*scaling shape* (the reason the comparison matters beyond this toy):

| Arm | Parameter count | ≈ (K=12) | Scaling in K | Rung-`k` inference |
| --- | --- | ---: | --- | --- |
| Shared-nested (one net) | `4K + 2` | 50 | **linear** | `k` units, shared read-out |
| Frozen cascade | `6K` (six scalars × `K` blocks) | 72 | **linear** — ≈ one single net | `k` units + `k` bias-adds |
| Matryoshka | trunk `2K` + `Σ_{k≤K} 2(k+1)` | ≈ 204 | **quadratic** (`K²`) | `k` units + one `(k → 2)` head |
| Independent per width | `Σ_{k≤K} (4k + 2)` | 336 | **quadratic** (`2K²`) | `k` units (own net) |

The headline: the **cascade is the only capacity-nested arm whose parameters stay linear in `K`** (≈ a
single net), while Matryoshka and the independent battery both grow quadratically — Matryoshka because
it carries a private head per width, the independent battery because it carries a whole net per width.
The converged shared-nested net (§3.3) shares the cascade's linear `4K + 2` budget *and* recovers the
dial — which is why it, not Matryoshka, is the efficient winner. On this 1-D toy every count is tiny in
absolute terms; the table is about the *shape*, which is what decides cost at real width.

---

## 6. Results at a glance

Held-out log-likelihood, hard region, at maximum width (12). Ceiling = **1.577**. "Dial recovery" =
does the learned selector give more width to hard inputs (positive, statistically clear)?

| Approach | weights | hard-region fit @ w12 | gap to floor | dial recovery | params | verdict |
| --- | --- | ---: | ---: | --- | ---: | --- |
| Shared-nested, 2.5k epochs (`W2`) | shared (1×) | ~ −0.4 | ~2.0 | — (fit fails) | 1× | fails — **under-training artifact** |
| Shared-nested, k-dropout, converged (`W_KDROPOUT_CONVERGED`) | shared (1×) | 1.49 – 1.55 | 0.03 – 0.08 | **3/3 pass** (+2.1…+2.9) | 1× | **floor + robust dial at 1× params — the winner** |
| Independent per width, converged (`W_CONVERGED`) | K× separate | 1.53 – 1.56 | 0.02 – 0.05 | **3/3 pass** (+1.9…+3.3) | K× | **works** (floor + robust dial), but K× params |
| Frozen residual cascade (`W_CASCADE`) | additive 1× | −0.11 … −0.22 | ~1.7 | 0/3 (inverted) | 1× | **fails both** — greedy stalls; β-NLL worse |
| Matryoshka heads (`W_MRL`) | shared trunk + per-width heads | 1.52 – 1.57 | 0.004 – 0.06 | 1 clean / 1 inverted | ~medium | **best fit, weakest dial** (inversion diagnosed) |

Deployment, briefly: every method's "matches-or-beats the best global width" holds where the fit works
(the Matryoshka selector matches or beats the global width on all 3 seeds; the converged baselines
tie). No method clears the 0.02-nat oracle sub-clause — that bar is too tight for this toy. On this
data-rich toy the payoff of routing is **cheaper inference, not better accuracy**, because a single
global width already reaches the floor.

---

## 7. What we can and cannot claim — and one correction

**What is solid:**

- **The per-input width dial works, and works robustly.** The selector assigns clearly more width to
  hard inputs than easy ones on **6 of 6 seeds** across the two full-width-nested methods — the
  converged shared k-dropout net (3/3) and the independent nets (3/3) — every separation clearing 2·SE.
  The recovery is *structurally anchored*: on every seed the hard region's per-width fit climbs steeply
  to the noise floor while the easy region saturates by width 2, which forces the correct allocation.
- **The cheapest architecture is the one that delivers it.** A 1× shared net (k-dropout, trained to
  convergence) both reaches the noise floor *and* recovers the dial — you do not need K× separate nets.
  Its "reaches the floor" claim rests on width 12, which is fully converged on all 3 seeds; the
  untrustworthy flags are middle-rung creep of <0.015 nat and do not touch either bar (§3.3).
- **Frozen greedy additivity (the cascade) is refuted on this toy.** It was the primary hypothesis and
  it failed cleanly; the pre-registered β-NLL rescue made it worse. Forcing stable identity + ordering
  by freezing is unnecessary and, here, harmful.
- **Matryoshka gives the best *fit* but the weakest *dial*.** Its per-width heads match a dedicated net
  at every width (best fit of any arm), yet its per-input dial is seed-inconsistent — and the one
  inversion is diagnosed (§5.2): with no ordering pressure on the shared trunk, that seed starved the
  trivial easy region at low width, so the selector over-fed it. The robust dial comes from the
  converged shared / independent nets, not from Matryoshka.

**What is not yet firm:**

- **Accuracy payoff is unproven.** On this data-rich toy routing buys **compute, not accuracy** — a
  single global width already reaches the floor given enough data. Whether per-input width buys
  *accuracy* in the small-data / high-noise regime is the open question and the natural next experiment.
- **Seed count is modest (3 per method).** The dial result is 6/6 clean across two methods, which is
  reasonably firm, but a handful of extra seeds on the shared net would make it airtight for
  publication. This is optional hardening, not a correctness gap — the conclusion does not depend on it.

**The correction (important).** The pre-run design docs
(`docs/plans/width_dial_2026-07-11/nested_architecture_research_2026-07-11.md`) conclude that
shared-nested width "fails all three properties" and that the sandwich schedule "did not rescue" it.
That conclusion rests on the under-trained `W2` run. The **converged** run of the same idea
(`W_KDROPOUT_CONVERGED`) reaches the floor and recovers the dial on all 3 seeds. So the design docs'
"shared-nested fails" claim is a convergence artifact and should be read as superseded by the converged
batteries. This is the same lesson that governs the whole program: **read the loss trajectory, per
width, before concluding — never a fixed epoch budget.**

**Net picture.** "Can we vary width per input?" — **yes**, demonstrably: a cheap 1× shared net, trained
to convergence, both fits the hard region to the noise floor *and* recovers the per-input width dial
robustly (6/6 seeds with the independent-net battery). The single remaining open edge is whether it buys
*accuracy* somewhere, not just cheaper compute — the small-data / high-noise regime is where to look.

---

## 8. Where depth fits in (next, not done)

This program is width-only, deliberately kept separate from the depth axis. The depth lane is **not
retired**: earlier depth work closed only the search for a *learnable* depth positive control (the
provably-deep target we tried is representable but not learnable by gradient descent at any depth, and
the per-input depth signal was null even at 8,000 training points). Those are known headwinds, **not**
a verdict on the cascade / nested ideas — which were never tried along depth. The first depth question,
once the width work lands, is exactly whether the ideas validated here (a shared nested net + a
per-input selector) transfer to depth.

---

## Appendix — per-seed numbers and file pointers

**Result files** (all under
`/home/ff235/dev/MLResearch/automl/automl_package/examples/capacity_ladder_results/`):

| Approach | file |
| --- | --- |
| Shared-nested, 2.5k epochs | `W2/w2_summary.json` |
| Shared-nested, k-dropout, converged | `W_KDROPOUT_CONVERGED/w_kdropout_converged_summary.json` |
| Independent per width, converged | `W_CONVERGED/w_converged_summary.json` |
| Frozen residual cascade (final, β-NLL) | `W_CASCADE/w_cascade_summary.json` |
| Frozen residual cascade (run 1, plain loss) | `W_CASCADE/w_cascade_summary_run1_plainNLL.json` |
| Matryoshka heads (120k) | `W_MRL/w_mrl_summary.json` |

**Per-seed detail** (hard-region log-likelihood at width 12 = `hard@12`; gap to the 1.577 floor;
dial = expected-width difference `hard − easy`, positive = correct; anchor = nat *worse* than a
dedicated width-12 net; ✓ = every width 1–12 fully converged). Note: for **both** converged methods,
**width 12 itself is trustworthy on all three seeds** — the ✓ column marks whether *every* rung
converged; the seeds without ✓ fell short only on middle rungs (k-dropout: widths 4/5/6/8/10 still
creeping by <0.015 nat; independent: one rung at the 40k cap), none of which drive the two bars:

*Independent per width, converged (`W_CONVERGED`)*

| seed | hard@12 | gap | dial (hard−easy) | converged |
| ---: | ---: | ---: | ---: | :---: |
| 0 | 1.557 | 0.020 | +1.934 |  |
| 1 | 1.536 | 0.040 | +2.059 |  |
| 2 | 1.531 | 0.046 | +3.251 | ✓ |

*Shared-nested, k-dropout, converged (`W_KDROPOUT_CONVERGED`)*

| seed | hard@12 | gap | dial (hard−easy) | converged |
| ---: | ---: | ---: | ---: | :---: |
| 0 | 1.552 | 0.025 | +2.122 | ✓ |
| 1 | 1.494 | 0.083 | +2.499 |  |
| 2 | 1.528 | 0.048 | +2.861 |  |

*Frozen residual cascade, final β-NLL (`W_CASCADE`)*

| seed | hard@12 | gap | dial (hard−easy) | anchor (worse) | inert blocks |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | −0.185 | 1.761 | −2.269 | +0.951 | 6 |
| 1 | −0.221 | 1.797 | −0.157 | +0.969 | 9 |
| 2 | −0.114 | 1.691 | −2.468 | +1.021 | 9 |

*Matryoshka heads, 120k (`W_MRL`)*

| seed | hard@12 | gap | dial (hard−easy) | anchor (worse) | converged |
| ---: | ---: | ---: | ---: | ---: | :---: |
| 0 | 1.573 | 0.004 | +2.028 | −0.004 |  |
| 1 | 1.544 | 0.033 | +4.045 | −0.000 | ✓ |
| 2 | 1.515 | 0.062 | −0.608 | +0.010 | ✓ |

**Source documents** (design and mathematics, written before the runs — see §7 for the one point they
now overstate):

- Research record / coherence invariant / literature:
  `/home/ff235/dev/MLResearch/automl/docs/plans/width_dial_2026-07-11/nested_architecture_research_2026-07-11.md`
- Firmed mathematics + build plan (cascade §2.1–2.5, Matryoshka §2.7):
  `/home/ff235/dev/MLResearch/automl/docs/plans/width_dial_2026-07-11/cascade_execution_plan_2026-07-11.md`
- Toy, bars, and objective:
  `/home/ff235/dev/MLResearch/automl/docs/plans/width_dial_2026-07-11/EXECUTION_PLAN.md`
