# ProbReg toy design under MSE-only, constant-σ scope — design spec

**Status: DRAFT for user review. ⛔ Do not build until signed off.** Written 2026-07-20 after the σ
scope change (`docs/probreg_benchmark/benchmark_spec.md` §2) invalidated the existing toys.

Owner: `docs/plans/capacity_programme/probreg.md`. Model definitions: that file §1.

---

## 1. Why the existing toys cannot be reused

The current toys draw an equal-weight two-component target
(`automl_package/examples/_toy_datasets.py:174-178`):

$$c \sim \mathrm{Bernoulli}(\tfrac12), \qquad y = \left(c - \tfrac12\right)\,s(x)\,\sigma \;+\; \sigma\,\varepsilon,
\qquad \varepsilon \sim \mathcal N(0,1)$$

so the two modes sit at $\pm\tfrac12 s(x)\sigma$ with equal weight. Therefore

$$\mathbb E[y \mid x] \;=\; \tfrac12\left(+\tfrac{s(x)\sigma}{2}\right) + \tfrac12\left(-\tfrac{s(x)\sigma}{2}\right) \;=\; 0
\qquad \textbf{for every } x .$$

**The conditional mean is identically zero by construction.** Under squared-error scoring the
Bayes-optimal predictor is the constant $0$, achieving $\mathrm{MSE}^\star = \operatorname{Var}(y\mid x)$,
and it is achieved by $k=1$, by $k=10$, and by a straight line alike. The toy has **zero power** to
detect any effect of $k$ on the mean. A benchmark run on it would return a clean-looking tie and be
read as "resolution does not help", when in fact nothing was measurable.

The variance-matched controls (`_toy_datasets.py:181-184`) set
$\sigma_{\text{broad}}(x) = \sigma\sqrt{1 + s(x)^2/4}$, i.e. they are **heteroscedastic by
construction** — also out of scope now.

⇒ Both families are retired for this benchmark. They remain valid for the distributional work when
σ-fitting is taken up as its own programme.

---

## 2. What the replacement must isolate

$k$ is a **resolution dial on the output representation**. ProbReg predicts

$$\hat f(x) \;=\; \sum_{c=1}^{k} p_c(x)\, \mu_c(x)$$

— a classifier $p(\cdot\mid x)$ over $k$ classes gating $k$ regression heads. For $k$ to be
measurable under squared error, **the conditional mean itself must be something a $k$-way
decomposition approximates better than a single head.** That is true when the target mean is
*quantised*: it lives on a small set of well-separated levels, and the difficulty is in deciding
*which* level, not in interpolating between them.

This is not a workaround for the σ restriction — it is the original "classification as a
regulariser" claim stated in a form squared error can see.

---

## 3. Generative model (positive family) — "level staircase"

Let $x \sim \mathrm{Unif}[0,1]^d$ ($d=1$ for the primary suite; $d>1$ is a stated extension).

**Zones.** Partition $[0,1]$ into $J$ contiguous zones $Z_1,\dots,Z_J$ of equal width. Zone $j$ is
assigned a **true resolution** $R_j \in \{1,2,\dots,R_{\max}\}$ — this is the per-input ground truth.

**Levels.** Zone $j$ carries $R_j$ levels, equally spaced with gap $\Delta$ and centred on zero:

$$\mathcal L_j \;=\; \Big\{\, \Delta\big(r - \tfrac{R_j+1}{2}\big) \;:\; r = 1,\dots,R_j \,\Big\}$$

**Regime assignment.** Within zone $j$, the level index is a deterministic, *balanced* function of a
within-zone coordinate $u(x) \in [0,1)$:

$$r(x) \;=\; 1 + \big\lfloor R_j \cdot \phi(u(x)) \big\rfloor$$

where $\phi:[0,1)\to[0,1)$ is a fixed **scrambling map** (§3.1). Balanced ⇒ each level in a zone has
probability exactly $1/R_j$.

**Target.**

$$y \;=\; \mu_{r(x)} \;+\; \sigma\,\varepsilon, \qquad \varepsilon\sim\mathcal N(0,1), \qquad
\boxed{\sigma \text{ constant — identical at every } x}$$

So $\mathbb E[y\mid x] = \mu_{r(x)}$, which is genuinely non-trivial and **piecewise constant with
$R_j$ distinct values in zone $j$** — exactly the structure a $k$-class bottleneck represents
natively and a smooth regressor must approximate with sharp transitions.

### 3.1 The scrambling map $\phi$ — and why it is required

If $r(x)$ were monotone in $x$, the mean would be a monotone staircase and a plain network could fit
it by brute force with enough capacity, making the comparison a capacity contest rather than a
representation one. $\phi$ makes the level assignment **fine-grained in $x$** (many thin alternating
strips) without changing the marginal level distribution.

Primary choice: $\phi(u) = \{m u\}$ (fractional part), $m$ an integer **strip multiplicity**. $m$ is
the *difficulty dial*: $m=1$ is a plain staircase, large $m$ is a rapidly alternating one.

**$m$ is a declared confound and is swept, not fixed** — see §7 C4.

### 3.2 Parameters

| Symbol | Meaning | Primary value |
|---|---|---|
| $J$ | zones | 3 |
| $(R_1,R_2,R_3)$ | true per-zone resolution | $(1, 2, 4)$ |
| $R_{\max}$ | ceiling offered to the model | 10 (≫ 4, so the ceiling never binds) |
| $\Delta$ | level gap | 1.0 (sets the scale) |
| $\sigma$ | noise, **constant** | swept, §5 |
| $m$ | strip multiplicity | swept $\{1, 4, 16\}$ |
| $n$ | samples | §6 |

$R_1 = 1$ is deliberate: **zone 1 is an embedded negative control.** Its mean is constant, so the
correct answer there is the bypass, and any per-input readout that assigns $k>1$ in zone 1 is
manufacturing resolution on data that has none — detectable within the same run, on the same model.

---

## 4. The analytic floor (this is what makes the toy a measurement, not a demo)

Because the levels are discrete, equiprobable and known, **the optimal achievable MSE at each $k$ is
computable in closed form** — no model needed.

A $k$-class model can emit at most $k$ distinct values per input neighbourhood. Restricted to zone
$j$, predicting with $k$ values is exactly optimal $k$-point quantisation of the uniform
distribution on $\mathcal L_j$. Writing $D_j(k)$ for that minimal distortion:

$$D_j(k) \;=\; \min_{\substack{\text{partition of } \mathcal L_j \\ \text{into } \le k \text{ groups}}}
\;\; \frac{1}{R_j}\sum_{\text{groups } G} \sum_{\mu \in G} \big(\mu - \bar\mu_G\big)^2 ,
\qquad \bar\mu_G = \tfrac{1}{|G|}\textstyle\sum_{\mu\in G}\mu$$

with $D_j(k) = 0$ for all $k \ge R_j$. For equally spaced levels this is a one-dimensional
$k$-means on a uniform grid and is solved exactly by dynamic programming in negligible time (the
driver computes it; it is never hand-tabulated into the plan).

**Oracle risk in zone $j$:**

$$\mathrm{MSE}_j^\star(k) \;=\; \sigma^2 \;+\; D_j(k)$$

This gives, before any training:
- the exact **knee location** ($k = R_j$, where $D_j$ first hits 0),
- the exact **penalty for under-resolving**, $D_j(k)$ for $k<R_j$,
- the **noise floor** $\sigma^2$ — so "did the model reach the floor" is a decidable question rather
  than a judgement call.

⚠️ **This is an independent anchor** in the sense `docs/plans/capacity_programme/probreg.md` §3.6
requires: it is derived from the generative construction, **not** computed by the model under test.
It is therefore the correct positive-control target and cannot launder a bug into a verified result.

---

## 5. The SNR sweep — the falsifiable prediction

The programme's standing claim is that $k$ is an **SNR/difficulty-adaptive** dial, not a count of
intrinsic components. This toy makes that claim testable, because the true $R_j$ is **fixed** while
$\sigma$ varies.

Define the separation in noise widths $s = \Delta/\sigma$. Two levels $\Delta$ apart are
statistically distinguishable from $n_\ell$ samples only when $\Delta \gg \sigma/\sqrt{n_\ell}$; more
sharply, the *selected* $k$ under the cheapest-within-tolerance rule (`benchmark_spec.md` §2.1)
should drop once the distortion saved by splitting a group, $\approx \Delta^2/4$, falls inside the
tolerance band of the MSE estimate, which scales as $\sigma^2\sqrt{2/n_{\text{cal}}}$.

**Pre-registered prediction:** with $R_j$ held fixed, the selected $k$ in zone $j$ is
**non-increasing in $\sigma$**, falling to 1 as $s \to 0$ — and the point where it falls is
predicted, not fitted, by the analytic floor of §4 combined with the tolerance rule.

Sweep $\sigma$ over $s = \Delta/\sigma \in \{8, 4, 2, 1, 0.5\}$.

**This is the discriminating experiment.** A readout that returns the true $R_j$ regardless of noise
is not measuring required resolution — it is counting components, which is the failure mode already
on record for this line of work. A readout that collapses to 1 even at $s=8$ is inert.

---

## 6. Starvation arithmetic (pre-registered, so a null cannot be blamed on it afterwards)

Zone $j$ receives $n/J$ points, split across $R_j$ levels ⇒ $n_\ell = n/(J R_j)$ per level; the
selection set receives a fraction $\rho$ of the training portion (`probreg.md` PB).

Two requirements:

1. **Level estimation.** The standard error of a level's mean is $\sigma/\sqrt{n_\ell}$. Requiring it
   below $\Delta/10$ gives $n_\ell \ge 100/s^2$. At the hardest swept point $s = 0.5$ this is
   $n_\ell \ge 400$, hence $n \ge J R_{\max}^{\text{used}} \cdot 400 = 3 \cdot 4 \cdot 400 = 4800$.
2. **Boundary learning.** With strip multiplicity $m$, zone $j$ contains $m R_j$ alternating strips,
   each of width $1/(J m R_j)$ in $x$ and holding $n/(J m R_j)$ points. Requiring $\ge 30$ points per
   strip at $m=16$, $R_j=4$: $n \ge 30 \cdot J m R_j = 30\cdot 3\cdot 16\cdot 4 = 5760$.

**Primary $n = 20{,}000$** (≈3.5× the binding requirement, leaving margin for the $\rho$ split and a
held-out test). **A run at any $n$ below 5,760 is not evaluable and must be reported NOT EVALUABLE,
never as a negative result.**

---

## 7. Confound ledger

| # | Confound | Control |
|---|---|---|
| **C1** | **Parameter count.** A $k$-class ProbReg has more parameters than $k=1$; a win could be capacity, not representation. | The plain-NN baseline is **parameter-matched to ProbReg at $R_{\max}$**, not to $k=1$. Report both the matched and the default NN. |
| **C2** | **Bypass availability.** Already cost this programme once: an arm that can select "plain regression" against one that cannot. | Every arm's rung set is $1..R_{\max}$ including the bypass (`benchmark_spec.md` §14.2 C3). |
| **C3** | **The toy is built for the mechanism.** A piecewise-constant mean is exactly what a classify-then-regress model represents natively. | **Stated as a limitation, not neutralised.** This suite establishes (a) that the mechanism works where it should and (b) that the *readout* is faithful — it is **not** evidence of general superiority. The real-data half carries that burden. Any report sentence implying otherwise is a defect. |
| **C4** | **Difficulty vs resolution.** $m$ changes how hard the boundary is; a $k$ effect could be an $m$ effect. | $m$ swept $\{1,4,16\}$ at fixed $R_j$. The claim is about $k$ only if the selected $k$ tracks $R_j$ across all $m$. |
| **C5** | **Zone leakage.** Neighbour-averaged per-input readouts smear across zone borders. | Report per-input readouts on zone **interiors** only (drop a margin of one strip width each side), and state the dropped fraction. |
| **C6** | **Ceiling.** $R_{\max}=10$ offered vs 4 used. | If any selection returns 10, the ceiling bound and the result is censored (`benchmark_spec.md` §3.2 ceiling check). |

---

## 8. Falsifiers — each kills a specific false-positive story

- **F1 — the plain regressor must actually struggle.** Fit a parameter-matched single-head network.
  If it reaches $\sigma^2$ (the noise floor) on held-out data, the bottleneck buys nothing here and
  **the toy is not discriminating** — report that and stop; do not proceed to the $k$ claims.
- **F2 — zone 1 must stay at $k=1$.** Its mean is constant. Any arm selecting $k>1$ there is
  manufacturing resolution; that invalidates the per-input readout regardless of what zones 2–3 do.
- **F3 — the smooth negative control.** A separate dataset, $\mathbb E[y|x] = \sin(2\pi x)$, same
  constant $\sigma$, no levels. Correct answer: $k=1$ everywhere. Reuses
  `make_toy_a` (`automl_package/examples/_toy_datasets.py:35`), which is already homoscedastic.
- **F4 — the model must reach the analytic floor at $k = R_j$.** If trained MSE at $k=R_j$ sits well
  above $\sigma^2 + D_j(R_j) = \sigma^2$, the arm is under-fit and **MASTER Decision 16 applies**:
  that is an optimisation finding, never an architecture verdict.

---

## 9. Pre-registered bars (numeric, frozen before any run)

Let $\hat k_j$ be the selected resolution in zone $j$ and $\mathrm{MSE}_j(k)$ the held-out risk.

- **B1 (recovery).** At $s \ge 4$: $\hat k_j = R_j$ for $j = 1,2,3$, on $\ge 2$ of 3 seeds.
- **B2 (floor).** At $s \ge 4$ and $k = R_j$: $\mathrm{MSE}_j \le \sigma^2 (1 + 0.10)$.
- **B3 (under-resolution penalty is real).** At $s \ge 4$: $\mathrm{MSE}_j(R_j - 1) - \mathrm{MSE}_j(R_j)$
  is within 25 % of the analytic $D_j(R_j-1)$ — i.e. the model actually pays the predicted price,
  confirming the mechanism rather than a coincidence of ordering.
- **B4 (SNR adaptivity).** $\hat k_j$ is non-increasing in $\sigma$ across the sweep, and
  $\hat k_j = 1$ for all $j$ at $s = 0.5$.
- **B5 (negative controls).** $\hat k = 1$ on F3's smooth data and in zone 1, at every $s$.

**Either outcome is publishable.** B1+B4 passing establishes an SNR-adaptive resolution dial with an
analytic reference. B1 passing but B4 failing means the readout counts components rather than
measuring required resolution — a substantive negative that redirects the programme.

---

## 10. Non-goals

No variance/σ fitting (that is the separate programme this scope change created). No multimodal
targets — the mean is single-valued by construction here. No real data. No new selection algorithm.
No change to the model definitions in `docs/plans/capacity_programme/probreg.md` §1.

---

## 11. Review verdict

*(To be completed. ⛔ USER GATE — this spec is delivered for sign-off before any implementation, per
the standing rule that toy designs are never improvised mid-run.)*
