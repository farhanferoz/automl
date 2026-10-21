# Flexible Architecture Scope Review — input document (2026-07-09)

> Provenance: scope document provided by Jordan on 2026-07-09 as input to the
> k-selector / flexible-architecture review & planning task. Preserved verbatim
> below. Task framing (from the same briefing):
>
> 1. **Problem**: three instances of the same failure mode in this repo — a
>    capacity parameter that cannot be learned in-sample because training loss
>    always rewards more capacity:
>    (a) ProbabilisticRegression n_classes k (fixed / global-learned / per-input),
>    (b) FlexibleNN per-input architecture-size selection,
>    (c) heteroscedastic variance estimation (NN + linear regression models that
>        jointly fit Gaussian mean+variance in-sample → variance collapses).
> 2. **Transferred idea**: nested rank ladder + post-training held-out arbiter
>    from ~/dev/turing/autocast-private (rank selection for the low-rank model).
>    For variance: MacKay evidence framework, the MILKA framework
>    (autocast-private), or an arbiter-family mechanism.
> 3. **Deliverables**: (i) critical review of the document below — no
>    rubber-stamping, focused web searches where unsure; (ii) comprehensive
>    execution-level plan for ProbReg (all three k regimes), FlexibleNN, and the
>    variance mechanism, each validated on toy problems; detailed enough for
>    autonomous orchestrator execution.
> 4. **Standing rulings** (REVIEW_HANDOVER_2026-07-03.md): bottleneck
>    architecture frozen; K_PENALTY-class terms banned; mixture-vs-blend open,
>    leaning mixture. Nested retraining touches the frozen-architecture ruling →
>    flag, don't assume.

---

# Flexible Architecture Scope Review

This document scopes a line of work for a flexible neural architecture project inspired by the rank-ladder discussion: learning an ordered model family once, then adapting the effective model size per input or per deployment setting. The central recommendation is to prioritize a **nested flexible architecture** rather than a fully conditioned or fully trained mixture architecture, because the nested approach preserves a coherent ordered prefix structure, supports clean post-hoc selection, and is much cheaper to train than evaluating all capacities for every sample.[1][2][3]

## Objective

The target problem is to turn a fixed-width feed-forward network into a flexible-width model whose active number of hidden units can vary by input. In the rank-ladder setting, this corresponded to learning a single ordered covariance ladder and then deciding how many columns to use; in the neural setting, the analogous goal is to learn a single ordered hidden representation and then decide how many hidden units should be active for a given input.[1][4][5]

A useful design principle from nested dropout is that smaller subnetworks should be prefixes of larger subnetworks, rather than unrelated masked variants. This creates an ordered family of subnetworks in which lower-capacity models remain valid truncations of higher-capacity ones, which is important for interpretability, adaptive computation, and post-hoc routing.[1][2]

## Conceptual mapping

The rank-ladder ideas transfer naturally to flexible-width feed-forward networks through the following correspondence.[1][3]

| Rank-ladder concept | Flexible architecture analogue | Why it matters |
|---|---|---|
| Rank $$r$$ | Active width $$r$$ | Capacity is indexed by a scalar budget.[3] |
| Loading prefix $$M_{1:r}$$ | First $$r$$ hidden units | Smaller models are prefixes of larger ones.[1][2] |
| Nested training | Prefix-width dropout | Learns an ordered basis of useful units.[1][5] |
| Held-out rank knee | Held-out width knee | Chooses smallest sufficient width out of sample. |
| Post-hoc mixture weights | Post-hoc width probabilities | Supports soft adaptive routing after training.[6][7] |

Let a one-hidden-layer MLP have hidden activations
$$
h(x)=\phi(Wx+b)\in\mathbb{R}^n.
$$
Then width $$r$$ means using only the first $$r$$ hidden units:
$$
f_r(x)=V_{:,1:r}h_{1:r}(x)+c.
$$
This is the direct analogue of the nested covariance ladder: width $$r+1$$ extends width $$r$$ by exactly one more unit.

## Three architecture options

### Nested flexible architecture

The nested option is the recommended baseline. One full hidden layer of size $$n$$ is learned, but each training example is assigned a sampled active width $$r_i$$, and only the first $$r_i$$ hidden units are used for that example. This is the feed-forward analogue of nested dropout, where smaller subnetworks form coherent prefixes of larger ones.[1][2][5]

In practice, a binary prefix mask is applied:
$$
m_{r_i}=(\underbrace{1,\dots,1}_{r_i},\underbrace{0,\dots,0}_{n-r_i}),
\qquad
\tilde h_i = m_{r_i}\odot h(x_i).
$$
The sample loss is then computed from $$\tilde h_i$$, and the batch loss is the mean over examples. Per-sample width sampling is important: it exposes many widths in one minibatch and encourages the hidden units to self-order by usefulness, with earlier units carrying more reusable signal.[1][4]

The main attraction of this design is that adjacent widths have a stable meaning: width $$r+1$$ is width $$r$$ plus one more feature. This makes post-hoc selection, distillation, and deployment logic much cleaner than if width were encoded through free learned gates.

### Conditioned flexible architecture

The conditioned alternative would treat the desired width or capacity level as an explicit input. A controller could map a width embedding into soft gates over all hidden units,
$$
g(r)=\sigma(W\,\mathrm{embed}(r)+b),
\qquad
\tilde h = g(r)\odot h(x).
$$
This is flexible, but it breaks the prefix guarantee: width $$r+1$$ need not equal width $$r$$ plus one more unit, because all gates may change simultaneously.

That makes the conditioned design mathematically valid but semantically weaker for this project. It no longer learns an ordered subnetwork family in the strict sense, so any post-hoc "how many units are needed?" interpretation becomes less stable. The rank-ladder discussion suggested that such a conditioned design can be kept as a control, but it is difficult to justify over nested when the project goal is an ordered flexible architecture rather than a generic gated family.

### Mixture flexible architecture

The mixture analogue would keep the same ordered width ladder, but instead of sampling one width per example, it would evaluate every width for every example and combine them with learned weights. For losses $$\ell_r(x_i,y_i)$$ at each width, the training objective becomes
$$
-\log \sum_{r=0}^{n} \pi_r \exp(-\ell_r(x_i,y_i)).
$$
This is a standard finite-mixture log-sum-exp objective.[6][7][8]

This option is conceptually useful because it defines a soft distribution over capacities and can model heterogeneity in required width. However, it is computationally expensive because all widths must be evaluated for every sample, and mixture training tends to enjoy in-sample headroom simply by combining many explanatory routes.[6][8][9] In the rank-ladder discussion, that combination of high cost and ambiguous selection signal made the fully trained mixture unattractive as the main operational method.

## Why nested should be the primary path

The nested route is the best match to the flexible-architecture objective for three reasons.

First, it imposes an ordered inductive bias: hidden unit 1 should be more universally useful than hidden unit 2, and so on. Nested dropout was developed precisely to learn such ordered representations and to make smaller subnetworks valid truncations of larger ones.[1][4][2]

Second, it supports clean post-hoc model selection. Because width $$r+1$$ differs from width $$r$$ by only one new unit, held-out incremental gains have a direct interpretation. This is the neural analogue of the rank-ladder knee rule.

Third, it is computationally efficient. Each example pays only for one sampled width during training, whereas the mixture version pays for all widths. Adaptive-width work broadly confirms that ordered or flexible-width subnetworks are attractive when a single trained network can serve multiple compute budgets.[3][10]

## Post-hoc width selection after nested training

A major advantage of the nested setup is that width selection can be separated from representation learning. After training the ordered network once, each candidate width can be evaluated on held-out data, and a width decision rule can be applied without retraining the base model.

### A3-style held-out knee

For each held-out example or trajectory, compute the loss at every width,
$$
\ell_r(i)=\ell(f_r(x_i),y_i).
$$
Then define cumulative gain over width 0 and marginal improvement from one extra unit. The neural analogue of the rank-ladder arbiter is to select the smallest width whose next-unit improvement is no longer distinguishable from noise under a bootstrap or repeated-split uncertainty estimate. This yields one global width $$r^*$$ for deployment.

This approach is attractive when the project wants a single default compact architecture rather than per-input adaptation. It is also easy to explain operationally: keep adding units only while each added unit still produces a statistically meaningful held-out gain.

### A5-style post-hoc stacking

If the project wants a soft distribution over widths rather than one selected width, nested training can still support that post hoc. Freeze the trained network, compute all held-out losses $$\ell_r(i)$$, and fit width weights by maximizing
$$
\sum_i \log \sum_r \pi_r \exp(-\ell_r(i)).
$$
This is the same mixture objective as above, but it is optimized only after the ordered network has already been trained.[6][7]

The EM updates are standard:
$$
q_i(r)=\frac{\pi_r\exp(-\ell_r(i))}{\sum_{r'}\pi_{r'}\exp(-\ell_{r'}(i))},
\qquad
\pi_r \leftarrow \frac{1}{F}\sum_i q_i(r).
$$
These $$q_i(r)$$ act as posterior responsibilities over widths. This is appealing because it recovers most of the mixture interpretability without paying the full training cost of a fully trained mixture model.[6][7][9]

## Learning per-input width through the nested setup

The most promising extension is to learn a per-input width predictor on top of the nested model, rather than replacing the nested model itself. The base ordered network is trained once; then a router is trained to predict how much of that ordered hidden layer should be activated for a given input.[11][12]

### Two-stage hard routing

A practical two-stage method is:

1. Train the nested-width network.
2. On held-out data, evaluate every width and assign each example or trajectory an oracle width.
3. Train a classifier or router $$h_\phi(x)\to r$$ on those oracle labels.
4. At inference, use the predicted width $$r(x)$$.

The oracle width can be defined as the best-loss width,
$$
r_i^{\mathrm{oracle}} = \arg\min_r \ell_r(i),
$$
or as the smallest width within a tolerance of the minimum, which often gives a more conservative compute-aware policy.

### Two-stage soft routing

A smoother variant is to construct soft target distributions over widths using the post-hoc responsibilities,
$$
q_i(r) \propto \pi_r \exp(-\ell_r(i)),
$$
and train a router $$q_\phi(r\mid x_i)$$ to match those targets, for example using KL divergence. This avoids forcing a brittle hard label when several nearby widths perform similarly.

### Prefix-preserving stopping rule

The most nested-faithful parameterization is not arbitrary gating of all units, but a stopping-rule model. The router outputs continue probabilities $$c_k(x)$$ for successive units, and the implied probability of width $$r$$ is
$$
q(r\mid x)=\left(\prod_{k=1}^{r} c_k(x)\right)(1-c_{r+1}(x)).
$$
This preserves the prefix interpretation exactly: unit $$k+1$$ can only be active if all previous units are active. It therefore inherits the main conceptual virtue of the nested setup while allowing input-adaptive compute.[11][12]

## What this project should not conflate

Several related ideas should be kept conceptually separate during scoping.

- **Ordered representation learning**: learn hidden units so that prefixes are meaningful.[1][2]
- **Global width selection**: choose one width for all inputs using held-out evaluation.
- **Post-hoc soft width weighting**: fit a global mixture over widths after training.[6][7]
- **Per-input routing**: predict width as a function of the input, using labels or soft targets derived from held-out scoring.[11][12]

Treating these as separate modules will make the project easier to scope, benchmark, and ablate.

## Suggested experimental plan

### Phase 1: establish the ordered-width baseline

Train a one-hidden-layer or multi-layer MLP with prefix-width masking. Compare against a standard fixed-width baseline at several widths and verify that the first $$r$$ units form a useful ordered family. The main question in this phase is whether a single nested model can recover competitive performance across a range of widths.[1][3]

### Phase 2: global post-hoc width selection

Evaluate all widths on held-out data and implement two readers:

- a knee-based smallest-sufficient-width rule;
- a post-hoc EM stacking rule over widths.

This phase will show whether the ordered model already provides a strong compute-performance frontier without any per-input routing.

### Phase 3: per-input router

Train a hard or soft router from input features to width. Compare:

- global fixed width chosen by the knee rule;
- global soft width mixture from post-hoc stacking;
- per-input hard routing;
- per-input soft routing with expected width control.

Metrics should include predictive quality, expected active width, latency, and calibration of the routing decisions. Adaptive-width models are especially compelling when they dominate fixed-width baselines on the quality-versus-compute tradeoff.[3][10]

## Key technical risks

| Risk | Why it matters | Mitigation |
|---|---|---|
| Weak self-ordering of hidden units | Early units may not become clearly more useful than later units. | Use strong prefix masking during training; inspect width-wise performance curves.[1][2] |
| Too much flexibility in routing | Free gates may destroy nested semantics. | Prefer stopping-rule or prefix-preserving routers over unconstrained gating. |
| Label noise in oracle widths | Best width may be unstable for individual examples. | Use soft targets or trajectory-level aggregation instead of brittle point labels. |
| Overhead from evaluating all widths post hoc | Per-input oracle generation can be costly. | Restrict evaluation to a candidate ladder of widths or use coarse-to-fine screening. |
| Mixture overfitting | Fully trained mixtures gain in-sample fit too easily. | Prefer post-hoc stacking on held-out data over end-to-end global mixture training.[6][8][9] |

## Recommended scope for the review agent

The review should answer the following questions.

1. Does nested-width training reliably produce an ordered family of subnetworks for the target problem class?
2. Is post-hoc global width selection sufficient, or is there clear evidence that per-input routing materially improves the quality-versus-compute frontier?
3. Which routing targets are most stable: hard oracle widths, tolerance-based widths, or soft posterior responsibilities?
4. Does a prefix-preserving stopping-rule router outperform unconstrained gating in robustness and interpretability?
5. How much of the value of flexible width comes from adaptive compute alone, versus genuine specialization to different input regimes?

The immediate recommendation is to scope the work around a **nested ordered-width architecture + post-hoc readers + optional amortized router**. That plan captures the main technical ideas discussed so far, keeps the inductive bias coherent, and avoids the expense and ambiguity of a fully trained mixture-first approach.[1][2][11][12]
