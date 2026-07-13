# Literature verification: frozen residual cascade for probabilistic regression

## ITEM 1 — NGBoost

**Claim:** NGBoost performs stagewise additive updates in the distribution's parameter space — for a
Normal output, each stage adds an increment to mu and log-sigma (scaled by a learning rate), fitting
the natural gradient of a proper scoring rule (e.g. NLL). Any prefix of the boosting stages already
defines a valid Gaussian predictive distribution.

**Verdict: CONFIRMED** (mechanism), **prefix-validity: structural byproduct, not discussed by the paper**.

**Citation:** Duan, T., Avati, A., Ding, D. Y., Thai, K. K., Basu, S., Ng, A. Y., & Schuler, A. (2020).
NGBoost: Natural Gradient Boosting for Probabilistic Prediction. *Proceedings of the 37th International
Conference on Machine Learning (ICML 2020)*, PMLR 119. arXiv:1910.03225 (v1: Oct 8 2019; v4: Jun 9 2020).

**Quotes (Algorithm 1 and surrounding text, arXiv:1910.03225v4):**
- Additive parameter-space update, explicit formula:
  > "y|x ∼ P_θ(x), θ = θ^(0) − η Σ_{m=1}^{M} ρ^(m) · f^(m)(x)."
  and: "A prediction y|x on a new input x is made in the form of a conditional distribution P_θ, whose
  parameters θ are obtained by an additive combination of M base learner outputs (corresponding to the
  M gradient boosting stages) and an initial θ^(0)."
- Per-parameter base learners, Normal example:
  > "For example, when using the Normal distribution in our experiments, θ = (μ, log σ) ... for a Normal
  distribution with parameters μ and log σ, there will be two base learners, f_μ^(m) and f_{log σ}^(m)
  per stage, collectively denoted as f^(m) = (f_μ^(m), f_{log σ}^(m))."
- Natural gradient of a proper scoring rule (Algorithm 1, inner loop):
  > "g_i^(m) ← I_S(θ_i^(m-1))^{-1} ∇_θ S(θ_i^(m-1), y_i)" — S is any proper scoring rule (MLE/NLL or CRPS
  are the two worked examples).
- Learning-rate-scaled additive per-example update (Algorithm 1):
  > "θ_i^(m) ← θ_i^(m-1) − η(ρ^(m) · f^(m)(x_i))."

**On prefix-validity:** the paper never frames "stopping after M' < M stages" as a deliberate,
discussed feature. It is a structural consequence of the fact that θ^(m) is a complete, well-formed
parameter vector after *every* iteration (Algorithm 1's inner loop updates all n examples' full θ_i at
each m), not something Duan et al. call out or evaluate (no experiment on partial ensembles / anytime
use was found in the Introduction, Sections 3.1–3.5, or the start of Section 4 — the parts read).
Confidence: **high** for the mechanism (full Algorithm 1 + Sections 1–4 opening read verbatim);
**high** for the negative claim about prefix-validity discussion (nothing found in the sections read,
consistent with the paper's framing as a fixed-M batch training method, not an anytime method).

---

## ITEM 2 — Cascade-Correlation

**Claim:** Cascade-Correlation (Fahlman & Lebiere, NIPS 1990) grows a network by adding hidden units one
at a time; each new unit is trained to maximize correlation with the current residual error and then
its INPUT weights are frozen forever; only downstream/output weights keep adapting.

**Verdict: CONFIRMED, precisely as stated.**

**Citation:** Fahlman, S. E., & Lebiere, C. (1990). The Cascade-Correlation Learning Architecture. In
D. S. Touretzky (Ed.), *Advances in Neural Information Processing Systems 2* (NIPS 1989 conference;
proceedings published 1990 by Morgan Kaufmann), pp. 524–532.
(proceedings.neurips.cc indexes it under the 1989 conference year; the paper itself and most citations
give the 1990 publication/proceedings year — both are correct depending on which date is meant.)

**Quotes (verbatim, from the NeurIPS proceedings PDF, pp. 524–526):**
- Abstract: "Once a new hidden unit has been added to the network, its input-side weights are frozen.
  This unit then becomes a permanent feature-detector in the network."
- Figure 1 caption: "Boxed connections are frozen, X connections are trained repeatedly."
- Body text (p. 525): "Cascade-Correlation combines two key ideas: The first is the cascade
  architecture, in which hidden units are added to the network one at a time and do not change after
  they have been added. The second is the learning algorithm, which creates and installs the new hidden
  units. For each new hidden unit, we attempt to maximize the magnitude of the correlation between the
  new unit's output and the residual error signal we are trying to eliminate."
- Precise freeze scope (p. 525): "Each new hidden unit receives a connection from each of the network's
  original inputs and also from every pre-existing hidden unit. The hidden unit's input weights are
  frozen at the time the unit is added to the net; only the output connections are trained repeatedly."
- Training objective, formalized (p. 526): candidate unit's incoming weights are adjusted "to maximize
  S, the sum over all output units o of the magnitude of the correlation (or, more precisely, the
  covariance) between V, the candidate unit's value, and E_o, the residual output error observed at
  unit o," with S = Σ_o |Σ_p (V_p − V̄)(E_{p,o} − Ē_o)|.

So: exactly what's frozen is the new unit's *input-side* (incoming) weights, permanently, at
installation time; everything already frozen stays frozen (units "do not change after they have been
added"); only the (growing) set of output-layer weights keeps being retrained after each addition.
Confidence: **high** — read directly from the original paper's PDF (pp. 524–526), not a secondary
source.

---

## ITEM 3 — β-NLL / heteroscedastic NLL pathology (Seitzer et al. 2022)

**Claim:** In joint Gaussian NLL training the gradient of the mean is scaled by 1/σ², so
high-predicted-variance / poorly-fit regions get an attenuated mean-fit gradient, causing under-fitting;
β-NLL multiplies the per-point NLL by σ^(2β) (β=0.5 recommended) to counteract this.

**Verdict: CONFIRMED, matches the paper closely** (one nuance: the paper frames it as "high-error
points get down-weighted", which for a locally-calibrated model is the same set of points as
"high-variance", but the causal chain in the paper runs error→variance-shrinks-to-match→gradient
attenuates, not "the model's variance is high so the gradient is attenuated" as a raw input condition
— see quote below).

**Citation:** Seitzer, M., Tavakoli, A., Antić, D., & Martius, G. (2022). On the Pitfalls of
Heteroscedastic Uncertainty Estimation with Probabilistic Neural Networks. *ICLR 2022*.
arXiv:2203.09168 (v1: Mar 17 2022; v2 shown: Apr 1 2022).

**Quotes (verbatim, from the PDF, pp. 2–6):**
- Gradient of the NLL loss w.r.t. the mean (Eq. 3):
  > "∇_μ̂ L_NLL(θ) = E_{X,Y}[(μ̂(X) − Y) / σ̂²(X)]"
- The 1/σ² attenuation mechanism, stated explicitly (Sec. 3.2, p. 4):
  > "the gradient ∇_μ̂ L_NLL of the NLL w.r.t. the mean scales the error μ̂(X) − Y by 1/σ̂²(X) (Eq. 3).
  As symmetry is broken and the true function starts to be fit locally, the variance quickly shrinks in
  these areas to match the reduced MSE... Data points with already low error will get their contribution
  in the batch gradient scaled up relatively to high error data points — 'rich get richer'
  self-amplification."
- Summary-of-contributions framing (p. 2): "the issue arises due to the NLL loss scaling down the
  gradient of poorly-predicted data points relative to the well-predicted ones, leading to effectively
  undersampling the poorly-predicted data points."
- β-NLL definition, exact (Eq. 7, p. 5):
  > "L_{β-NLL} := E_{X,Y}[⌊σ̂^{2β}(X)⌋ (½ log σ̂²(X) + (Y − μ̂(X))² / (2σ̂²(X)) + const)]"
  where "⌊·⌋ denotes the stop-gradient operation."
- Recommended β (p. 5): "we find that β = 0.5 generally achieves the best trade-off between accuracy
  and log-likelihood." (β=0 recovers plain NLL; β=1 recovers MSE-for-the-mean gradient.)

Confidence: **high** — full text read (Sections 1–4, pp. 1–6), all quotes verbatim from the PDF.

---

## ITEM 4 — "Anytime" / prefix-valid / early-exit boosting as an explicitly calibrated probabilistic model

**10-minute search, result: not found this pass** (for the specific combination — every prefix of an
ensemble explicitly treated as a deployable, *calibrated probabilistic* model).

What *does* exist and is the closest primary source: **SpeedBoost** (Grubb & Bagnell, AISTATS 2012,
PMLR v22) — "SpeedBoost: Anytime Prediction with Uniform Near-Optimality." It is a genuine anytime
boosting method: functional-gradient boosting that produces a single predictor usable at any
computation/stage budget, with a per-stage optimality guarantee ("provably competitive with any
possible sequence of weak predictors with the same total complexity"). However, this is for **point
prediction**, not probabilistic/distributional output — no calibration claim for partial ensembles as
predictive distributions was found.

Also surfaced but not on-point: "Obtaining calibrated probabilities from boosting" (Niculescu-Mizil &
Caruana) — post-hoc calibration of a *finished* boosted classifier's probability outputs, not
about prefixes/partial ensembles being calibrated. "Calibrated Boosting-Forest" (Wu, arXiv:1710.05476)
— same style, full-ensemble calibration, not anytime/prefix.

No paper was found that explicitly discusses/verifies that intermediate prefixes of a probabilistic
(distributional) boosting ensemble such as NGBoost are themselves calibrated or evaluates them as
deployable partial models — absence, not a weak-match stretch.
Confidence: **medium** (targeted search only, ~10 min budget as instructed; not exhaustive).
