# NOTE-MOE cold-read gate log

**Round 1 (2026-07-10, fresh cold reader on the 10pp PDF): findings triaged, ALL blockers fixed, rebuilt.**
- s_i symbol collision (App B skill vs §3 held-out score) → App B renamed to ability a_i with explicit
  disambiguation.
- Figure 1 contradicted "rails to 0/1" (dynamics plateaued at 0.27) → added staleness decay ρ=0.02 to the
  two-expert model (idle expert's parameters go stale — faithful to the quoted mechanism); simulation now
  genuinely rails; caption + App B updated; parameter values (η, ρ, β) stated for reproducibility.
- "capacity" two senses (model-size vs hardware token budget) → explicit word-shift warning in §2.3.
- "bias" collision (§4 logit offset vs §6 estimator sense) → §6 disambiguated ("systematic error ...
  not the logit offset of Section 4").
- "coherence check" undefined in §6 → defined inline (held-out gap nested vs dedicated same-size model);
  naive-rule-vs-stacking relationship stated (any reader of nested scores inherits the distortion).
- Dual-ascent jargon → plain price-adjustment gloss added; "dual step size" → "price step size".
- Half-nat-per-parameter (Figure 2's slope) → degree-of-freedom accounting spelled out in §2.4 text +
  caption; x-axis units (fitted parameters) stated.
- Table 1 unparseable row → "The balance goal itself, stated probabilistically".
- e^{s_i(c)} substitution made explicit in §3.2; "average" → "total" log score (App A); EM monotone-ascent
  justified via Jensen; N defined; x/y defined at Eq (1); "importance values" glossed; "charges" defined
  (§1); "rung" defined (§3.1); Dirichlet choice motivated.
- Under-shown companion evidence → headline numbers inlined (router 0.856/0.885 etc., 7/9 soft-vs-hard,
  controls ≤0.016 vs 0.16–0.20) + companion report named (*Reading Model Capacity From Held-Out Data*,
  2026); §3.3-vs-§6 router/structure tension reconciled explicitly.
- Reader confirmed CLEAN: §§1, 2.1–2.3, remedies list (all quotes attributed w/ coefficients), 3.1,
  Appendix A (Eq 4 properness + concavity), references all resolve.
