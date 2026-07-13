# Step-1 findings — variational-EM k-selector (Basis A grounding)

**Validated.** The core idea is confirmed and the one check that was failing — the shape
test — is resolved once it is read off the right quantity. The headline readouts are the
**bypass/engagement weight** (does the bottleneck engage with resolvable structure?) and the
**held-out mixture NLL** (does scoring the full shape matter?), both of which the design note
§9 already designates. The integer **surviving-`k` count over-counts** (bin tiling, exactly as
note §10 anticipated) and is kept only as a caveated resolution diagnostic, not a mode count.
Numbers in `results.md` / `results.json` (k_max=6, K=7 incl bypass, n=800, σ=0.3, 500 epochs,
5 seeds, constant baseline, bypass on).

## What works — SNR-adaptive engagement (Check 1 headline)

The cleanest, most robust signal is the **bypass weight**, and it behaves as designed. As the
two modes go from merged to resolved (spacing 0.5 → 6 noise-widths), weight hands over from the
unconstrained bypass to the discretisation bins:

    spacing  0.5    1.0    1.5    2.0    3.0    4.0    6.0
    bypass_w 0.888  0.870  0.886  0.627  0.334  0.234  0.198

With only `α₀ = 0.1` and **no hand-tuned penalty**, the model uses the simple direct regressor
when there is no resolvable structure (bypass ≈ 0.88 while the modes overlap) and engages the
classes once the modes stand apart (bypass falls to ≈ 0.2). That "use the bottleneck only when
the data support it" behaviour — the whole point of the redesign — is validated. The integer
surviving-`k` count over the same sweep is noisy (2, 2, 1, 3, 4, 4, 3) and is *not* the headline;
see below for why.

**Check 2 passes cleanly.** At a resolved spacing the surviving count is stable across
`α₀ ∈ {0.05, 0.1, 0.2, 0.5, 1/K}` (median 4 throughout). The count does **not** track `α₀`, so
`α₀` is a weak setting, not the old hand-set knob in disguise.

**Negative control** (smooth unimodal, x informative): the count stays at 1 at low noise
(σ ≤ 0.1, bypass carries the plurality) and drifts only to 2 at higher noise — a mild version
of the over-counting below. The method does not invent classes on featureless data.

## Check 3 resolved — judge by held-out NLL, not by the count

The shape pair is bimodal vs. a single broad Gaussian **matched in mean and variance**
(both empirically (≈0.0, 0.44)). A moment/summary objective sees them as identical; the
question is whether the genuine blend-of-probabilities can tell them apart. It can — on
**held-out** data (a fresh draw from the same generators), comparing the genuine mixture
against a single Gaussian fit by moments (the summary model):

    held-out NLL        mixture   single Gaussian   mixture edge
    bimodal (2 peaks)    0.921         1.018          +0.096   (structure real → mixture helps)
    broad   (1 bell)     1.072         1.008          -0.064   (no structure → tiling overfits)

The mixture **beats** the summary model on the two-peaked target and is **worse** than it on
the broad bell — a clean separation (+0.096 vs −0.064 nats), and exactly the §2/§3 claim made
operational: the full-shape likelihood rewards genuine peaks and is penalised, out of sample,
for splitting a single bell it cannot improve on. This is the principled arbiter; it works
where the integer count never could. (Bypass-only NLL is huge — 6.7 / 22.7 — because the
bypass head is abandoned once the mixture engages, so the moment-matched single Gaussian, not
the bypass, is the correct summary baseline.)

## The over-counting — diagnosed, and why it is not a count to "fix"

For 2 true modes the surviving-`k` count reaches ~4 at resolved spacings, and the broad bell
also reads ~4. The cause is structural, not a bug or under-training:

- **The weight prior empties *unused* classes; it cannot merge *populated* tiles.** The pruning
  force is the digamma effective weight `exp(ψ(γ_c))`. A near-empty class sits at `γ ≈ α₀ < 1`
  where `exp(ψ(α₀))` is tiny — it dies. But a class carrying real data has `exp(ψ(γ)) ≈ γ − ½`
  — essentially its full count, no suppression. When two adjacent bins each tile half of one
  mode, both hold large counts (~200 of 800), so neither is touched. The mechanism prunes
  empty classes, never well-fed tiles.
- **There is no Occam charge on the component locations/widths.** The note (§4) deliberately
  puts a prior only on the weights `w` and point-estimates each class's `(μ_c, σ_c)` by
  gradient. Maximum likelihood on those parameters always prefers narrower components that tile
  a mode (finer fits the training sample strictly better) — the §2 model-selection trap
  recurring one level down. The weight prior fixed the "too many *empty* classes" version of
  the trap; it does nothing about the "tile one mode with several *full* narrow classes"
  version.

So the integer count is structurally a **resolution measure**, not a component count — which is
exactly how we already frame `k` (an SNR-adaptive resolution dial). The reason the count says
~4 while the held-out NLL still separates the shapes is that tiling is a *training-set*
artifact: the trained weights record it, but the held-out evidence reveals it as overfitting
on the broad case and as genuine on the bimodal case. Hence: judge by held-out NLL, report the
count only as a diagnostic.

## Resolution of the fork (Basis A grounding is complete)

The fork in the earlier write-up is resolved as **option 1, the principled half**:

- **Adopt the readouts the note §9 already specifies:** bypass/engagement weight for Check 1,
  held-out mixture NLL for Check 3. No tuning, no new loss term, no model change.
- **Demote the integer count** to a caveated diagnostic (reported with `effective_number =
  exp(H)` as a continuous cross-check).
- **Basis A's job is done.** Its only purpose was to confirm the pruning machinery prunes at
  all — the bypass handover (Check 1) and the smooth negative control prove it does. The
  populated-tile over-counting is the §10 caveat made real; it does not invalidate the
  grounding under the engagement readout. Proceed to **Basis B** (the per-input contribution),
  carrying the held-out-NLL readout.

Rejected alternatives: **fewer k_max** re-introduces the hand-set resolution knob the redesign
exists to kill; a **repulsion / merge-pressure loss** is an arbitrary loss component (out by the
durable constraint); a **merge-bins-by-mean readout** is a heuristic, acceptable only as a
secondary "distinct predictive modes" diagnostic. If a *true integer component count* is ever a
goal, the only principled route is putting a prior on `(μ_c, σ_c)` and integrating them too — a
real model change, out of scope now.
