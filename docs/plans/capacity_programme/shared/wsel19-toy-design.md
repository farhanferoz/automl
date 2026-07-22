# WSEL-19 multi-feature toy design — FOR USER GO (2026-07-22)

Written per the standing toy gate: no toy is built without a reviewed design spec. This document
is what the user GOes; the 1-D slice of WSEL-19 uses only the canonical §3.8 toys and does not
wait on it. Owner: `docs/plans/capacity_programme/width.md` WSEL-19.

## 1. What these toys must decide

The bake-off (WSEL-19) compares four router backends across input dimensionality and
selection-set size. The multi-feature toys must (a) present the router with inputs of dimension
d > 1 while *provably preserving* the per-input routing signal the 1-D canonical toy was
certified on, and (b) contain the overfitting regime that motivated the regularisation ruling
(WSEL-7 ruling 3).

## 2. Generative construction — the 1-D canonical toy behind a projection

**Inputs:** `x ~ N(0, I_d)`, standardized coordinates.

**Projection:** a unit vector `v ∈ R^d` defines `t_raw = v·x`. Because the design is spherical
Gaussian, `t_raw ~ N(0, 1)` for EVERY unit `v` — axis-aligned and oblique variants have
*identical* projection marginals by construction, so geometry is the only thing that differs
between them (confound C1, §6).

**Target:** `t = A(Φ(t_raw))` where `Φ` is the standard normal CDF and `A` the affine map onto
the canonical 1-D toy's input interval; then `y = h(t) + ε`, `ε ~ N(0, σ(t)²)`, where `h` and
the noise profile `σ(·)` are EXACTLY the canonical hetero construction — the implementer reads
both from the canonical toy's source at build time and re-uses that code (§3.9 reuse rule);
this spec deliberately restates neither. The Φ-map gives `t` the same marginal the 1-D toy
trains on, for every `d` and every `v`.

**Consequence:** the easy-region / width-hungry-region structure — the thing that makes the
oracle best-width vary per input, certified in 1-D — is inherited unchanged. `d = 1` with
`v = (1)` recovers the canonical toy exactly (identity check, §5).

## 3. Grid axes

| axis | values | why |
|---|---|---|
| input dimension `d` | 2, 8, 32 | spans the rule-fitting range; 32 is the starvation regime |
| geometry of `v` | AXIS (`v = e_1`) · OBLIQUE (dense, drawn once per seed, unit norm) | AXIS is tree-friendly (one informative coordinate, d−1 nuisance); OBLIQUE is tree-hostile (every coordinate partially informative). Both at every d. |
| selection-set size `N_sel` | 75, 300, 1200 | 75 = the observed overfitting regime (WSEL-6 re-run evidence); 1200 = unstarved contrast |
| seeds | 0, 1, 2 | canonical |

Sparse-oblique (`v` on 2 of d coordinates) is EXCLUDED for compute — AXIS already supplies the
nuisance-dimensions stress at `d−1` nuisance coordinates; recorded here so its absence is a
decision, not an oversight.

## 4. Starvation arithmetic (why 75 points is the stress cell)

The frozen router at `d = 32` (hidden `(32, 32)`, 12-way output): `32·32+32 + 32·32+32 +
32·12+12 = 2508` parameters against 75 selection points — 33 parameters per label. At `d = 2`:
`1548` parameters — 21 per label. Both starved cells sit far past interpolation capacity;
`N_sel = 1200` gives ~2 parameters per label at `d = 32`. The bake-off's regularisation
question *requires* the starved cells; the unstarved cells are the control.

## 5. Pre-registered validity checks (per (d, geometry, seed) cell, BEFORE any backend verdict)

1. **Identity check:** `d = 1` construction reproduces the canonical toy's `(t, y)` samples
   bit-for-bit on a shared seed.
2. **Hidden-ness falsifier (routing signal exists):** the oracle per-input width (error-table
   row minimum under the standing 0.25 tie-band) must beat the best single fixed width by ≥ 10%
   of the latter's held-out routed error — the strand's decision bar reused as the signal floor.
   A cell that fails is reported VOID-FOR-ROUTING and excluded from backend verdicts (never
   silently included). *(10% is a chosen default, flagged for review with everything else here.)*
3. **Regime visibility:** the per-width error table must show different argmin widths in the
   easy vs width-hungry regions (sanity that the projection preserved the structure).

## 6. Confound ledger

- **C1 geometry vs input distribution** — closed by the spherical design + Φ-map (§2): AXIS and
  OBLIQUE have identical `t` marginals; geometry is the only difference.
- **C2 backend-favoring geometry** — trees favor AXIS, MLPs tolerate OBLIQUE; both run at every
  `d`, and any backend verdict must hold on both geometries or be reported geometry-conditional.
- **C3 error-table equality** — the per-width model sweep is trained ONCE per (d, geometry,
  seed) under the WSEL-4-vetted protocol (Tanh, full-batch, convergence gate, σ fixed at truth
  per §3.7); every backend fits the SAME table. Router choice is the only free variable.
- **C4 standardization** — identical input standardization for all backends.
- **C5 size × dimension** — `N_sel` and `d` fully crossed, never confounded.
- **C6 tie-band** — labelling tolerance stays at its frozen default; not swept (Decision 18).

## 7. Compute note

18 (d × geometry × seed) per-width sweeps × 12 widths of small-net training dominate; router
fits are seconds. Hours-scale under the 3-concurrent cap; runs as the wave-D multi-feature
slice after GO, root-run, backgrounded, land-as-produced.

## 8. Non-goals

No new target families beyond the projected canonical construction; no real data (WSEL-9's
scope); no tolerance sweep; no depth/ProbReg cells; no router-constant changes (FP-5.b binds).
