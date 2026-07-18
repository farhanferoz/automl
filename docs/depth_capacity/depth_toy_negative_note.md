# Depth toy construction — negative result (D1 kill criterion triggered)

**Date:** 2026-07-16. **Status:** strand `depth.md` task D1 exhausted all three pre-registered
candidate constructions without a pass. Per the task's kill criterion, the strand does not
proceed to Phase II (architecture battery) until a depth-hungry-but-learnable toy exists.

## What was tried

All three candidates from `depth.md` §D1, tried in the mandated order, each on seeds `{0, 1}`,
against the four pre-registered bars (P1 learnable-deep, P2 depth-hungry, P3 graded, P4
easy-region-flat — thresholds and formulas in `automl_package/examples/depth_toy.py`'s module
docstring). All probe nets converged trustworthily (`convergence.py`'s full-trajectory rule:
`converged=True`, `hit_cap=False`, not still improving) — every failure below is a genuine
converged read, not an artifact of under-training.

Convergence-gate constants were recalibrated once during construction (`MIN_DELTA`: `3e-5` →
`1e-3`) after the `hierarchical_spline` pilot showed full-batch Adam's own late-training val-loss
jitter (O(1e-4..1e-3) in standardized-y units) was large enough that the original `min_delta`
never let a genuinely-plateaued run declare `converged`. All reported numbers below use the
recalibrated constant.

### Candidate 1 — gentle composition (`f_D(x) = g^{o4}(x)`, `g(u)=sin(1.5u)`)

Fails **P2** on both seeds: the param-matched wide-shallow net (width 80, 1 hidden layer) does
NOT stall. `M_hard` ratio-to-floor is 1.19 (seed 0) and 1.41 (seed 1) — well under the 2.0x
stall bar, and close to the depth-4 net's own ratio (0.92, 0.98). A follow-up numerical check
(no training — direct evaluation of the composed map) explains why: composing `sin(1.5u)` with
itself saturates into `[-1, 1]` after the FIRST application, so further composition does not add
macroscopic oscillation — the extrema count of `g^{oD}(x)` over a fixed domain is essentially
INDEPENDENT of `D` (checked for `D=1..4` at several domain widths; extrema count matched `g^{o1}`
in every case). The construction is deep to WRITE but not deep to LEARN: it is representationally
close to a single sine wave, so a wide shallow net fits it just as well as a deep narrow one.

Evidence: `automl_package/examples/capacity_ladder_results/D_TOY_PROBES/depth_toy_probe_gentle_composition_seed0.json`,
`..._seed1.json`.

### Candidate 2 — hierarchical piecewise-smooth (`s(s(u))`, `s(u)=sin(1.5*pi*u)^2`)

Fails **P3** on both seeds, and P2 is seed-INCONSISTENT (passes on seed 0 with a large margin,
ratio 4.07; fails on seed 1, ratio 1.36). The `M_hard` depth ladder is not monotonic and not
seed-stable: on seed 0, depths 1-3 all stall near the wide-shallow level (~4x floor) while depth 4
alone drops to 0.87x floor (a cliff, not a graded curve); on seed 1, depths 2-4 all fit well
(~1.0x floor) and only depth 1 is bad. This pattern — a result that flips between "clearly
depth-hungry" and "clearly not" depending only on the random seed — is the signature of an
optimization-landscape/local-minimum artifact rather than a genuine, reproducible capacity
story, and is exactly what the pre-registered P3 bar (graded, monotonic ordering across BOTH
seeds) exists to catch. `s(u)=sin(1.5*pi*u)^2` is close in spirit to a smooth logistic/tent-map
conjugate; composing it twice appears to reintroduce the training-instability risk the tent map
was refuted for, even though each level individually is "low-frequency."

Evidence: `automl_package/examples/capacity_ladder_results/D_TOY_PROBES/depth_toy_probe_hierarchical_spline_seed0.json`,
`..._seed1.json`.

### Candidate 3 — 2-D multiplicative (`sin(a*x1)*sin(a*x2)` hard, `sin(a*x1)+sin(a*x2)` easy)

Fails **P4** on both seeds (`M_easy` ratio 3.47, 2.68 — needs `<=1.3`), and P1/P2/P3 also fail on
at least one seed. Root cause, confirmed with a targeted follow-up (`depth1`/`width=8` net trained
on the ADDITIVE formula ALONE, no hard/transition region mixed in): even in isolation the easy
region's ratio is 1.34 (seed 0) — already at the edge of the 1.3x bar — and a frequency sweep
(`a in {0.4, 0.6, 0.8, 1.0}`) never brought it comfortably under 1.3 on both seeds (ratios ranged
1.20-1.49). Unlike candidates 1/2's easy region (a single straight line, trivial for 1 active
node), candidate 3's easy region is itself a genuine two-frequency function of two INPUT
dimensions; the pre-registered `NARROW_WIDTH=8` (chosen to match candidates 1/2's 1-D easy-line
baseline) does not comfortably cover this even before the hard region competes for the same
shared weights. This is a construction mismatch, not a bug: the width budget that makes width/
depth-flatness trivial in 1-D is tight-to-insufficient for a 2-D two-term sum.

Evidence: `automl_package/examples/capacity_ladder_results/D_TOY_PROBES/depth_toy_probe_multiplicative_2d_seed0.json`,
`..._seed1.json`.

## Verdict

Kill criterion met: all three pre-registered candidates exhaust the probe bars on 2 seeds each
with no candidate passing all four. Per `depth.md` D1, the strand does not proceed to D2
(architecture harness) or D3 (3-arm battery) on a non-hungry toy.

## What a fourth attempt would need to address

(For the orchestrator's use if a fresh D1 candidate is attempted — not an instruction to proceed.)

- Candidate 1's failure mode (composition saturates, doesn't add oscillation) suggests any
  future "iterate `g` `D` times" construction needs `g` to NOT contract into a bounded range after
  one application — e.g. an expanding map on an unbounded or periodic domain, or an explicit
  per-level frequency multiplier so input-space complexity genuinely grows with `D` (while still
  staying short of the tent map's already-refuted GD-unlearnability).
- Candidate 2's failure mode (seed-unstable optimization) suggests avoiding maps whose local
  slope at their own fixed point exceeds 1 (a `chaos-adjacent` regime) even for compositions as
  shallow as two levels; a monotonic (non-oscillatory) generating spline composed with itself
  would avoid the instability but likely also fails to be depth-hungry (monotonic composition
  doesn't add wiggle).
- Candidate 3's failure mode (width budget too tight for a genuinely 2-D easy region) suggests
  either a wider `NARROW_WIDTH` specifically for 2-D candidates (breaking uniformity with
  candidates 1/2, which may be fine since it is a probe-construction choice, not a bar), or an
  easy region that is trivial in the SAME sense as 1/2's (e.g. a plain linear function of
  `x1, x2`, not a sinusoid), reserving the sinusoidal additive/multiplicative contrast entirely
  for the hard-vs-easy comparison at a HIGHER width.

## File manifest

- `automl_package/examples/depth_toy.py` — generators (all 3 candidates) + net builders + probe
  driver + `--selftest`.
- `automl_package/examples/capacity_ladder_results/D_TOY_PROBES/*.json` — 6 probe results (3
  candidates x 2 seeds), each with full convergence trajectories.
