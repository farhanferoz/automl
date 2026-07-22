# WSEL-19 multi-feature toy REDESIGN (v2, the rotated-box construction) — ✅ GO (root verdict under the standing delegation, 2026-07-22 late)

**Verdict record.** Adversarial adjudicator review (2026-07-22 late) returned GO-WITH-AMENDMENTS:
the §2 construction mathematics CONFIRMED in full by an executed numeric probe (Householder
identities ≤ 1.6e−15; `v·x = t` exact in float64, ≤ 7.4e−7 after the float32 cast; AXIS
exchangeability airtight at stated scope; tanh-saturation hunt at d = 32 EMPTY — pre-activation
scale is d-independent under the actual init, the certified canonical regime); R1/R2/R4 satisfied;
no construction-level defect found. Three mandatory amendments (A1 verdict aggregation +
pre-dispatch regime reads; A2 calibration failure taxonomy; A3 trustworthiness field pinned) and
four recommended fixes (majority pass bar per the traced measure; selection pool-and-prefix;
scale-aware selftest tolerances; calibration-loader call sites) are ALL incorporated below — this
amended spec is the build authority. Root refinements over the adjudicator's proposed text, from
strand precedent: the trustworthiness re-run is a raised-cap retrain (the WSEL-8 same-precedent
repair), and a majority-gap failure gets the trace's own 4000-point evaluation protocol as its
pre-registered diagnostic. The adjudicator's key claims were independently re-verified at source
by the root before amending (oracle pointwise-min structure; `trustworthy` meta semantics at
`width_wsel19.py:959`; `min_seeds_for_pattern = 2` in the trace's declared criteria; the 1-D
pool-and-prefix convention at `width_wsel19.py:68-78`). The user is informed, not awaited (standing
delegation, re-confirmed twice 2026-07-22).

**Process record.** The original lifted construction (`wsel19-toy-design.md`, amended, GO 2026-07-22)
FAILED its own pre-registered d = 1 calibration check
(`automl_package/examples/capacity_ladder_results/WSEL19/wsel19_calibration_d1.json`, `passed: false`).
The mechanism was traced, not guessed (`automl_package/examples/capacity_ladder_results/WSEL19/warp_trace.json`, commit `8b3796a`): **two-sided width
convergence** — the probability-integral-transform lift `u = Φ⁻¹((t+r)/2r)` makes the easy region
need MORE width (practical-floor 3 → 5) and the hard region LESS, collapsing the per-region
practical-floor gap to 0 on 3/3 seeds while the canonical control differentiates on 3/3. This v2
spec is the redesign the trace's four binding requirements (R1–R4, recorded in `width.md`'s WSEL-19
multi-feature block, commit `ac5777b`) demand. Per the standing delegation (user, 2026-07-22, twice
re-confirmed) the go/no-go is rendered by the ROOT after an independent adversarial review; the user
is informed, not awaited. Iteration budget: this is redesign iteration 1 of at most 3 before the
recorded-negative-result terminal state.

**What carries over unchanged from the amended v1 spec** (re-affirmed, not re-litigated): the
inverted construction (F2), the §2b architecture generalization and its write-set/equivalence
guards (F1), the §3 grid axes and deterministic maximally-oblique geometry, the §3b provisioning
with independent draws and declared seed offsets (F5, C8), the F6 anchor-relative fit gate and its
pre-authorized n_train fallback, the F4 generator-true oracle falsifier, and every non-goal.
Sections below restate only what v2 CHANGES or what R1–R4 require to be re-derived.

## 1. What these toys must decide (unchanged)

The bake-off's multi-feature cells must (a) present routers with d > 1 inputs while provably
preserving the per-input routing signal certified in 1-D, and (b) contain the overfitting regime
motivating WSEL-7 ruling 3. The 1-D slice is already on record (72/72 cells); these cells decide
the input-size-relative half of rulings 2/5 and ruling 6's d ∈ {8, 32} test.

## 2. Generative construction v2 — the rotated uniform box (R1)

The construction stays INVERTED (F2: `make_hetero` draws its own input internally,
`nested_width_net.py:174`, no injection point). The Φ-warp is REMOVED entirely:

1. `(t, y, region) = make_hetero(n, draw_seed, r, σ)` — the canonical call, VERBATIM.
   `t ~ U(−r, r)` i.i.d. (source: `rng.uniform(-r, r, n)`), `r = 4π`, `σ = 0.05`.
2. **Decoy coordinates:** `c = (c₂, …, c_d)`, `c_j ~ U(−r, r)` i.i.d., drawn from a DECOUPLED RNG
   stream keyed `draw_seed + 5000` (the same decoupling idiom and offset the v1 `z`-stream used;
   the stream now feeds `uniform`, not `standard_normal`). At d = 1 there are no decoys.
3. **Embedding:** `t_vec = (t, c₂, …, c_d) ∈ ℝᵈ`, and

   `x = H t_vec`,   `H = H(d, geometry, seed)` a DETERMINISTIC orthogonal matrix whose first
   column is `v = geometry_vector(d, geometry, seed)` (v1's `v`, unchanged: AXIS `v = e₁`;
   OBLIQUE `v = s/√d`, `s` the per-seed deterministic ±1 pattern):
   - AXIS: `H = I_d` exactly (no arithmetic at all).
   - OBLIQUE: the Householder reflection `H = I − 2wwᵀ/‖w‖²` with `w = e₁ − v` (if `w = 0`,
     `H = I`). Standard identities give `H = Hᵀ`, `HᵀH = I`, `H e₁ = v` — asserted numerically in
     the selftest (`‖HᵀH − I‖_max ≤ 1e−12`, `‖H e₁ − v‖_max ≤ 1e−12`), not taken on faith.

   Consequences, each exact (up to the float32 storage cast):
   - `v·x = (H e₁)ᵀ (H t_vec) = e₁ᵀ t_vec = t` — **the network-visible coordinate along `v` IS the
     canonical coordinate `t`. No warp, no reparameterization of any kind: R1 is satisfied by the
     identity map.** The target function the model must learn is `f(x) = h(v·x) = h(t)` with the
     canonical seam geometry, region widths, and frequencies untouched.
   - At d = 1, AXIS: `x = t` exactly — the calibration cell's model input is the canonical toy's
     input, byte-for-byte (float32 cast aside). The §5.2 anchor is a true anchor, not an analogy.
   - `x` is distributed as an **H-rotated uniform box**: `t_vec ~ U(−r, r)ᵈ` with i.i.d.
     coordinates (t from `make_hetero`'s stream, decoys from the decoupled stream — independence
     is by disjoint RNG streams, the strand's standing idiom), so `x = H t_vec` has the box law
     rotated by the known, deterministic `H`. At AXIS, `x ~ U(−r, r)ᵈ` with i.i.d. coordinates
     EXACTLY.

`y`, `region`, `t` pass through untouched — the §5.1 identity check is unchanged and still holds
by construction.

**Why this cannot reproduce the traced failure mode:** the trace localized the two-sided
convergence to the Φ-reparameterization of the model-visible coordinate (easy-side degradation
unconfounded; canonical control clean). v2 has NO reparameterization — along `v` the model sees
`t` itself, so at d = 1 the per-width sweep is the canonical sweep. The only new model-visible
structure at d > 1 is `d − 1` irrelevant i.i.d. coordinates, which is exactly the intended
difficulty (find the signal direction), not a distortion of the signal.

**Scale note (recorded as a ledger entry, C4):** no backend and no per-width net standardizes its
inputs anywhere in this pipeline (verified against `routing.py` and every `_fit_*` in
`width_wsel19.py` — standardization is identically ABSENT, uniformly). The 1-D slice's routers
consumed raw `t` on (−r, r); the v1 lift silently moved router inputs to N(0, 1) scale, a protocol
deviation from the 1-D anchor that v2 REMOVES: router inputs return to (−r, r)-scale in every
cell, matching the certified 1-D slice.

## 2b. Per-width architecture generalization (F1 — unchanged)

Carried verbatim from the amended v1 spec: the §2b input-dimension generalization already landed
(`architectures.py`; 1-D behavior byte-identical, equivalence suite 9/9) and is untouched by v2.
No new package edits.

## 3. Grid axes (unchanged)

d ∈ {2, 8, 32} × geometry ∈ {AXIS, OBLIQUE (deterministic, maximally oblique)} × N_sel ∈
{75, 300, 1200} × seeds {0, 1, 2} × 4 backends × 2 modes = **432 cells**. Sparse-oblique excluded;
hetero3 absent (both recorded decisions, unchanged).

## 3b. Data provisioning (unchanged) + the R2 calibration protocol pin

Provisioning is v1's with ONE amendment (adjudicator finding 5): train = TRAIN_N = 1500 at the
cell's base seed; report = an independent draw of 2000 at seed + 2000; F6 fallback n_train = 4000
pre-authorized at d = 32. **Selection sets are POOL-AND-PREFIX, mirroring the certified 1-D
slice's own protocol (`width_wsel19.py:68-78`):** ONE independent §2 draw of max(N_sel) = 1200
points at seed + 1000, shuffled by a seeded permutation (`np.random.default_rng(seed)`); every
smaller N_sel is that shuffled pool's PREFIX. The three sizes are then nested — a size-vs-size
comparison reflects added selection data, never independently-resampled noise (v1's independent
per-size draws shared an x-prefix but resampled y-noise, verified by probe: x prefix shared, y
prefix not).

**R2 (new, binding): the §5.2 regime check trains at effective n = 600, EQUAL to the canonical
carve.** The traced confound: canonical WSEL-6 sweep nets trained on 600 points (1500-draw →
50/50 selection carve → 750 → every-5th internal val split → 600), while v1 calibration nets
trained on 1200 (1500 → every-5th val split, no carve). v2 pins the calibration's regime block to
`n_train = 750`, which under the shared every-5th val convention (`VAL_EVERY = 5`, reused
verbatim) yields exactly 600 gradient-visible points — the canonical-vs-lifted comparison is now
at matched effective training size, killing the confound at the protocol level. The artifact
records `n_train` and `n_train_used` per seed so the pin is checkable from the JSON alone.

## 5. Pre-registered validity checks (in order; BEFORE any backend verdict)

1. **Identity (§5.1, unchanged):** `(t, y, region)` equals `make_hetero(n, seed)`'s output
   bit-for-bit at every d. Plus the v2 construction selftests: `v·x = t` to float32 precision
   (the existing 1e−5 tolerance carries over safely — measured cast error ≤ 7.4e−7 at d = 32
   OBLIQUE); Householder orthonormality and `H e₁ = v` to 1e−12; at AXIS ONLY, box-support and
   moment checks on the marginals with SCALE-AWARE tolerances (adjudicator finding 6 — the v1
   N(0,1)-era constant 0.1 legitimately fails on box scale): `max|x_ij| ≤ r`;
   `max_j |mean_j| ≤ 0.05·r`; `max_j |std_j − r/√3| ≤ 0.05·r/√3` (measured on a correct n = 5000,
   d = 32 draw: 0.28 and 0.157 — comfortably inside, and a 10%-of-scale bug trips both). At <!-- numcheck-ignore: selftest-tolerance calibration measurements (a no-training marginal probe), not a ledger result -->
   OBLIQUE, ONLY `v·x = t` and orthonormality run — per-coordinate support exceeds r by design
   (rotated box; measured up to ≈ 2.4·r at d = 32), so support/moment checks are AXIS-scoped
   structurally, not leniently. At d = 1 AXIS, `x ≡ t.astype(float32)` exactly.
2. **d = 1 calibration cell (§5.2, R2 + R3):** two blocks, one artifact
   (`wsel19_calibration_d1.json`), gating every d > 1 cell exactly as before
   (`_load_calibration_or_refuse` semantics unchanged):
   - **Regime block (R2-pinned, n_train = 750 → 600 effective):** per seed, per-width sweep at
     d = 1 AXIS; compute the per-region generator-TRUE error curves on the report split; apply
     the R3 differentiation criterion below. **PASS requires differentiation on ≥ 2 of 3 seeds
     (the traced measure's own declared bar, `min_seeds_for_pattern = 2` — adjudicator finding 4:
     3/3 on a fresh draw is a coin the design does not need to flip; the trace's canonical
     margins are gaps 5/1/3 with seed 1 exactly at the boundary) AND every gate-counted net
     trustworthy per the cached meta's own field (replay-trajectory trustworthy AND
     `hit_cap = False` — the `width_wsel19.py:959` semantics, unchanged; amendment A3). The 3/3
     outcome is recorded informationally.**
   - **Anchor block (grid-provisioned, n_train = 1500):** the SAME sweep at the grid's own
     provisioning, providing `anchor_ratio_to_noise_floor` (worst-seed best-fixed-MSE /
     noise-floor, F6's anchor — compared apples-to-apples against grid cells that train at the
     same provisioning) and requiring all nets trustworthy. Its regime/floor numbers are
     DECISION-BEARING via the §7 pre-dispatch read (amendment A1) — no longer merely
     informational.
   - **Failure taxonomy (pre-registered, declared before any v2 net is trained — amendment A2):**
     at d = 1 AXIS the v2 construction is bit-identical to the canonical toy (`x = t`), so NO
     §5.2 outcome can indict the construction; a d = 1 failure is a protocol/draw artifact and
     NEVER consumes a construction-redesign iteration — the 3-iteration budget binds d > 1
     mechanisms only. Bounded pre-authorized responses: (a) an untrustworthy net → exactly ONE
     raised-cap retrain (cap × 2 — the WSEL-8 same-precedent repair), both results recorded; a
     seed still untrustworthy after that drops from the gate's numerator (pass still possible on
     the remaining seeds under the majority bar; fewer than 2 trustworthy seeds → protocol-level
     halt, recorded). (b) A majority-gap failure with trustworthy nets contradicts the trace's
     own canonical control → ONE pre-registered diagnostic: re-evaluate the SAME nets' floors on
     a 4000-point fresh draw at the trace's own evaluation protocol (`n_fresh = 4000`, seed
     offset + 50000 — `warp_trace.json`); if differentiation passes there, the 2000-point report
     split was the artifact — recorded, the 4000-point evaluation becomes §5.2-operative, gate
     PASSES; if it fails there too → protocol-level negative result, terminal for the
     multi-feature slice, recorded — still not a construction iteration.
3. **R3 differentiation criterion (replaces raw argmin EVERYWHERE it gated):** for a per-region
   true-error curve, the **practical-floor width** at fraction φ is the smallest w ∈ {1..12} with
   region-mean true error ≤ φ · σ² (σ² = 0.0025, the noise floor; the traced measure,
   implementation ported from `warp_trace.py::_practical_floor_width`). The check: at the primary
   fraction φ = 0.2, both regions' floors EXIST and
   `floor_hard − floor_easy ≥ 1`. Fractions {0.1, 0.3} are recorded alongside as robustness
   context (recorded, not gated — same as the trace). Raw per-region argmins remain recorded
   informationally; they no longer gate anything (the trace showed them unusable: genuinely
   non-monotonic true-error curves at tiny errors).
4. **Fit gate (§5.3/F6, unchanged mechanics):** per (d, geometry, seed) cell,
   best-fixed-width held-out MSE / noise floor must not exceed the anchor block's
   `anchor_ratio_to_noise_floor`. Failure → §3b fallback; still failing → VOID_FOR_FIT.
5. **Hidden-ness falsifier (§5.4/F4, unchanged):** generator-true oracle must beat the best fixed
   width by ≥ 10% of the latter's true error on the report split; else VOID_FOR_ROUTING.
6. **Regime visibility per cell (§5.5, now R3-consistent):** the per-cell check applies the R3
   practical-floor criterion (primary fraction, gap ≥ 1) to the cell's own true-error table. A
   missing floor in either region ⇒ `regime_visible = false`. Recorded per cell (with the floor
   widths and the fractions-robustness triple), not refused on — VOID statuses remain exactly
   {fit, routing}, as in v1.
7. **Verdict aggregation (pre-registered — amendment A1):** every bake-off ruling read from the
   multi-feature slice is computed over cells with `regime_visible = true`, `fit_status = ok`,
   and `routing_status = ok`. Other cells are reported but carry NO verdict weight. Since regime
   visibility and both VOID statuses are properties of the (d, geometry, seed) sweep and its
   error tables (shared by all 8 backend×mode cells at that triple), the unit of aggregation is
   the triple: **if at any (d, geometry) level fewer than 2 of 3 seeds yield regime-visible,
   non-void triples, every ruling depending on that level (the input-size-relative half of
   rulings 2/5; ruling 6's decisive d ∈ {8, 32} test) is recorded OPEN at that level — never
   decided from the surviving cells** (the survivor-bias closure).

## 6. Confound ledger — re-derived from scratch (R4)

- **C1 (geometry vs input distribution) — restated honestly, exact where it can be:** every
  (d, geometry) carries the SAME `(t, y, region)` bit-for-bit (inversion, unchanged — exact). The
  input law is the H-rotated uniform box: AXIS and OBLIQUE laws are exact rotations of one
  another by the known deterministic H. Cross-geometry identity IN COORDINATES (which v1's
  rotation-invariant Gaussian bought, at the price of the warp that killed the toy) is
  deliberately NOT claimed — see the trilemma below.
- **The R4 marginal-sniffing derivation (the reason the "obvious" candidate was rejected):** the
  obvious no-warp candidate — uniform ridge `t` along `v` with a GAUSSIAN complement — flags the
  signal direction unsupervisedly at AXIS: coordinate 1's marginal is U(−r, r) (bounded, flat)
  against d − 1 Gaussian marginals, so a backend can locate the signal coordinate from inputs
  alone at AXIS but not at OBLIQUE (where all d marginals are identical mixtures). That
  asymmetry rides exactly the contrast C2 measures and would be attributed to backend geometry.
  **v2 closes it exactly:** at AXIS all d coordinates are i.i.d. U(−r, r), so the input law is
  EXCHANGEABLE under coordinate permutation — no statistic of x alone can distinguish the signal
  coordinate from a decoy (a symmetry argument, not an empirical claim). At OBLIQUE, the signal
  direction is one of the d box axes {H e₁, …, H e_d}; input-only structure can recover the axes
  as a SET (the box's own geometry) but carries no information about WHICH axis is the signal.
  The unsupervised information about `v` is therefore equalized across geometries: in both, "the
  signal is one of d indistinguishable box axes."
- **The residual, declared:** the box's orientation co-rotates with `v`, so AXIS vs OBLIQUE
  varies the coordinate-alignment of the WHOLE generative structure (signal direction AND input
  support), not the signal direction in isolation. This is the intended manipulated variable for
  C2's purpose (trees natively exploit coordinate alignment; that is what the geometry axis
  tests), and verdicts remain geometry-conditional per C2 — but any axis-vs-oblique gap must be
  attributed to "coordinate alignment of the structure," never to split-mechanics specifically.
  **The trilemma, stated so the choice is checkable:** (i) R1 (`v·x = t ~ U(−r, r)` exactly,
  unwarped), (ii) identical coordinate-space input law across geometries, and (iii) no
  unsupervised signal-direction tell at AXIS — no known non-degenerate construction satisfies
  all three (the classical all-projections-uniform law, the Archimedes sphere, exists only at
  d = 3 with measure-zero support — unusable as a regression toy family across d ∈ {2, 8, 32}).
  v1 chose (ii)+(iii) and its warp broke the toy at (i)'s expense; v2 chooses (i)+(iii) — the
  strongest satisfiable combination that keeps R1 — and downgrades (ii) to exact-up-to-rotation.
- **C2 (backend-favoring geometry):** unchanged — both geometries at every d; verdicts hold on
  both or are reported geometry-conditional.
- **C3 (error-table equality):** unchanged — per-width models trained ONCE per (d, geometry,
  seed); every backend fits the SAME table.
- **C4 (standardization):** re-verified at source for v2 — NO input standardization exists
  anywhere in the pipeline (routers and per-width nets alike consume raw x), identically for all
  backends, and v2 returns router inputs to the (−r, r) scale of the certified 1-D slice (the v1
  lift had silently moved them to N(0,1) scale — a deviation now removed).
- **C5 (size × dimension):** unchanged — fully crossed; N_sel independent of training data.
- **C6 (tie-band):** frozen default, not swept (Decision 18).
- **C7 (oracle optimism):** closed by the generator-true oracle (§5.4, unchanged).
- **C8 (provisioning independence):** unchanged — independent draws, declared offsets.
- **C9 (NEW — calibration protocol match, from R2):** the regime block trains at effective 600 =
  the canonical carve; recorded per seed in the artifact. The v1 600-vs-1200 confound cannot
  recur without being visible in the JSON.

## 6b. Starvation arithmetic (restated for v2 — unchanged numerically)

Router training sees N_sel ∈ {75, 300, 1200} points with d ∈ {2, 8, 32} inputs and up to 12
classes. At N_sel = 75, d = 32: 75 labels against a frozen (32, 32) MLP (~1.5k router weights at
d = 32 input) — the overfitting regime WSEL-7 ruling 3 flagged, deliberately in-grid; the 1-D
slice's starved-cell inversion (frozen most robust at n = 75, early-stopping ~6× worse) is the
on-record anchor this grid re-tests at real dimensionality. v2 changes none of these sizes; the
decoys change the router's INPUT dimension only, which is the point.

## 7. Compute note (updated for v2)

Per-width sweeps dominate; router fits are seconds-scale. New sweep cache required (the DATA
changed): calibration = 36 nets at n_train 750 + 36 at 1500 (d = 1); grid = 18 (d × geometry ×
seed) sweeps × 12 nets at n_train 1500 (d = 32 unmeasured — the d = 1 calibration plus the FIRST
d = 32 sweep measure per-net cost before the bulk dispatches, v1 §7 discipline retained). All
runs `AUTOML_DEVICE=cpu OMP_NUM_THREADS=4`, ≤ 3 concurrent heavy processes, `systemd-inhibit`
wrapped.

**Pre-dispatch decision reads (pre-registered — amendment A1; zero extra compute, both sweeps
already staged):** before the bulk grid dispatches, the root reads (a) the anchor block's
1200-effective practical floors and (b) the §5.5 floors of the FIRST d = 2 and FIRST d = 32
sweeps (d = 32's is the §7 cost-measurement sweep). If the anchor block shows no floor gap ≥ 1 on
≥ 2 of 3 seeds AND neither early sweep shows a floor gap ≥ 1, bulk dispatch HALTS and the outcome
is handled under the §5.2 failure taxonomy — the 432-cell grid never runs against a construction
whose operative-protocol regime signal is already measured dead. If floors are None at the first
d = 32 sweep, the F6 fallback applies BEFORE bulk d = 32 dispatch. Partial-regime outcomes (some
levels alive) proceed; the §5.5 aggregation rule governs what the surviving levels may decide.

## 8. Build plan (exact, for the implementing session)

Write set: `automl_package/examples/width_wsel19_toys.py`, `automl_package/examples/width_wsel19.py`
(+ the results dir by runs). NO package edits; NO edits to `routing.py`/`width_wsel7.py`/
`width_wsel6.py`.

1. `width_wsel19_toys.py`: remove `_uniform_to_standard_normal`/`scipy.stats.norm` and the
   z-projection; add `basis_matrix(d, geometry, seed)` (Householder, AXIS short-circuit to I);
   `make_hetero_multifeature` implements §2 v2 (decoy stream at `draw_seed + 5000`); selftest
   replaces the N(0,1) marginal checks with §5.1's v2 checks (box support/moments at AXIS,
   `v·x = t`, orthonormality, d = 1 exact reduction) and keeps identity + provisioning checks.
2. `width_wsel19.py`: port `_practical_floor_width` from `warp_trace.py`; rewrite
   `_regime_prefers_different_widths` → practical-floor form returning (visible, floors, argmins,
   robustness triple); rewrite `run_calibration` → two-block artifact + v2 pass rule + failure
   taxonomy fields; `_load_calibration_or_refuse` reads the new schema and returns a normalized
   dict that PRESERVES the `anchor_ratio_to_noise_floor` key (adjudicator finding 7 — BOTH
   consumers named: the §5.3 gate comparison and the cell-JSON provenance record in
   `run_cell_multifeature`, currently `width_wsel19.py:1189` and `:1229`); §3b selection
   provisioning → pool-and-prefix (finding 5); per-cell §5.5 + cell JSON `validity_checks` gain
   the floor fields; `--selftest` updated to the new schemas. **Cache hygiene:** add a
   construction tag (`box`) to `_mf_model_cache_paths` so v2 caches can never collide with any
   older scheme. (The v1 spec's delete-the-Φ-cache step is a NO-OP — no `_mf_cache/` exists in
   the working tree; the Φ-era nets live only in the prior session's scratchpad. Verified.)
3. The failed v1 calibration artifact is superseded on disk by the v2 run (its record: git
   history + the width.md table). `warp_trace.py`/`warp_trace.json` stay untouched as the frozen
   trace record; reproducing the trace requires checkout ≤ `8b3796a` (it imports the live toys
   module) — recorded here, not engineered around.
4. Verify (build): toys selftest PASS; driver selftest PASS; equivalence suite still 9/9; ruff
   clean; `--summarize` still aggregates the 72 committed 1-D cells unchanged; `git status`
   clean but for the two drivers before commit.

## 9. Non-goals (unchanged from v1, restated)

No new target families beyond the (now unwarped) lifted canonical construction; no real data; no
tolerance sweep; no depth/ProbReg cells; no router-constant changes (FP-5.b binds); no hetero3
cells; no re-tuning of any backend hyperparameter for v2.
