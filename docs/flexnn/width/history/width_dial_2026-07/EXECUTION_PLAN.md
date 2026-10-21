# Execution plan — per-input WIDTH dial on the ramped-sinc positive control (2026-07-11)

## 0. Objective (one sentence)

Build a **nested-width** network + a **distilled per-input width selector** — the width analog of the
depth machinery — and show on the ramped-sinc toy that the selector correctly reads, per input, how
many hidden nodes each input needs (few in the flat tails, many in the oscillating centre), and that
per-input width matches-or-beats a single global width. This is the **learnable positive control** the
depth lane never had.

## 1. Why this toy / why width (background, settled)

- Depth lane finding stands: the provably-deep tent map is representable-but-not-learnable; no learnable
  depth positive control exists in the bounded search (T1, certified `T1/T1_PATH2_ADJUDICATION.md`).
- Ramped sinc `y = sin(x)/x + 0.04x`, `x∈[-5π,5π]`, noise σ=0.05, is **learnable by a small net** (source
  paper: single hidden layer, N≈7 nodes → >99.3% correlation). The property that makes it learnable
  (low frequency) is why it needs **width, not depth** — so we read WIDTH here, keeping the two dials
  strictly separate (user directive 2026-07-11).
- **Step-0 probe already run** (`scratchpad/sinc_width_probe.py`): fixed-width single-layer nets, per
  region, MSE as multiple of the σ²=0.0025 noise floor:
  | width | tail(easy) | centre(hard) |
  |--:|--:|--:|
  | 1 | 7.4× | 42× |
  | 7 | 2.4× | 1.5× |
  | 16 | 1.3× | 1.2× |
  The centre is dramatically width-hungry AND **learnable** (reaches ~1.2–1.5× floor). Per-input width
  gradient confirmed. (Caveat: tails not perfectly flat — a global net shares capacity; region boundary
  can be tightened. Core signal unambiguous.)

## 2. Design (decision-complete — no judgment left to the builder)

**Doctrine (bind):** strictly probabilistic (per-example Gaussian log-likelihood; NO MSE-only, no penalty/λ);
held-out LL is the only bar; frozen model then post-hoc selector (two-stage); examples-level only, **NO
library edits** (a library port is a separate user-gated decision, like K7). Keep depth fixed = single
hidden layer; the ONLY capacity axis is width.

### 2a. Nested-width network — `automl_package/examples/nested_width_net.py`
- Arch: `Linear(1 → W_max)` → `tanh` → hidden vector `h ∈ R^{W_max}`; two readouts off `h`:
  `mean = Linear(W_max→1)`, `logvar = Linear(W_max→1)`. (Probabilistic: outputs (μ, logσ²).)
- **Width-k forward:** zero `h[k:]` (mask nodes k..W_max−1) before BOTH readouts → only first k nodes
  contribute. width-k ⊂ width-(k+1), shared weights (nested-dropout / slimmable structure).
- **NESTED-width training schedule** (mirror `LayerSelectionMethod.NESTED` semantics, but over nodes):
  per minibatch sample `k ~ U{1..W_max}`, forward with the width-k mask, Gaussian-NLL loss, backprop.
  Trains every prefix to be a valid predictor. NO selector head during phase-1 (schedule, not a gate).
- Template to mirror: the depth NESTED block/masking in
  `automl_package/models/flexible_neural_network.py` (hidden_layers_blocks + identity-prefix idea) and
  the Gaussian-LL formula in `capacity_ladder_f2._fixed_depth_log_likelihood`.
- **Selftest (no training):** (a) prefix property — width-k output is invariant to perturbing weights of
  nodes ≥k (columns k.. of both readouts); (b) width-k and width-(k+1) agree on nodes <k (nesting);
  (c) a monkeypatched all-widths pass returns finite (μ, logvar) of the right shapes.

### 2b. Per-width scoring + WIDTH construction bar
- After phase-1, evaluate the frozen nested net at every width k=1..W_max on held-out x → per-sample
  Gaussian LL vector per width (`fixed_width_ll: dict[k -> (n,)]`), exactly analogous to T1's
  `fixed_depth_ll`.
- **Construction bar** (mirror `capacity_ladder_t1._construction_bar`, per region, 2·SE plain-bootstrap):
  centre (|x|≤2π) LL improves k_lo→k_mid by >2·SE; tail (|x|>2π) flatter. Confirms the nested schedule
  preserved the per-input width gradient (the Step-0 probe showed it for dedicated nets; this checks the
  ONE nested net reproduces it).

### 2c. Distilled per-input width selector (two-stage, SOFT)
- Freeze the nested net. Train a small selector `x → distribution over widths {1..W_max}` (the S1/H1
  **SOFT** recipe, prior-dominant), distilling the per-input held-out-best width. Reuse the standalone
  router template `capacity_ladder_h1.py::_RouterMLP` + the SOFT objective verbatim in structure.
- Selector trained on a held-out-within-train split (index-parity, like H1/S1 — no leak).

### 2d. Pre-registered bars (write `capacity_ladder_results/W1/PREREGISTRATION.md` BEFORE the real run)
- **Bar (i) CONSTRUCTION:** centre held-out LL climbs with width (k1→k_mid, >2·SE) and reaches within
  ~1.5× noise floor; tail comparatively flat. ≥2/3 seeds. (Positive control HAS signal + is learnable.)
- **Bar (ii) RECOVERY:** distilled selector assigns strictly larger expected width to centre than tail
  (separation > 2·SE) on ≥2/3 seeds. (Per-input width is READ correctly.)
- **Bar (iii) DEPLOY:** selector's per-input blended held-out LL ≥ best single global width (matches-or-
  beats, one-sided) AND within 0.02 nat of the per-input oracle width. ≥2/3 seeds. (It deploys — mirror
  of H1's shipping bar.)
- Metric: per-example Gaussian LL; oracle = `-log(0.05) - 0.5·log(2π) - 0.5`. Config: 3 seeds
  (0,1,2), W_max=16, single hidden layer, tanh, sinc data from the probe (reuse verbatim), 1500 train /
  500 test, Adam lr 1e-2, ~2500 epochs phase-1 (tune to convergence in smoke).

### 2e. Driver — `automl_package/examples/sinc_width_experiment.py`
- `--selftest` (2a/2c wiring, no train), `--smoke` (W_max=6, 300 ep, 1 seed, no save), `--config`/all.
- Structure/JSON-summary style mirror `capacity_ladder_t1_path2.py` + `capacity_ladder_h1.py`.
- Writes `capacity_ladder_results/W1/w1_summary.json` with per-seed bars + verdict
  (FOUND_LEARNABLE_WIDTH_CONTROL if bars i+ii pass ≥2/3 and iii holds; else the specific failure).

## 3. Execution structure (token-efficient tiering)

- **Phase A — plan (this doc).** Opus/main, done.
- **Phase B — BUILD (delegate: ONE `task-worker`, Sonnet/high).** Implement 2a + 2b + 2c + 2e + both
  selftests + the W1 PREREGISTRATION.md, against this spec, mirroring the named templates. It is ONE
  coherent build (arch + driver share an interface) — do not split. Input contract handed to it =
  §2 verbatim + exact template file:paths + the verify command below + non-goals.
  - **Verify command the worker MUST run green before returning:**
    `AUTOML_DEVICE=cpu ~/dev/.venv/bin/python automl_package/examples/nested_width_net.py --selftest`
    and `... sinc_width_experiment.py --selftest` and `... sinc_width_experiment.py --smoke`
    (smoke must show centre LL rising with width). Worker returns: selftest PASS/FAIL + smoke centre-vs-
    width numbers + the file paths written.
- **Phase C — VERIFY (Opus/main, cheap).** I re-run the three commands myself and read ONLY the numbers
  (selftests green; smoke centre-LL climbs with width; tail flatter). No full code re-read unless a
  number is off. If off → targeted fix (me or bounce to worker with the exact failing check).
- **Phase D — RUN (compute, ~free tokens).** Launch the 3-seed real run detached
  (`setsid nohup … AUTOML_DEVICE=cpu … sinc_width_experiment.py > scratchpad/w1.log 2>&1 &`), monitor
  for `W1/w1_summary.json` from the main thread (bg wait-for-file loop).
- **Phase E — INTERPRET + CERTIFY (Opus/main; adjudicator only if it becomes a shared claim).** Read
  bars i/ii/iii from the summary. If clean → this is the learnable positive control; fold a short note.
  One fresh-context adjudicator cert only if we promote it to the report (mirrors the program's bar).

## 4. Non-goals / guards
- No library edits (examples-level only; library port is separately user-gated).
- Do not touch depth machinery or the depth-lane finding — width is a SEPARATE dial.
- No penalty/λ; strictly probabilistic held-out LL.
- Keep the depth-lane path-2 write-up staged/uncommitted decision separate (user to call).

## 5. Rough cost
Opus: this plan + a numeric verify + interpretation (small). Sonnet: the whole build (bulk). Compute:
3 seeds × (phase-1 nested fit + width sweep + selector) on 1-D data ≈ minutes–low-tens-of-minutes CPU.

---

## 6. W1 RESULT + W2 revision (2026-07-11, post-first-run)

**W1 (nested-width on ramped sinc) → `FAIL_i_CONSTRUCTION` (bar i 0/3).** Diagnosis (grounded in the
per-width held-out LL curves, `W1/w1_summary.json`): (1) the sinc has **no width-flat region** — the
"tail" gains ~0.9 nat from width, nearly as much as the centre, so there is little per-input contrast
(my earlier "tail far less width-sensitive" read of the Step-0 MSE-multiples was an overstatement — in
nats the tail is width-hungry too); (2) **nested uniform-width training under-fits the hard centre**
(0.55 nat short of floor at w_max=16 vs dedicated nets' ~0.09 nat) because a per-sample uniform width
draw trains w_max only 1/w_max of the time. Net effect: selector assigned ~max width (12–13/16) to
BOTH regions → no differentiation → deploy barely beat a single global width. W1 files kept for record,
uncommitted.

**W2 toy — heterogeneous, PROBED LEARNABLE (`scratchpad/hetero_toy_probe_v2.py`), decision-complete:**
- `make_hetero(n, seed, R=4*pi)`: `x ~ Uniform(-R, R)`; easy `x<0`: `y = (0.5/R)*x` (a straight line);
  hard `x>=0`: `y = 0.5*sin(x)` (2 native-frequency periods over `[0,4π]`); noise σ=0.05; continuous at 0.
  `region`: 0=easy(`x<0`), 1=hard(`x>=0`). **Standardize x** (spans ±4π).
- Why native frequency: the packed `sin(2π·P·x)` on `[0,1]` is UNLEARNABLE by a small net (needs
  first-layer weights ~2πP; spectral-bias stall — the same frequency wall as the tent map, confirmed in
  `hetero_toy_probe.py` for P=2,3,4). Spreading 2 periods over a wide range keeps the input-space
  frequency ~1, which a small net learns (like the original sinc).
- Probed contrast (dedicated fixed-width nets, ×noise floor): easy FLAT ~1.2–2× at all widths; hard
  52×(w1)→14×(w4)→3.8×(w6)→**1.8×(w7)**→1.3×(w10). Clean gradient: easy needs 1 node, hard needs ~7.
  Learnable. (One high-width bad-seed fluke seen → use keep-best-by-train over 2–3 restarts, or +epochs,
  to stabilize high-width fits.) Use `W_max=12`.

**W2 schedule fix — slimmable SANDWICH rule** (replaces the per-sample uniform width draw in
`nested_width_net.train_nested_width`): each training step, ALWAYS train width=1 (min) AND width=W_max
(max), PLUS 2 random intermediate widths; sum the 4 Gaussian-NLL losses and step once. Guarantees w_max
is fully trained → fixes the hard-region under-fit. (Optional escalation if the hard region still
under-fits: inplace distillation — full-width `(mean, log_var)` as a soft target for the sub-widths.)

**W2 bars:** identical 3 bars/thresholds (construction/recovery/deploy). Now bar (i) `tail_flat` should
genuinely hold (easy region is a line). Expected clean pass if the sandwich rule fixes the centre fit.

**W2 execution:** dispatch ONE Sonnet `task-worker` with this §6 as the delta-spec (mirror the W1 code):
add `make_hetero` + a `hetero_width_experiment.py` driver (or `--toy hetero` flag) and the sandwich
training mode in `nested_width_net.py`; keep W1 sinc code intact. Verify selftest + smoke (hard climbs,
easy flat); run 3-seed → `W2/w2_summary.json`; interpret bars on Opus.
