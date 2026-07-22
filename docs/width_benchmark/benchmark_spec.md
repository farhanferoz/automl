# WSEL-9 real-data benchmark spec

Freezes the real-data comparison `docs/plans/capacity_programme/width.md` WSEL-9 asks for: the
three §1 width-choosing models (W-SHARED, W-PERINPUT, W-SWEEP) against the baseline set the task
names (LightGBM, a plain single-output NN, linear regression), on real regression datasets. This
document is itself a deliverable of that task ("write the spec first"); the driver
(`automl_package/examples/width_wsel9.py`) implements exactly what is frozen here and reads its
own numeric constants from the artifacts cited below rather than hardcoding a copy of them.

Non-goals of this document and of WSEL-9 itself: no report writing (WSEL-10, gated behind the
joint results review); no re-tuning of the selector, the router, or the width ladder (all three
are frozen constants owned by earlier tasks, read at run time — see "Constants read" below); no
changes to `routing.py`, `capacity_accounting.py`, or any package module.

## 1. Datasets

Five real regression datasets, split small/medium by sample count, chosen so every one loads
today in `~/dev/.venv` with no code changes — verified by actually importing and loading each one
(2026-07-22, `sklearn==1.8.0`) before freezing it here, per the dataset rule this task was given:
never freeze a dataset that could not be loaded. Four of the five loaders are the SAME UCI
loaders `automl_package/examples/full_benchmark.py` already uses (`_uci_yacht`/`_uci_energy`/
`_uci_kin8nm`/`_uci_california`) — reused, not re-derived; `diabetes` is new, added because it is
bundled with scikit-learn (no network fetch at all), which makes it the most robust choice for
`--selftest`-adjacent smoke checks and for any offline re-run.

| dataset | tier | n | d | target | source / load call |
|---|---|---:|---:|---|---|
| `diabetes` | small | 442 | 10 | quantitative disease-progression score, one year post-baseline | `sklearn.datasets.load_diabetes()` — bundled, no network |
| `yacht` | small | 308 | 6 | residuary resistance per unit displacement | `sklearn.datasets.fetch_openml(name="yacht_hydrodynamics", version=1, as_frame=False, parser="auto")`; `d.data`/`d.target` used directly (`full_benchmark.py::_uci_yacht`) |
| `energy` | small | 768 | 8 | heating load (y1) | `fetch_openml(name="energy-efficiency", version=1, as_frame=False, parser="auto")`; `x = d.data[:, :8]` (this OpenML mirror folds the SECOND target, cooling load `y2`, into `data` as a 9th column — verified via `d.feature_names[-1] == "y2"` — dropped, not used as a feature), `y = d.target` (`full_benchmark.py::_uci_energy`) |
| `kin8nm` | medium | 8192 | 8 | joint-angle-dependent forward kinematics of an 8-link robot arm | `fetch_openml(name="kin8nm", version=1, as_frame=False, parser="auto")` (`full_benchmark.py::_uci_kin8nm`) |
| `california` | medium | 20640 | 8 | median house value (block group) | `sklearn.datasets.fetch_california_housing()` — cached locally at `~/scikit_learn_data/cal_housing_py3.pkz` (`full_benchmark.py::_uci_california`) |

Load-verification results (this task, 2026-07-22, before freezing): all five loaded successfully
with the shapes above (`diabetes (442, 10)`, `yacht (308, 6)`, `energy (768, 9)` raw → `(768, 8)`
after the y2 drop, `kin8nm (8192, 8)`, `california (20640, 8)`); `energy`/`yacht`/`kin8nm` fetch
from OpenML (network, attempted once each, succeeded — OpenML's own cache directory
`~/scikit_learn_data/openml/` now holds them for any offline re-run); `california` reused an
already-cached archive; `diabetes` never touches the network. No dataset in this candidate set
failed to load, so none was dropped.

Root may subsample the `medium` tier's FIT/STOP/SELECT pool for wall-clock reasons via the
driver's `--max-train` flag (never touches TEST); this is a recorded, provenance-tracked deviation
from the frozen default (no cap), not a silent one — see `config.max_train` in every cell's JSON.

## 2. Split protocol

One split per (dataset, seed), generalizing the toy protocol every sibling WSEL width driver uses
(`width_wsel6._build_split`'s p1/p2 carve) to arbitrary real (X, y) with no fixed toy generator:

1. Load `(x_full, y_full)`, shuffle once with `numpy.random.default_rng(seed)`.
2. **TEST** = the first `round(0.2 * n)` shuffled rows — held out, touched by nothing else
   (REPORT split). `TEST_FRACTION = 0.2` is this driver's own choice (no upstream artifact owns a
   real-data test fraction); stated here as the frozen value, not re-derived per dataset.
3. The remaining ~80% ("pool") is carved exactly like every other WSEL driver's p1/p2 split:
   even-indexed rows → **p1** (trains + monitors), odd-indexed → **p2** (the selection pool).
4. **p1** is further split via `converged_width_experiment.VAL_EVERY` (every 5th p1 row is held
   out): the rest is **FIT** (gradient training for every model), the VAL_EVERY subset is **STOP**
   (early-stopping monitor only, for every NN-based arm).
5. **p2** is shuffled once more (seeded), then subsampled to the first
   `round(fraction_pct / 100 * n_pool)` rows — **SELECT** — where `fraction_pct` is READ from
   WSEL-6's frozen selection-set-fraction artifact (see "Constants read" below), never hardcoded.
   SELECT feeds `fit_global_selector` (W-SHARED), `fit_router` (W-PERINPUT), and W-SWEEP's
   held-out error table — the SAME pool for all three, per §1's "same seeds throughout" rule.

FIT/STOP/SELECT/TEST are disjoint; TEST is used by nothing but final scoring, matching WSEL-8's
"the reported numbers come from a split not used for stopping or selection" rule.

## 3. Standardization

Every feature is z-scored (`(x - mean) / std`, `mean`/`std` fit on FIT only, std floored at 1 to
avoid divide-by-zero on a constant column), applied identically to FIT/STOP/SELECT/TEST, and
identically to every one of the six arms — this is provably harmless for the two non-NN baselines
(LightGBM's tree splits depend only on each feature's within-column ORDER, which a positive-scale
per-feature affine transform never changes; ordinary least squares' predictions are invariant to
any invertible affine reparameterization of X, standardization included) and is the standard,
necessary choice for the four gradient-trained NN arms (the dial net, W-SWEEP's 12 dedicated nets,
and the plain-NN baseline).

The target `y` is z-scored the same way (mean/std fit on FIT) **only for the four NN-based arms**
— gradient training benefits from a target on an activation-friendly scale; predictions are
inverse-transformed back to the dataset's own units before any MSE is computed, so every arm's
`held_out_mse` is comparable in one common (original) unit. LightGBM and the linear-regression
floor train directly on raw `y` — trees are shift/scale-invariant on the target for point
prediction, and OLS reproduces textbook raw-scale predictions exactly (again, affine invariance),
so standardizing `y` for those two would add an inverse-transform step for no numerical benefit.

## 4. Variance / objective — §3.7 does not transfer as written; here is what does

`width.md` §3.7 fixes `sigma` at the toy generator's TRUE, known per-point noise value. Real
datasets carry no such generator — there is no ground truth to clamp `sigma` to. The applicable
case is §3.7's own Tier-1 rule restated for the case where the (unknown, constant-vs-heteroscedastic)
noise level is simply not knowable: **plain MSE**, which is the fixed-constant-sigma Gaussian
likelihood up to one positive, unknown scale factor — exactly the objective every certified
tier-1 (`hetero`) run already uses. Consequence, mechanical, no discretion:

- Every NN-based arm (dial net, W-SWEEP's dedicated nets, plain-NN baseline) trains on plain MSE.
  No `log_var` head is ever exercised: the certified `SharedTrunkPerWidthHeadNet`'s `log_var` is a
  dummy zero excluded from the loss by construction (`architectures.py:179-181`), and the plain-NN
  baseline is built with `UncertaintyMethod.CONSTANT`, which never allocates a variance output
  (`neural_network.py`'s `_PyTorchNNModule`: the output width only doubles under `PROBABILISTIC`).
- LightGBM trains under `UncertaintyMethod.CONSTANT` too (plain `"regression"` / RMSE objective,
  not the Gaussian-NLL objective path that class also supports) — no learned variance there either.
- `width_candidates.weighted_squared_error` (§3.7's tier-2/3 mechanism, reading a per-point true
  sigma from a toy's `region` output) does not apply here — there is no `region` label and no true
  sigma on real data — and is not imported by this driver.

## 5. Constants read from their artifacts — fail loudly if missing

Binding per the task brief: the driver reads every constant below at startup and exits non-zero
with a clear message if the owning artifact is missing, rather than silently substituting a
default. Every results JSON records a `constants` key naming exactly what it read.

| Constant | Owning artifact | What the driver does with it |
|---|---|---|
| Selection-set fraction (`fraction_pct`) | `automl_package/examples/capacity_ladder_results/WSEL6/frozen.json` | Sizes the SELECT subsample (§2 step 5). Missing → `SystemExit`. |
| Width ladder / `w_max` | any per-cell JSON under `automl_package/examples/capacity_ladder_results/WSEL8/` (its own `w_max` field) | Fixes the width grid `1..w_max` for the dial net and the 12 W-SWEEP nets. Missing → `SystemExit`. |
| Router hidden/depth/epochs/lr | `automl_package/examples/capacity_ladder_results/WSEL7/frozen.json` | **Not overridden** — see below. Read only to verify, at startup, that its recorded `config.frozen_default_at_authoring_time` still matches `routing.py`'s shipped `DEFAULT_HIDDEN`/`DEFAULT_N_EPOCHS`/`DEFAULT_LR` (a drift guard); missing artifact or a mismatch → `SystemExit`. |
| Per-model selection cost | `automl_package/utils/capacity_accounting.py` (WSEL-5's module; import, not a JSON) | Imported directly (`global_cheap_cost`/`per_input_cost`/`sweep_cost`/`held_out_read_cost`); an import failure is the fail-loud signal — there is no separate numeric artifact to read for this row. |

**Why the router constant is read-but-not-overridden.** `width.md` WSEL-7's 2026-07-22 sign-off
ruling 1 is explicit: *"the frozen default STAYS (32×2, 300 epochs, lr 0.01). `new_default` is NOT
adopted."* `FlexibleWidthNN.fit_router()` never takes hidden/epochs/lr arguments at all — it always
constructs a `DistilledCapacityRouter` at that class's own constructor defaults
(`routing.py:57-60`), so calling `model.fit_router(x_sel, y_sel)` plainly already IS "the frozen
router defaults as-is." The startup read exists to catch the one way this could go stale: if
`routing.py`'s shipped constants ever change without a fresh ruling, the drift guard above stops
the driver rather than silently running under an un-ratified router.

## 6. The three width-choosing models (§1)

All three are read off `automl_package.models.flexnn.width.model.FlexibleWidthNN`, `input_size=d`
(the dataset's own feature count), `widths=1..w_max` (§5), Tanh activation (this strand's
established width-net convention — every sibling WSEL width driver uses it, for the confound-doctrine
reason `width_wsel4.py` recorded: the reference classes it reproduces are Tanh, and an earlier
ReLU run under-fit badly), trained via the established `_fit_single` bypass (matching
`width_wsel4._train_ported_default`) with `learning_rate=0.01`, `early_stopping_rounds=60`,
`min_delta=1e-4` (all three: `width_wsel4.PORTED_LR_DEFAULT`/`PORTED_PATIENCE`/`PORTED_MIN_DELTA`,
reused verbatim, cited not re-derived) and a mini-batch size (`min(256, n_fit)`) — the ONE
deliberate deviation from `width_wsel4`'s own full-batch protocol, justified because full-batch
was that task's confound-removal fix against a SPECIFIC historical toy reference that has no
counterpart here; there is nothing for real data to stay confound-free against except the other
five arms in the SAME cell, and mini-batching is applied identically to every NN-based arm, so it
introduces no difference between them. Epoch cap defaults to
`width_wsel4.PORTED_N_EPOCHS_CAP` (6000), CLI-overridable (`--epoch-cap`) for wall-clock reasons on
the larger datasets.

- **W-SHARED**: ONE multi-head net (`widths = 1..w_max`), trained by a per-width
  simultaneous-convergence loop (this driver's generalization of
  `width_wsel8._train_shared_to_convergence` to `input_size=d`; that function hardcodes
  `input_size=1` for its scalar toy — the per-width stop rule itself, one `ConvergenceTracker` per
  width, stop only once every width is simultaneously trustworthy-or-diverged AT THE SAME
  checkpoint, is copied verbatim because `FlexibleWidthNNModule.forward` sums every configured
  width's loss every step regardless of `d`, so the same aggregate-scalar blind spot applies
  unchanged). `fit_global_selector(x_select, y_select)` then picks ONE width for the dataset.
- **W-PERINPUT**: read off the SAME trained multi-head net (§1's single-difference rule — training
  is not a variable between W-SHARED and W-PERINPUT). `fit_router(x_select, y_select)` — no
  hyperparameter override (§5) — labels and trains the distilled router, then `predict()` routes
  per input.
- **W-SWEEP**: 12 dedicated single-width `FlexibleWidthNN(widths=(k,))` nets (one CLI invocation
  each), each trained under the identical protocol above; their held-out error table on SELECT
  feeds the SAME `cheapest_within_tolerance` selector W-SHARED uses (§1: "the same rule applies to
  W-SWEEP's curve"), picking ONE width for the dataset independently of the dial net.

`width.md` MASTER Decision 14 ("the known-good arm runs first, alone, and must reproduce before
any new number is read") does not bind here as a hard CLI gate the way it did for WSEL-4/WSEL-8:
those tasks were reproducing a specific historical reference on a specific toy; there is no
historical real-data reference for these models to reproduce first. What DOES bind, per the task
brief, is Decision 9 (trajectory discipline) and Decision 16 (optimization-first): every arm below
records its full held-out trajectory and a trajectory-verified convergence flag, and the
escalation ladder (LR sweep → clipping → warmup → init scheme) is exhausted before any arm that
looks like it lost is called an architecture loss rather than an optimization one.

## 7. Baseline set

Per the task brief's own rationale for each:

- **LightGBM** (`automl_package.models.lightgbm_model.LightGBMModel`) — a tree model with no
  architecture-vs-input-size problem at all; native regularization (early stopping, leaf/depth
  limits, shrinkage). Configuration reused verbatim from the ALREADY-ESTABLISHED convention in
  this same codebase, `full_benchmark.py`'s `common_tree`/model-factory lines (not re-derived):
  `n_estimators=200`, `early_stopping_rounds=15`, trained via the same `_fit_single(x_train,
  y_train, x_val=x_stop, y_val=y_stop)` bypass every sibling WSEL driver uses, which returns
  `(best_iteration, loss_history)` — a genuine per-boosting-round held-out RMSE trajectory,
  replayed through the same `ConvergenceTracker` machinery (patience = its own
  `early_stopping_rounds`, `min_delta=0.0` to mirror LightGBM's own "no strict improvement" native
  stopping rule exactly) for the trajectory/convergence fields Decision 9 requires. `hit_cap` =
  `best_iteration >= n_estimators` (LightGBM's own early stopping never triggered).
- **A plain single-output NN** (`automl_package.models.neural_network.PyTorchNeuralNetwork`) — the
  key control: one ordinary hidden layer sized at `hidden_size = w_max` (the SAME capacity as the
  dial net's widest configured head — "the dial network at fixed width ≈ this"), Tanh, trained
  under the IDENTICAL protocol as the width models (§6: same LR/batch/patience/epoch-cap/min-delta,
  the only difference being the absence of any dial/masking/selection machinery around it) via the
  same `_fit_single` bypass, `UncertaintyMethod.CONSTANT` (no variance head, §4), replayed through
  `width_wsel4._replay` for its trajectory/convergence verdict — this is what makes it a genuine
  single-difference control against W-SHARED/W-PERINPUT/W-SWEEP rather than a confounded one.
- **Linear regression** (`automl_package.models.normal_equation_linear_regression
  .NormalEquationLinearRegression`, `l2_lambda=0.0` — plain, unregularized OLS) — the floor, which
  makes an essentially-linear dataset legible instead of reading as an uninformative tie against
  the NN-based arms. `_fit_single` returns `(1, [])` by construction (`normal_equation_linear_
  regression.py:51-86`: a direct closed-form solve, no iterative loop) — genuinely no trajectory
  to record, not a gap in this driver's discipline; recorded as `trajectory_applicable: false`,
  `trustworthy: true` (there is nothing for a closed-form solve to fail to converge on), never a
  fabricated per-epoch curve.

## 8. Selection-cost accounting

Per §1's "EACH MODEL IS THE COMPLETE SYSTEM, INCLUDING ITS SELECTION MACHINERY" rule (no
companion-field pattern, WD5): every width model's number is priced end-to-end via
`automl_package.utils.capacity_accounting` (WSEL-5):

- **W-SHARED** → `global_cheap_cost(training_macs, net, capacity_grid=1..w_max, n_samples=|SELECT|)`.
- **W-PERINPUT** → `per_input_cost(training_macs, in_dim=d, n_capacities=w_max, n_samples=|SELECT|,
  n_epochs=router.n_epochs, hidden=router.hidden)` — `training_macs` is the SAME number
  `global_cheap_cost` uses (§1: "training is not a variable between W-SHARED and W-PERINPUT" —
  read off the one net both share), `in_dim=d` (not 1 — this is the one place the real-data
  dimensionality enters the cost formula directly, since the router's own first layer scales with
  the dataset's feature count).
- **W-SWEEP** → training cost is the sum of each dedicated net's own training MACs (12 independent
  trainings, no shared trunk to amortize) plus `held_out_read_cost` on SELECT for its own curve.

The three baselines carry `selection_cost: null`. This is not the forbidden companion-field
omission — there is no selection mechanism to hide beside their number: each of the three IS one
fixed thing (a fixed hyperparameter-configured LightGBM run, a fixed-architecture NN, a fixed
closed-form OLS solve), never a system that chooses among candidates, so WSEL-5's module (built,
by its own doctrine, for "every family that draws from the same capacity-dial module" — width,
depth, joint, MoE) has nothing to price for them. Registering `LightGBMModel`/`PyTorchNeuralNetwork`
against `capacity_accounting`'s `executed_flops`/`param_count` dispatch is out of this task's
scope (it would touch files outside WSEL-9's write set for no question this task asks).

## 9. Output schema

**Per-cell JSON** (`automl_package/examples/capacity_ladder_results/WSEL9/<dataset>_<seed>_<arm>[_<width>].json`),
common fields on every arm:

```
{
  "dataset": "<dataset>", "seed": <int>, "arm": "<w_shared|w_perinput|w_sweep|lightgbm|plain_nn|linear_reg>",
  "width": <int|null>,                 # W-SWEEP cells and, post-selection, w_shared/w_sweep's chosen width
  "held_out_mse": <float>,             # on TEST, ORIGINAL y units, every arm (see SS3)
  "held_out_trajectory": [[epoch_or_round, val_loss], ...],
  "trustworthy": <bool>, "hit_cap": <bool>, "converged": <bool>,
  "trajectory_applicable": <bool>,     # false only for linear_reg (SS7)
  "selection_cost": {"training_macs": <int>, "selection_macs": <int>, "total_macs": <int>} | null,
  "selection": {"fraction_pct": <int>, "n_selection_used": <int>, "fraction_source": "<path>"} | null,
  "config": {...},                     # every hyperparameter this cell ran under
  "constants": {"selection_fraction": {...}, "width_ladder": {...}, "router": {...}},  # SS5, always present
  "provenance": {...}                  # automl_package.utils.run_provenance.run_provenance()
}
```

The `dial` CLI arm writes ONE JSON carrying BOTH `w_shared` and `w_perinput` results (they share
one trained net, §6) — see the driver's own `--arm dial` schema note for the exact keys.

**Per-dataset CSV** (`WSEL9/<dataset>_summary.csv`, written by `--summarize --dataset <dataset>`):
one row per (seed, arm) — `dataset,seed,arm,width,held_out_mse,training_macs,selection_macs,
total_macs,trustworthy,hit_cap,converged,n_selection_used` — `training_macs`/`selection_macs`/
`total_macs`/`n_selection_used` are blank for the three baselines (§8), not zero (zero would claim
a real, priced-at-nothing mechanism; blank states plainly that none exists).

## 10. CLI shape

One cell per invocation (the root runs the grid, never this driver):

```
AUTOML_DEVICE=cpu OMP_NUM_THREADS=4 ~/dev/.venv/bin/python automl_package/examples/width_wsel9.py \
    --dataset yacht --seed 0 --arm w_sweep --width 5
AUTOML_DEVICE=cpu OMP_NUM_THREADS=4 ~/dev/.venv/bin/python automl_package/examples/width_wsel9.py \
    --dataset yacht --seed 0 --arm dial
AUTOML_DEVICE=cpu OMP_NUM_THREADS=4 ~/dev/.venv/bin/python automl_package/examples/width_wsel9.py \
    --dataset yacht --seed 0 --arm lightgbm
AUTOML_DEVICE=cpu OMP_NUM_THREADS=4 ~/dev/.venv/bin/python automl_package/examples/width_wsel9.py \
    --dataset yacht --seed 0 --arm plain_nn
AUTOML_DEVICE=cpu OMP_NUM_THREADS=4 ~/dev/.venv/bin/python automl_package/examples/width_wsel9.py \
    --dataset yacht --seed 0 --arm linear_reg
AUTOML_DEVICE=cpu ~/dev/.venv/bin/python automl_package/examples/width_wsel9.py --summarize --dataset yacht
```

`--selftest` runs every arm end-to-end on a tiny synthetic array (not a real dataset — fast,
deterministic, offline) plus the constants-drift guard's own pass path, then exits.

## 11. Non-goals

No real-grid execution (the root's wave-E job, not this authoring task's); no HPO/Optuna search
for any arm (every model here is a frozen, cited configuration — matching this whole file
family's established convention, not a search); no report writing (WSEL-10); no changes to
`routing.py`, `capacity_accounting.py`, `width_wsel4.py`, `width_wsel6.py`, `width_wsel8.py`, or
any plan document; no tolerance-sweep (MASTER Decision 18); no dataset outside the five frozen
above without repeating the load-verification step first.
