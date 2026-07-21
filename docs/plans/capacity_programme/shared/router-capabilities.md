# Router capability classification (Task FP-5)

**Owner:** `docs/plans/capacity_programme/flexnn-package.md` FP-5. This note is the deliverable
FP-5.c's verify clause (e) checks against — every router symbol listed in `PROTECTED.tsv` under
`capacity_ladder_{k6,s1,s2,t2}.py` is classified below as `ported` or `out-of-scope`, so no
capability is silently dropped in reconciling the router (§3's binding rule).

**What changed in the package:** `automl_package/models/common/distilled_router.py`'s
`DistilledCapacityRouter` gained `fit_soft()` (soft-target cross-entropy training, alongside the
existing hard-label `fit()`) and `blend_scores()`/`blend_nll()` (blend-likelihood evaluation).
Both share the class's existing scalar/vector-generalized `_CapacityRouterMLP` — no separate
scalar code path was ever needed there (`in_dim` is inferred from `x`'s shape). No script router
(`capacity_ladder_k6.py`, `capacity_ladder_t2.py`, `capacity_ladder_s1.py`,
`capacity_ladder_s2.py`) was modified, deleted, or shimmed: none of their five importers
(`capacity_ladder_h1.py`, `capacity_ladder_s1.py`, `capacity_ladder_s2.py`,
`depth_selection_toy.py`, `sinc_width_experiment.py` — re-derived by grep, see FP-5.a) needed a
shim, because nothing at those import sites changed. The four shared router defaults
(`DEFAULT_TOLERANCE`, `DEFAULT_HIDDEN`, `DEFAULT_N_EPOCHS`, `DEFAULT_LR`) are untouched, per
FP-5.b — three sibling sensitivity studies (WSEL-7, DSEL-9, PC) have not yet produced their
tables.

## Classification

| symbol | disposition | file(s) | what "ported"/"out-of-scope" means here |
|---|---|---|---|
| `_RouterMLP` | ported | `capacity_ladder_k6.py`, `capacity_ladder_t2.py` | Subsumed by `_CapacityRouterMLP` (`distilled_router.py:97-118`, unchanged this task) — scalar input is `in_dim == 1`, not a separate class; T2's vector generalization was already the package class's starting point. |
| `_train_router` | ported | `capacity_ladder_k6.py`, `capacity_ladder_t2.py` | K6's hard-label branch was already `fit()`; its soft-target branch and T2's vector soft-target trainer are now `fit_soft()`, sharing `_fit_from_targets()` with `fit()`. |
| `_soft_targets` | out-of-scope | `capacity_ladder_k6.py` | Per-tercile EM-stacked responsibility *construction* (`quantile_bins` + `perbin_stack`) is experiment-protocol specific to K6's toy setup, not a generic router capability. `fit_soft()` trains on whatever distribution the caller builds; it does not build one. `capacity_ladder_t2._knn_soft_targets` (the kNN analog, not itself a PROTECTED row but the same shape of capability) stays out-of-scope for the identical reason. |
| `_train_router_direct` | out-of-scope | `capacity_ladder_s2.py` | PROTECTED.tsv's own reason stands: "UNRESOLVED research arm — merging it ships an unadjudicated method as a default." Porting a still-being-adjudicated training objective into the shared library is picking a winner by inspection, which this task's non-goals forbid. Stays with the driver until the S2 arm is adjudicated. |
| `ARM_NAMES` | out-of-scope | `capacity_ladder_s1.py` | The five-arm label-construction factorial (`soft`, `soft_no_prior`, `soft_smoothed`, `hard_knee`, `raw_argmax`) is a certified *comparison*; the library already implements only its winner (the cheapest-within-tolerance hard-label rule `fit()` uses). The other four arms exist only to have been compared against, not to be reused. |
| `_blend_scores` | ported | `capacity_ladder_t2.py` | Now `DistilledCapacityRouter.blend_scores()`. |
| `_blend_nll` | ported | `capacity_ladder_t2.py` | Now `DistilledCapacityRouter.blend_nll()`. |
| `_eval_router` | ported | `capacity_ladder_t2.py` | A thin combiner returning `(nll_blend, nll_hard, col_hard)`. Fully covered by `blend_nll()` (blend half) + `route_index()` (hard half, already existed); the routed NLL is a one-line caller-side reduction (`-score[rows, col].mean()`) not worth a dedicated wrapper (minimum-viable-code: this needs no separate method). |

## Non-goals honored

- No script router file was deleted or rewritten. `capacity_ladder_k6.py`'s five importers
  continue to import it directly and unchanged.
- No behavioural difference between two implementations was resolved by inspection. Where K6 and
  T2 each had their own `_RouterMLP`/`_train_router`, both are covered by the SAME package
  implementation going forward — there was no need to pick a winner because the package's
  existing vector-generalized MLP structurally subsumes the scalar case; T2's soft-target trainer
  and K6's soft-target branch are the same algorithm (soft-label cross-entropy) already, only the
  input width differed, and the package's `_fit_from_targets` is that one algorithm.
- The four shared router defaults (`DEFAULT_TOLERANCE = 0.25`, `DEFAULT_HIDDEN = (32, 32)`,
  `DEFAULT_N_EPOCHS = 300`, `DEFAULT_LR = 1e-2`) were not changed. `fit()`/`fit_soft()` accept
  `hidden`/`n_epochs`/`lr` as constructor parameters and `tolerance` as a `fit()` parameter
  already — the per-call/per-family override FP-5.b asks for already existed as constructor/call
  arguments; this task added no new global.
