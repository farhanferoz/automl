# Batched GPU training: where it helps (note, 2026-11-01)

Context: `docs/probreg/PLAN.md` "Speed (owner, 2026-11-01)" — stack many small fits into one GPU computation; adopt
only if faster than parallel CPU runs and the results agree.

## What batching can stack

One batched computation trains M copies of ONE network shape at once (PyTorch `torch.func.stack_module_state` +
`functional_call` + `vmap`, present in the study environment's torch 2.10). Members may differ in weights,
initialisation and training data (one draw per seed), but not in shape or in settings that change the code path.
Adam updates element by element, so one optimiser over the stacked weights equals M separate optimisers; early
stopping must be tracked per member (each member's best epoch frozen at its own stop epoch) to reproduce separate
runs.

## Where the study's cost sits

| Work | Settings per fit | Can be stacked? |
|---|---|---|
| 5a size sweep (untuned, 7 sizes x 10 seeds x 9 toys) | fixed | yes, across seeds and toys of one size |
| 5c-5 per-k sweep | fixed per k (the reference's tuned settings) | yes, across seeds and toys, per k |
| 5d final refits | fixed | yes |
| Every tuned arm (5b, 5c): 20 Optuna trials per cell | each trial's settings come from that cell's own search, one trial after another | no: shapes and settings differ between cells and trials |

The tuned arms are most of the 5a-5c cost (about 90% by the review's per-card estimates: 5b 12, 5c-1 20, 5c-2 13,
5c-4 6.7, 5c-5 9 tuned against 3.6 for the sweep and 2.5 for 5a, CPU-hours at 10 seeds), so batching would cut
tonight's chain far less than hoped. With 3 seeds per cell, a stack over seeds alone holds 3 members, which does not
beat 3 CPU cores.

## What would make it pay

- Stacking across cells for the fixed-settings work above (5a, sweeps, 5d): worthwhile once 5d (100 trials per arm,
  then fixed refits) or larger seed counts come.
- For the searches: a batched search that asks several trials at once per cell and groups them by shape (Optuna's
  `ask`/`tell` with parallel suggestions), or a search space whose shape dimensions take few values. Both change how
  the search behaves, so they need their own agreement check before adoption.
- Not supported by `vmap` (from its documented limits, not tested here): data-dependent Python control flow and
  `.item()` inside the forward/loss; ProbReg's loss paths would need checking.

## Decision taken while the owner was away (reversible)

Not built tonight: the chain runs on CPU (DGX 8 cores; the laptop's 16 cores for cards whose reference runs there
too). Batching is proposed for 5a-style fixed-settings work and 5d, to be decided with the owner.
