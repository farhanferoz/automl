# X4 pre-registration — E-lane non-nested discriminator (3-seed)

**Task:** EXECUTION_PLAN §8.5 X4 / W8 (§8.2). WS1's NESTED-k surrogate (K4) is seed-fragile on
toy E (moving modes): its arbiter reads flat/partial on 2/3 seeds. The mid-June NON-nested
instrument — the **per-input held-out arbiter** (mixture-vs-best-single-Gaussian NLL advantage,
neighbour-averaged) — apparently recovered E's hump on a single fit (tercile Δ̂ read
≈ −0.018 / +0.149 / −0.026: ~0 at both x-tails, strongly positive in the middle band). W8's
hypothesis: E's nested failure is **nesting-specific** (one global component ordering cannot serve
an x-varying importance ordering), NOT generic per-input-count unidentifiability. X4 discriminates
by running the SAME non-nested arbiter on the **identical K4 E data**, across 3 seeds.

## Instrument (pinned — reuse, do not reinvent)

The June arbiter already exists: `probreg_variational_em_toy_e_hump.py::run_condition`, which
drives `probreg_variational_em_step2_perinput_arbiter` (the Step-2 held-out per-input arbiter =
mixture-vs-best-single-Gaussian advantage) and `_variational_em_perinput` (Basis-B per-input
model), and also computes the resample **gold-standard** Δ*(x) and the effective count
eff#(x)=exp(entropy of per-input weights). X4 reuses this machinery verbatim; it changes only the
DATA (to match K4) and the SEED loop (3 seeds), and records the hump per seed.

## Identical-K4-E-data requirement (load-bearing)

K4's `run_structured_toy("E", seed)` used: train `td.make_toy_e(n=1000, seed=s)`, test
`td.make_toy_e(n=2500, seed=s+500)`, for s∈{0,1,2} (constants `cl._N_TR=1000`, `cl._N_TE=2500`).
X4 MUST generate E via the same calls (same n, same seeds, same default sigma/sep params) so the
underlying samples are bit-identical to what K4's nested surrogate saw — the nested-vs-non-nested
contrast is only valid on identical data. The E_broad twin uses `td.make_toy_e_broad` the same way.

## Pre-registered bars

- **X4-recovery (the discriminator):** the non-nested arbiter's neighbour-averaged Δ̂(x) recovers
  the hump — significantly positive in the middle band [X_LO, X_HI] (where `sep_hump(x)>2`) and
  ≈0 in both tails — on **all 3 seeds**. Read per seed as (middle tercile mean Δ̂) > 0 by the
  arbiter's own two-sided band AND > both tail tercile means.
  - **3/3 recover → W8 CONFIRMED: E's nested fragility is nesting-specific.** The non-nested
    arbiter succeeds where the nested ladder fails 2/3 seeds.
  - **< 3/3 (fragility mirrors the nested failure) → W8 NOT supported:** moving-mode count is hard
    for ANY instrument; the June single-fit read was lucky. (Report which seeds fail.)
- **X4-no-false-positive:** on E_broad (variance humps, but always one mode) the arbiter stays
  flat / ≤ 0 in the middle band on 3/3 seeds — it must not credit the variance hump as bimodality.
- **Consistency (soft, not a bar):** the recovered tercile Δ̂ means are in the neighbourhood of the
  June single-fit read (−0.018/+0.149/−0.026); large divergence is noted, not failed (different
  data instance / n).

## Contrast artifact (for the write-up)

Pull K4's nested-E arbiter/knee per seed from `capacity_ladder_results/K4/k4_summary.json`
(the fragile side) and place it beside X4's non-nested read (the same 3 seeds) in the results
table — the head-to-head that settles the open question.

## Governance

3 seeds {0,1,2} (§0b). Selftest before the real read: the module's **gold-standard** Δ*(x) is a
known-answer oracle — assert it humps on E and is flat on E_broad (independent of whether the
fitted arbiter recovers it). Artifacts → `capacity_ladder_results/X4/`. Heavy run on the main
thread (`AUTOML_DEVICE=cpu`). Strictly-probabilistic: the ELBO's KL is the model's own term
(coeff 1, no tuned λ) — confirm no penalty is added. A failed bar or design-forcing surprise →
fresh-context adjudicator (§0c), never the producing session.
