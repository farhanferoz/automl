# G-WIDTH certification — adversarial re-derivation (2026-07-22)

Independent read-only re-derivation of the width architecture certification from the JSONs,
the bar-computing code, and git history. The two decisive findings (F1 curve-gate quarantine,
F2 small-data-corner fit failure) were independently spot-verified against the raw result files
by the session root before this report was recorded. Ruling on consequences rests with the user;
options are listed in docs/plans/capacity_programme/width.md SS2.

# Task

Adversarial re-derivation of the width architecture certification **G-WIDTH = PASS** (2026-07-16) — `docs/width_mse_2026-07-16/verdict_variable_width_mse.md` §10 + `docs/plans/capacity_programme/width-cert.md` gate row — from the JSONs in `automl_package/examples/capacity_ladder_results/W_KDROPOUT_CONVERGED/`, the bar-computing code, and git history. Read-only; no repo edits made.

# VERDICT: CLAUSE(S) DO NOT RE-DERIVE

The **numbers** in the certification are almost all faithful to disk (right architectures, right seeds, right values — verified file by file). What fails re-derivation is the **gate logic**: neither pre-registered clause passes under the rules as written. Clause (b) literally fails at one of its two required corners, and clause (a) passes only because a pre-registered quarantine gate was silently dropped exactly where it failed on the certified architecture. The empirical substance behind the certification (shared-trunk-per-width-heads reaches the noise floor where the K× control does; the dial separates at 12–70 SE) is well-supported by the data — the architecture choice is probably right — but "PASS per the pre-registered rule" is not what happened, and the rule's own failure branch ("escalate to user") was not taken.

# Findings, ranked by severity

## F1 (highest): Clause (a) passes only by ignoring the pre-registered curve-shape gate, which fails on both counted seeds — and not marginally

- Pre-registration, `docs/plans/width_mse_2026-07-16/EXECUTION_PLAN.md:258-259`: curve gate is read "per seed … **BEFORE the dial is read** … Fail → seed quarantined (training problem, **not dial evidence**)"; and `:87-90` "scrambled curve is a training failure — quarantine the seed"; and `:192-193` the remediation: "add seeds 3/4 if any seed is quarantined by the curve gate".
- On disk (`w_kdropout_converged_summary_shared_trunk_mse_hetero3_n2250_s0.05_wp3.json`): `curve_gate_pass=False` on **seed 0 and seed 2 — exactly the two seeds clause (a) counts**. Failures are large: seed 0 `easy_flat` at 3.57× its threshold (`m_easy_lo=0.01379` vs `1.3·m_easy_wmax=0.00386`) **and** `hard_drops_to_mid=False` (0.0926 vs 0.0786; the other seeds sit at 0.006–0.009 — a ~16× spread at starved width 6); seed 2 `easy_flat` at 2.49×. Only seed 1 passes the curve gate, and it is convergence-quarantined.
- Applied as written: clause (a) has **zero** eligible seeds. The pre-registered remediation (add seeds 3/4 on hetero3) was never run. Neither §10.1(a) nor the W1 RESULT line mentions the #2 curve-gate failures anywhere; the verdict discloses curve-gate trips only where they were marginal (independent arm, §5) or exempt (σ=0.5, §4).
- What would refute this: a reading of the plan in which the curve gate quarantines only clause-(b) dial reads and not the WP-3 noisy-easy read — but the noisy-easy bar is a dial-family read on the same `expected_width` vector (`kdropout_converged_width_experiment.py:534` feeds the same `expected_width` from `sinc_width_experiment.py:_selector_eval_mse` into both `_recovery_bar` and `_noisy_easy_bar_mse`), so "not dial evidence" covers it.

## F2: Clause (b) literally fails; the PASS is a post-hoc reinterpretation, and the pre-registered escalation branch was skipped

- Rule as written (`width-cert.md:308-311`): "(b) … fit bar passes **both** σ=0.05 corners on ≥2 trustworthy seeds each. … **FAIL on (b)** → run W8 FIRST, then **escalate to user** with both scans."
- On disk, n200/σ0.05 (`…_shared_trunk_mse_n200_s0.05_wp4c.json`): fit `pass=False` on all 3 seeds (1.5556 / 1.5236 / 1.5689 vs the 1.25 bar), `untrustworthy_seeds=[0,1,2]`. The corner fails outright.
- The gate row (`width-cert.md:332-335`) concedes the literal branch was "not triggered" because "the escalation was conditioned on n200-fail being a #2-specific signal" — **that condition does not exist in the pre-registered rule**; it was introduced in the W2 FLAG, which was written *after* the W2 data landed (post-hoc relative to data, pre-gate). The gate text calls this flag "pre-registered", which overstates it.
- The mitigating contrast IS factually verified: #3 at the same cell (`…_independent_mse_n200_s0.05_wp4.json`, `arch=independent`, `untrust=[]`) fails fit 1.3945 / 1.6410 / 2.3952 on 3/3 trustworthy seeds — so "universal small-data limit" is a true statement. But it converts the gate from a rule into an argument, and the rule reserved that judgment for the user.
- Sharper still: under the plan's own trustworthiness definition (bar 6, `EXECUTION_PLAN.md:267-268` — "widths 1 and 12 converged on every counted seed"), the n200/σ0.05 corner is **not** "0/3 trustworthy, uncertifiable": all 3 seeds have widths 1 and 12 trustworthy (untrusted widths are middles only — verified per-seed), so the corner is a countable, clean **3-seed fit FAIL**, not a void cell. The "data-starved, not certifiable" framing used the harness's stricter all-widths rule, which voids the corner rather than failing it.
- Dial half of (b) re-derives cleanly: `separation_beats_2se=True` on all 12 seed-cells of all 4 corners (recomputed; margins 12–36 SE).

## F3: The #2-specific starved-width signature was visible in certification-era data and went unreported (schedule confound, deliverable Q4)

- The sandwich schedule (`kdropout_converged_width_experiment.py:288-300`: always {1, w_max} + 2 draws from {2..11}) gives each middle width ~1/5 of steps. The gate's **fit clause reads only w_max** (trained every step) — immune. The dial clauses read the router's expected width, which depends on the starved widths' error table, but passed at 20–70 SE and survive the uniform-schedule ablation (`schedU`: fit 1.1413/1.1652/1.0871, dial still positive) — so **no gate clause outcome flips**; that part is a clean answer.
- BUT: the curve gate is precisely where starvation surfaces, and it shows a signed, architecture-specific picture: at σ=0.05, #2 fails `easy_flat` (width-2 easy-region error vs width-12) at **1.75–4.08×** threshold on 2/3 seeds of every cell (canonical: 4.08/0.79/1.75; n4000: 3.58/0.79/1.57; n200: 1.95/1.11/2.45), while #3 under the identical schedule is marginal (1.04–1.67, mostly passing). Concretely: #2's width-2 easy-region MSE at the canonical cell is 0.0068–0.0142 vs #3's 0.0029–0.0036 — the certified architecture's rarely-sampled heads sit stale against a moving shared trunk. This was detectable at certification time via the pre-registered gate that was dropped (F1), and it bears on the deploy/compute story (the router must route around #2's bad narrow widths). It should be a named limitation feeding WSEL-14.

## F4: §4 WP-4 ladder cell n200/σ0.5 is misreported in three ways

`…_independent_mse_n200_s0.5_wp4.json` vs verdict §4 footnote ("seed 1 quarantined (width-12 still descending 28% at stop)") and table entry "2/3":
- Disk quarantines **seeds 1 AND 2** (`untrustworthy_seeds=[1,2]`), not just seed 1.
- Seed 1's quarantining width is **5** (still improving, recent improvement 26.7% of best_val), not width 12; seed 1's width-12 is converged/trustworthy. (Seed 2's width-12 is the unconverged one, at 1.5% — the footnote conflates the two seeds.)
- The dial-separation failure in that cell is on **seed 0 — the only trustworthy seed** (`sep=[False,True,True]`). Under the document's own "quarantined seeds are named, never counted" discipline the cell is 0/1, not "2/3". The §7 ledger row "the width signal is robust across n∈[200,4000], σ∈[0.05,0.5]" leans on this cell. Not gate-load-bearing (σ=0.5 exempt; ladder is the #3 arm), but it is a real misreport in the robustness story. The adjacent n200/σ0.15 "3/3" also counts 2 quarantined seeds without a footnote.

## F5: Numeric misquotes and terminology errors (each verified against its JSON)

- §10.4 δ_tie=0 "near-lossless (0.0028 / 0.0028 / 0.0030)": disk (`…_shared_trunk_mse_n1500_s0.05_dsweep.json`) is **0.0032** / 0.0028 / 0.0030; seed 0 is +19% over its ~0.0027 baseline and its δ_tie ladder is non-monotonic (0.0032/0.0031/0.0038/0.0037). Matches neither arm (independent δ=0: 0.0032/0.0036/0.0033) — a wrong number, not a wrong-arm pull.
- §10.1(a) and W1 RESULT call 3.88/2.06/8.91/8.10/6.27/5.03 "mean **executed** width": they are mean **expected** widths (router-probability-weighted; `_selector_eval_mse` → `_noisy_easy_bar_mse`). Executed width is a different, hard-pick quantity. Values themselves match disk exactly.
- W2 RESULT ("dial weak on seed 0 (Δ0.44, <2·SE)" at n200/σ0.5): disk says Δ0.442 > 2·SE=0.360, `separation_beats_2se=True`. Conservative-direction error; §10.1's "4/4 on every seed" is the correct disk-faithful statement, so the two cert documents contradict each other on this cell.
- §10.5 "early-stop-disabled" for `affconf`: config is patience 40 / min_delta 2e-4 — weakened ~3×, not disabled (contrast the genuinely disabled patience=100000 longbudget run). Parameters are disclosed inline, so this is wording, not concealment.

## F6: Margin-vs-noise audit (deliverable Q3)

- Solid passes: fit at n4000/σ0.05 (1.044/1.088/1.063, ≥0.16 below the 1.25 bar vs 0.044 seed spread); all dial separations (12–70 SE); noisy-easy stays-narrow (3–4 width-units of slack).
- Solid fails: n200/σ0.05 fit (+0.27 above bar); #2 `easy_flat` (1.75–4.1× threshold).
- Decided inside noise: **seed 3's fit pass in the "5/5" headline** — 1.2274 vs bar 1.25, margin 0.023 against a cross-seed spread of ~0.17. Its inclusion is bar-6-compliant (widths 1 and 12 both trustworthy; only width 4 unconverged — verified), but the honest headline is "4 comfortable + 1 borderline", not an unqualified 5/5. Also inside noise: WP-3 independent curve trips (disclosed in §5) and the n200/σ0.5 seed-0 dial (exempt cell).

## F7: Post-certification regeneration of cited evidence files (provenance)

`w_kdropout_converged_summary_shared_trunk_mse.json`, `…_rhhalf.json`, `…_rhx2.json` were regenerated 2026-07-21 (commit `1d940a3`, a recorded provenance/repair commit). Cert-era versions are recoverable at commit `bb7e9dc` and **match the cert's quotes** (fit identical to 4 d.p.; §2's dial numbers +2.922±0.111 / +4.308±0.077 / +1.359±0.103 match only the `_prewsel12` backup / `bb7e9dc`, not the current file, whose router redraw shifted dials by ~0.1). W6 invariance re-derived from the cert-era files: fit bit-identical across router sizes (trivially — same trained net), hard-pick MSE 0.0030–0.0040 across variants vs cross-seed spread ~0.001 — "within seed noise" holds. No wrongdoing; just note the §8/§10.6 manifest paths no longer point at cert-era bytes.

# The IndependentWidthNet lead (deliverable Q5): CONFIRMED as history, and the certification handled it correctly

- At commit `27c7159` (the code that produced the Jul-11 `w_kdropout_converged_summary.json`, which has **no arch key**), the harness hardcoded `net = nwn.IndependentWidthNet(w_max=w_max)` (line 164; docstring line 19). The 2026-07-13 synthesis doc labels that same file "Shared-nested, k-dropout, converged" (`docs/width_dial_synthesis_2026-07-13/per_input_width_architecture_readthrough.md:751`) — the mislabel, confirmed.
- The certification is **clean on this**: verdict §7 row 1 openly retracts the mislabel; every file cited in §8/§10.6 carries explicit correct `arch`/`loss`/`schedule`/`seeds` config (full provenance sweep run — all 21 files correct); the unlabeled Jul-11 file is cited nowhere in the certification. The shared/nested architectures genuinely could not have been run before the `--arch` split existed.

# What re-derived cleanly

- §2 three-arch battery: every number (nested 5.8592/5.6434/3.7242 fail, 8/8/6 trustworthy; shared_trunk 1.0893/1.0609/1.0775 pass, 12/12; independent 1.0375/1.0057/1.1887 pass; all dial ±SE values) matches disk with correct provenance.
- §2.1 120k nested confirmation: patience=100000 (genuinely disabled), cap hit; trajectory global mins **0.0756 / 0.0627 / 0.0710** and seed 1's best at epoch **104500** recomputed exactly; 10k–120k window stays in [0.063, 0.13] — "no late breakthrough" holds. (The JSON's `still_improving_at_stop=True` is a degenerate-window artifact — `convergence.py:140-141` falls back to first-to-last when patience > trajectory length — correctly ignored by the cert.)
- Clause-(a) raw numbers, clause-(b) dial (4/4, every seed), n4000 fit, the #3 n200 contrast values, 5-seed values, schedU / wmax24 / wmax48 (incl. the honestly-reported seed-0 dial inversion −3.831) / dsweep baselines (val 0.00270/0.00265/0.00282 vs test 0.00269/0.00265/0.00282) / δ_tie frontier (7.93→4.42, 0.0032→0.0037) / affine seam (1.2047/1.5606/1.3828; affconf 1.3232/1.4268/1.3828, 10/12) — all match.
- §6 deploy: compute_saved 12/12 cells + 3/3 both arms; accuracy beats-by-2SE 0/12 (scarce-cell hypothesis honestly refuted); the specific quoted figures (0.0033–0.0035 vs 0.0026–0.0028; exec 3.9–5.9 vs k 7–9; n1500/σ0.5 exec mean 1.8 vs best-k mean 8.3) all recompute.
- No data leakage: net trains on even half of train, router on odd half, bars on a fresh test draw (`kdropout_converged_width_experiment.py:377-380`); the hindsight-baseline issue was disclosed and closed by W7.

# Uncertainty

- Whether the curve gate's quarantine was *intended* to cover the WP-3 noisy-easy read is an interpretation of plan prose; my reading (it does — same expected-width vector, and bar 2 says "not dial evidence") is the natural one but the plan never enumerates which bars a curve-quarantine voids. It unambiguously covers clause (b)'s dial reads either way.
- The curve gate is computed on TEST, not "slice-B curves" as pre-registered (`sinc` line 485 vs plan :258) — I could not recompute the slice-B version without re-running nets, but failures at 1.75–4× threshold will not flip on a half-swap.
- Coverage note: this review verified the certification against its own evidence; it cannot certify that no *other* certification-era runs were omitted from the record.

# Recommendation: fix-first (documentation + one cheap pre-registered remediation; the architecture itself likely survives)

1. Amend §10.1/width-cert to state plainly: (b) failed as written at n200/σ0.05 and was passed by the #3-contrast argument (post-hoc relative to the rule); (a) was read from seeds the pre-registered curve gate quarantines; the user escalation the rule required did not happen. Have the user ratify the reinterpretation — that is exactly the decision the rule reserved for them.
2. Run the pre-registered remediation for clause (a): #2 on hetero3, seeds 3/4 (cheap, ~90 s/arm) — the canonical-cell precedent (seeds 3 and 4 both pass the curve gate) suggests it will pass cleanly and would restore the clause honestly.
3. Correct the specific misreports: §4 n200/σ0.5 footnote and table, §10.4 δ_tie=0 seed 0 (0.0032), "executed"→"expected" width in §10.1(a)/W1, the W2 "<2·SE" line, and `width.md:139-141`'s summary sentence "fit-at-floor at both σ=0.05 corners — both passed", which is **false as written** and is the form downstream consumers now read.
4. Record the #2 starved-width `easy_flat` signature (F3) as a named limitation of the architecture of record; route it into WSEL-14's schedule work.

Key paths: `/home/ff235/dev/MLResearch/automl/docs/width_mse_2026-07-16/verdict_variable_width_mse.md`, `/home/ff235/dev/MLResearch/automl/docs/plans/capacity_programme/width-cert.md`, `/home/ff235/dev/MLResearch/automl/docs/plans/capacity_programme/width.md`, `/home/ff235/dev/MLResearch/automl/docs/plans/width_mse_2026-07-16/EXECUTION_PLAN.md`, `/home/ff235/dev/MLResearch/automl/automl_package/examples/capacity_ladder_results/W_KDROPOUT_CONVERGED/`, `/home/ff235/dev/MLResearch/automl/automl_package/examples/sinc_width_experiment.py`, `/home/ff235/dev/MLResearch/automl/automl_package/examples/kdropout_converged_width_experiment.py`, `/home/ff235/dev/MLResearch/automl/automl_package/utils/convergence.py`; git refs `27c7159`, `bb7e9dc`, `1d940a3`.