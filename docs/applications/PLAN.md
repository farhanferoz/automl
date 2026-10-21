# Applications — stream plan

**Status: parked** by the owner's toys-only ruling of 2026-10-20 (Q5 in `docs/probreg/PLAN.md`).
Real data, photometric redshift and astrophysics.

- `docs/applications/research_plan.md` — the April 2026 two-paper publication roadmap (status:
  draft). Partly superseded by the later capacity work; it also holds a ProbReg audit (N1–N7) and a
  FlexNN paper outline.
- `docs/applications/work_order.md` — the April 2026 queue (phase 10 photo-z, phase 11 cluster
  mass), never started.
- Code: photo-z metrics in `automl_package/utils/domain_metrics.py`, tested by
  `tests/test_photoz_domain_metrics.py`; `automl_package/examples/photo_z_domain_metrics_test.py`;
  three planned baselines never wired in, in `automl_package/models/baselines/` (`cqr.py`,
  `ngboost_model.py`, `tabpfn_model.py`).
- Real-data work inside other streams is parked with it: ProbReg task P4
  (`docs/probreg/shared/plan.md`) and the width stream's WSEL-6-R
  (`docs/flexnn/width/specs/wsel6r-realdata-selection-fraction.md`).
