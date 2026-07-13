#!/usr/bin/env bash
# Run the three Phase-2 follow-up sweeps sequentially in one detached session.
# Sequential (not parallel) because the XPU is a single shared-memory device —
# true parallel would trigger the same swap-thrash crash we saw 2026-04-23.
#
#   1. probreg_k20_sweep    — 80 runs  (k_max=20 asymptote test)
#   2. probreg_snr_sweep    — 30 runs  (exponential × 3 noise levels)
#   3. probreg_ordering_ablation — 80 runs (Cell B × 4 ordering weights)

set -euo pipefail

REPO=/home/ff235/dev/MLResearch/automl
PY=/home/ff235/dev/.venv/bin/python
LOG_DIR="${REPO}/automl_package/examples"
CHAIN_LOG="${LOG_DIR}/phase2_followups_chain.log"

cd "${REPO}"

if pgrep -af 'probreg_k20_sweep\|probreg_snr_sweep\|probreg_ordering_ablation' | grep -v "$$" | grep -v "run_phase2_followups" > /dev/null; then
    echo "ERROR: one of the follow-up sweeps is already running." >&2
    pgrep -af 'probreg_k20_sweep\|probreg_snr_sweep\|probreg_ordering_ablation' >&2
    exit 1
fi

chmod +x "${LOG_DIR}/_phase2_followups_chain.sh"

echo "=== launching phase2-followups chain at $(date -Is) ==="     | tee -a "${CHAIN_LOG}"
echo "=== caps: MemoryMax=20G  MemorySwapMax=2G  MemoryHigh=16G ===" | tee -a "${CHAIN_LOG}"

systemd-run --user --scope \
    --property=MemoryMax=20G \
    --property=MemorySwapMax=2G \
    --property=MemoryHigh=16G \
    setsid nohup systemd-inhibit \
        --what=idle:sleep:handle-lid-switch \
        --why="phase2 followup sweeps" \
        "${LOG_DIR}/_phase2_followups_chain.sh" \
    >> "${CHAIN_LOG}" 2>&1 &
disown

sleep 1
echo "Launched. PID tree:"
pgrep -af 'phase2 followup\|probreg_k20\|probreg_snr\|probreg_ordering' || echo "  (not yet visible — check ${CHAIN_LOG})"
echo ""
echo "Progress:"
echo "  tail -f ${CHAIN_LOG}"
echo "  wc -l ${LOG_DIR}/probreg_k20_sweep_results/results.csv"
echo "  wc -l ${LOG_DIR}/probreg_snr_sweep_results/results.csv"
echo "  wc -l ${LOG_DIR}/probreg_ordering_ablation_results/results.csv"
