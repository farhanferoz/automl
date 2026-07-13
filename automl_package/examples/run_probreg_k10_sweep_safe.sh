#!/usr/bin/env bash
# Launch the ProbReg k_max=10 sweep with the same safeguards used for the
# 840-run sweep. 120 runs (~15 min) — shorter, but still needs suspend-safety
# on this GNOME-idle-suspend box.

set -euo pipefail

REPO=/home/ff235/dev/MLResearch/automl
PY=/home/ff235/dev/.venv/bin/python
OUT_DIR="${REPO}/automl_package/examples/probreg_k10_sweep_results"
LOG="${OUT_DIR}/run.log"

mkdir -p "${OUT_DIR}"
cd "${REPO}"

if pgrep -af 'probreg_k10_sweep' | grep -v "$$" | grep -v "run_probreg_k10_sweep_safe" > /dev/null; then
    echo "ERROR: probreg_k10_sweep is already running." >&2
    pgrep -af 'probreg_k10_sweep' >&2
    exit 1
fi

echo "=== launching probreg k10 sweep at $(date -Is) ==="                  | tee -a "${LOG}"

systemd-run --user --scope \
    --property=MemoryMax=20G \
    --property=MemorySwapMax=2G \
    --property=MemoryHigh=16G \
    setsid nohup systemd-inhibit \
        --what=idle:sleep:handle-lid-switch \
        --why="probreg k10 sweep" \
        "${PY}" -m automl_package.examples.probreg_k10_sweep \
    >> "${LOG}" 2>&1 &
disown

sleep 1
echo "Launched. PID tree:"
pgrep -af probreg_k10_sweep || echo "  (not yet visible — check ${LOG} in a few seconds)"
echo ""
echo "Progress:"
echo "  tail -f ${LOG}"
echo "  wc -l ${OUT_DIR}/results.csv"
