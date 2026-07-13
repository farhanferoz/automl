#!/usr/bin/env bash
# Relaunch the ProbReg k-sweep (#48) with memory + suspend safeguards.
#
# Why: 2026-04-23 the system soft-locked at run [206/840] — shared-memory
# iGPU box, XPU allocations come from system RAM, slow accumulation across
# 840 model instantiations thrashed swap until the kernel watchdog fired.
#
# This wrapper adds:
#   - user-scope cgroup with hard memory cap  (kills only the sweep on OOM)
#   - capped swap                             (prevents the thrash-to-death)
#   - soft memory ceiling                     (kernel reclaims early)
#   - suspend/idle/lid inhibit                (survives GNOME idle-suspend)
#   - setsid + nohup + disown                 (survives terminal / Claude close)
#
# The sweep's resume logic is append-only per row, so if this process does
# get OOM-killed, just re-run this script and it picks up where it left off.

set -euo pipefail

REPO=/home/ff235/dev/MLResearch/automl
PY=/home/ff235/dev/.venv/bin/python
LOG="${REPO}/automl_package/examples/probreg_k_sweep_results/run.log"

cd "${REPO}"

# Guard against double-launch.
if pgrep -af 'probreg_k_sweep' | grep -v "$$" | grep -v "run_probreg_k_sweep_safe" > /dev/null; then
    echo "ERROR: probreg_k_sweep is already running." >&2
    pgrep -af 'probreg_k_sweep' >&2
    exit 1
fi

echo "=== launching probreg k-sweep at $(date -Is) ==="                    | tee -a "${LOG}"
echo "=== caps: MemoryMax=20G  MemorySwapMax=2G  MemoryHigh=16G ==="       | tee -a "${LOG}"
echo "=== progress: tail -f ${LOG}  |  wc -l ${LOG%run.log}results.csv === "

systemd-run --user --scope \
    --property=MemoryMax=20G \
    --property=MemorySwapMax=2G \
    --property=MemoryHigh=16G \
    setsid nohup systemd-inhibit \
        --what=idle:sleep:handle-lid-switch \
        --why="probreg k-sweep" \
        "${PY}" -m automl_package.examples.probreg_k_sweep \
    >> "${LOG}" 2>&1 &
disown

sleep 1
echo "Launched. PID tree:"
pgrep -af probreg_k_sweep || echo "  (not yet visible — check ${LOG} in a few seconds)"
