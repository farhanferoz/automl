#!/usr/bin/env bash
# Inner chain runner for phase2 follow-ups. Called by run_phase2_followups_safe.sh
# via systemd-run; keep as its own file so env-var propagation is trivial.

set -uo pipefail

PY=/home/ff235/dev/.venv/bin/python
LOG_DIR=/home/ff235/dev/MLResearch/automl/automl_package/examples
cd /home/ff235/dev/MLResearch/automl

echo "=== CHAIN start $(date -Is) ==="
for NAME in probreg_k20_sweep probreg_snr_sweep probreg_ordering_ablation; do
    OUT="${LOG_DIR}/${NAME}_results"
    mkdir -p "${OUT}"
    LOG="${OUT}/run.log"
    echo ""
    echo "=== [$(date -Is)] launching ${NAME} ==="
    if "${PY}" -m "automl_package.examples.${NAME}" >> "${LOG}" 2>&1; then
        echo "=== [$(date -Is)] ${NAME} OK ==="
    else
        echo "=== [$(date -Is)] ${NAME} FAILED (continuing) ==="
    fi
done
echo "=== CHAIN done $(date -Is) ==="
