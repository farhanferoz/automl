#!/usr/bin/env bash
# Sequential orchestration of every runnable sweep, followed by a final PDF.
#
# Each sweep writes its own <name>_results/ directory (unchanged behavior);
# this wrapper only adds cross-cutting status tracking and a final aggregator
# report. Failures in one sweep do not stop later sweeps.
#
# Usage:
#   nohup bash automl_package/examples/run_all_sweeps.sh > sweeps.log 2>&1 &
#
# Env toggles:
#   ALL_DATASETS=1      — expand hpo_sweep from uci-yacht only to uci-yacht + california
#   SKIP_<NAME>=1       — skip a sweep by name (names listed below)
#                         e.g. SKIP_HPO=1 SKIP_ABLATION_STUDY=1
#   ONLY=<csv>          — run only the named sweeps (comma-separated names below)
#   TIMEOUT_HOURS=<h>   — per-sweep wall-clock cap via GNU timeout (default: 12)
#
# Sweeps (in run order):
#   identifiability      probreg_identifiability_sweep       ~1 hr
#   classreg_k           classreg_k_sweep                    moderate
#   full_benchmark       full_benchmark                      moderate
#   probreg_ablation     probreg_ablation (Paper A table)    many hours
#   flex_nn_ablation     flex_nn_ablation (Paper B table)    many hours
#   ablation_study       ablation_study (smaller probe)      moderate
#   hpo                  hpo_sweep                           overnight (ALL_DATASETS=1 → longer)

set -u
set -o pipefail

PY="${HOME}/dev/.venv/bin/python"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="automl_package/examples/sweep_runs/${STAMP}"
STATUS_FILE="${LOG_DIR}/status.tsv"
mkdir -p "${LOG_DIR}"

# TSV with columns: timestamp \t event \t sweep \t elapsed_s \t rc \t log_path
printf "timestamp\tevent\tsweep\telapsed_s\trc\tlog_path\n" > "${STATUS_FILE}"

TIMEOUT_HOURS="${TIMEOUT_HOURS:-12}"

log_status() {
    # log_status <event> <sweep> <elapsed_s> <rc> <log_path>
    printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$(date +%F_%T)" "$1" "$2" "${3:-}" "${4:-}" "${5:-}" \
        | tee -a "${STATUS_FILE}"
}

should_run() {
    local name="$1"
    if [ -n "${ONLY:-}" ]; then
        case ",${ONLY}," in *",${name},"*) return 0 ;; *) return 1 ;; esac
    fi
    local skip_var="SKIP_$(printf '%s' "${name}" | tr '[:lower:]' '[:upper:]')"
    [ "${!skip_var:-0}" = "1" ] && return 1
    return 0
}

run_sweep() {
    local name="$1"
    local module="$2"
    local env_prefix="${3:-}"

    if ! should_run "${name}"; then
        log_status "SKIP" "${name}"
        return
    fi

    local log="${LOG_DIR}/${name}.log"
    log_status "START" "${name}" "" "" "${log}"
    local t0
    t0="$(date +%s)"

    # GNU timeout guards against a single sweep consuming the entire wall-clock budget.
    # `env ${env_prefix}` is unquoted on purpose so a multi-pair env string splits.
    local rc=0
    # shellcheck disable=SC2086
    timeout "${TIMEOUT_HOURS}h" env ${env_prefix} "${PY}" -m "${module}" \
        > "${log}" 2>&1 || rc=$?

    local dt=$(( $(date +%s) - t0 ))
    if [ "${rc}" -eq 0 ]; then
        log_status "OK" "${name}" "${dt}" "0" "${log}"
    elif [ "${rc}" -eq 124 ]; then
        log_status "TIMEOUT" "${name}" "${dt}" "${rc}" "${log}"
    else
        log_status "FAIL" "${name}" "${dt}" "${rc}" "${log}"
    fi
}

log_status "RUN_START" "orchestrator" "" "" "${LOG_DIR}"

run_sweep "identifiability"   "automl_package.examples.probreg_identifiability_sweep"
run_sweep "classreg_k"        "automl_package.examples.classreg_k_sweep"
run_sweep "full_benchmark"    "automl_package.examples.full_benchmark"
run_sweep "probreg_ablation"  "automl_package.examples.probreg_ablation"
run_sweep "flex_nn_ablation"  "automl_package.examples.flex_nn_ablation"
run_sweep "ablation_study"    "automl_package.examples.ablation_study"
run_sweep "hpo"               "automl_package.examples.hpo_sweep"  "${ALL_DATASETS:+ALL_DATASETS=1}"

# Final aggregator — reads every <name>_results/ and produces one PDF + JSON.
report_log="${LOG_DIR}/final_report.log"
log_status "START" "final_report" "" "" "${report_log}"
rc=0
t0="$(date +%s)"
"${PY}" -m automl_package.examples.final_results_report \
    --status-file "${STATUS_FILE}" \
    --out-dir "automl_package/examples/final_results_report" \
    > "${report_log}" 2>&1 || rc=$?
dt=$(( $(date +%s) - t0 ))
if [ "${rc}" -eq 0 ]; then
    log_status "OK" "final_report" "${dt}" "0" "${report_log}"
else
    log_status "FAIL" "final_report" "${dt}" "${rc}" "${report_log}"
fi

log_status "RUN_END" "orchestrator" "" "" "${LOG_DIR}"

# Human-readable tail summary on stdout.
echo
echo "==================== SWEEP RUN SUMMARY (${STAMP}) ===================="
awk -F'\t' 'NR>1 && $2 !~ /^(RUN_START|RUN_END|START)$/ { printf "  %-10s  %-18s  %6ss\n", $2, $3, $4 }' "${STATUS_FILE}"
echo "Log dir: ${LOG_DIR}"
echo "Report:  automl_package/examples/final_results_report/final_report.pdf"
