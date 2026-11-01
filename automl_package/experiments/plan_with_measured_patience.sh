#!/usr/bin/env bash
# Chain step: sets the early-stopping patience from card 5a-2's validation curves, then plans the next cards with it.
# usage: plan_with_measured_patience.sh <source cells.jsonl> <patience json out> <study> [<study> ...] [-- <hydra override> ...]
# Waits up to 60 minutes for the source records (they may arrive by Syncthing). Hydra is installed only in the DGX environment.
set -euo pipefail
case "$(uname -m)" in
  aarch64) PY=/home/ff235/cache/automl/venv-dgx/bin/python ;;
  *) echo "plan_with_measured_patience.sh needs Hydra, installed only in the DGX environment (aarch64); this machine is $(uname -m)" >&2; exit 2 ;;
esac
if [ "$#" -lt 3 ]; then echo "usage: $0 <source cells.jsonl> <patience json out> <study> [<study> ...] [-- <hydra override> ...]" >&2; exit 2; fi
SOURCE=$1; OUT=$2; shift 2
STUDIES=()
while [ "$#" -gt 0 ] && [ "$1" != "--" ]; do STUDIES+=("$1"); shift; done
if [ "$#" -gt 0 ]; then shift; fi
cd "$(dirname "$0")/../.."
PATIENCE=$("$PY" -m automl_package.experiments.patience "$SOURCE" --out "$OUT" --wait-minutes 60 | tail -n 1)
echo "measured patience: $PATIENCE"
for STUDY in "${STUDIES[@]}"; do
  "$PY" -m automl_package.experiments.run "+study=$STUDY" "study.protocol.early_stopping_patience=$PATIENCE" "$@"
done
