#!/usr/bin/env bash
# Runs a plan file with this machine's interpreter: the DGX side environment on aarch64, the workspace venv on x86_64.
# usage: run_cells.sh <cells.jsonl> [--shard I/N] [--workers W]
set -euo pipefail
case "$(uname -m)" in
  aarch64) PYTHON=/home/ff235/cache/automl/venv-dgx/bin/python ;;
  x86_64) PYTHON=/home/ff235/dev/.venv/bin/python ;;
  *) echo "no interpreter configured for $(uname -m)" >&2; exit 2 ;;
esac
cd "$(dirname "$0")/../.."
exec "$PYTHON" -m automl_package.experiments.execute "$@"
