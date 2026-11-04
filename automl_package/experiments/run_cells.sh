#!/usr/bin/env bash
# Runs a plan file with this repository's environment (see _env.sh).
# usage: run_cells.sh <cells.jsonl> [--shard I/N] [--workers W]
set -euo pipefail
source "$(dirname "$0")/_env.sh"
cd "$REPO"
exec "$PYTHON" -m automl_package.experiments.execute "$@"
