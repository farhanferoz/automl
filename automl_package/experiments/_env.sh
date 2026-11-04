# Sourced by the experiment scripts: sets REPO and PYTHON, this repository's own environment (.venv), built on each
# machine by `uv sync --frozen --extra <cu132 | xpu | cpu> --extra experiments`. AUTOML_PYTHON overrides it, e.g. to
# finish cells that started on an older environment (cells pair only within one environment fingerprint).
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON=${AUTOML_PYTHON:-$REPO/.venv/bin/python}
if [ ! -x "$PYTHON" ]; then
  echo "no automl environment at $PYTHON: in $REPO run uv sync --frozen --extra experiments --extra cu132 (DGX), --extra xpu (Intel laptop) or --extra cpu" >&2
  exit 2
fi
