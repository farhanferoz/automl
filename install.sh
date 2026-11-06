#!/usr/bin/env bash
# install.sh — Detect the hardware accelerator and build this repository's environment (.venv) with the matching torch build.
#
# Usage:
#   ./install.sh              # auto-detect (CUDA > XPU > CPU)
#   ./install.sh --cpu        # force CPU
#   ./install.sh --cuda       # force CUDA
#   ./install.sh --xpu        # force Intel XPU
#   ./install.sh --dev        # auto-detect + dev tools (pytest, ruff)
#
# On a machine with an NVIDIA GPU the libraries listed in scripts/gpu_builds.toml (LightGBM) are compiled for that GPU
# during the sync (see scripts/gpu_builds.py); the install fails if one lacks this GPU's code.
#
# Requires: uv (https://docs.astral.sh/uv/)
set -euo pipefail

BACKEND=""
DEV=""

for arg in "$@"; do
    case "$arg" in
        --cpu)  BACKEND="cpu" ;;
        --cuda) BACKEND="cuda" ;;
        --xpu)  BACKEND="xpu" ;;
        --dev)  DEV="--extra dev" ;;
        *)      echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

# --- Auto-detect if no backend specified ---
if [ -z "$BACKEND" ]; then
    if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
        BACKEND="cuda"
    elif python3 -c "import torch; assert hasattr(torch, 'xpu') and torch.xpu.is_available()" 2>/dev/null; then
        BACKEND="xpu"
    elif [ -d /opt/intel/oneapi ] || command -v sycl-ls &>/dev/null; then
        # Intel oneAPI installed — likely XPU machine even if torch isn't installed yet
        BACKEND="xpu"
    else
        BACKEND="cpu"
    fi
    echo "Auto-detected backend: $BACKEND"
fi

# --- Install: this repository's own environment (.venv) from uv.lock, with this machine's PyTorch build ---
case "$BACKEND" in
    cpu)  EXTRA="cpu" ;;
    cuda) EXTRA="cu132" ;;  # CUDA 13.2 build: needs an NVIDIA driver of the 580 series or newer
    xpu)  EXTRA="xpu" ;;    # Intel Arc/Flex/Max build, with triton-xpu
esac
cd "$(dirname "$0")"
# GPU-native builds (scripts/gpu_builds.toml): on a machine with an NVIDIA GPU, write the machine's uv configuration so this
# sync (and every later one, the weekly upgrade's included) builds the listed libraries for this GPU at the locked version.
python3 scripts/gpu_builds.py configure
REBUILD="$(python3 scripts/gpu_builds.py stale --venv .venv)"  # uv keeps an installed version, so name the ones still lacking GPU code
echo "Installing with the $EXTRA PyTorch build..."
uv sync --locked --extra "$EXTRA" --extra experiments $DEV $REBUILD
python3 scripts/gpu_builds.py check --venv .venv

# --- Verify ---
echo ""
echo "Verifying installation..."
.venv/bin/python -c "
import torch
from automl_package.utils.pytorch_utils import get_device
device = get_device()
print(f'  torch {torch.__version__}')
print(f'  device: {device}')
if device.type == 'cuda':
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
elif device.type == 'xpu':
    print(f'  GPU: {torch.xpu.get_device_name(0)}')
print('  Installation successful.')
"
