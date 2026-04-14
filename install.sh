#!/usr/bin/env bash
# install.sh — Detect hardware accelerator and install automl-package with the right torch backend.
#
# Usage:
#   ./install.sh              # auto-detect (CUDA > XPU > CPU)
#   ./install.sh --cpu        # force CPU
#   ./install.sh --cuda       # force CUDA
#   ./install.sh --xpu        # force Intel XPU
#   ./install.sh --dev        # auto-detect + dev tools (pytest, ruff)
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
        --dev)  DEV="[dev]" ;;
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

# --- Install ---
case "$BACKEND" in
    cpu)
        echo "Installing with CPU torch (default PyPI)..."
        uv pip install -e ".$DEV"
        ;;
    cuda)
        echo "Installing with CUDA torch..."
        uv pip install -e ".$DEV"
        # Default PyPI torch includes CUDA support when nvidia drivers are present
        ;;
    xpu)
        echo "Installing with XPU torch (Intel Arc/Flex/Max)..."
        uv pip install -e ".$DEV" \
            --extra-index-url https://download.pytorch.org/whl/xpu \
            torch torchvision torchaudio
        # triton-xpu is optional and currently has compatibility issues with torch._dynamo;
        # the package patches around this at runtime (see automl_package/__init__.py)
        uv pip install triton-xpu 2>/dev/null \
            --extra-index-url https://download.pytorch.org/whl/xpu || true
        ;;
esac

# --- Verify ---
echo ""
echo "Verifying installation..."
python3 -c "
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
