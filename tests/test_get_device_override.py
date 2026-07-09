"""Direct-runnable tests for the AUTOML_DEVICE override in get_device().

Run directly (repo-wide pytest collection is broken by an omegaconf conflict):
    ~/dev/.venv/bin/python tests/test_get_device_override.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from automl_package.utils.pytorch_utils import get_device


def test_env_override_forces_cpu() -> None:
    prev = os.environ.get("AUTOML_DEVICE")
    try:
        os.environ["AUTOML_DEVICE"] = "cpu"
        assert get_device().type == "cpu", "AUTOML_DEVICE=cpu must force the CPU device"
    finally:
        if prev is None:
            os.environ.pop("AUTOML_DEVICE", None)
        else:
            os.environ["AUTOML_DEVICE"] = prev


def test_env_override_case_and_whitespace_insensitive() -> None:
    prev = os.environ.get("AUTOML_DEVICE")
    try:
        os.environ["AUTOML_DEVICE"] = "  CPU  "
        assert get_device().type == "cpu", "override must be trimmed and lowercased"
    finally:
        if prev is None:
            os.environ.pop("AUTOML_DEVICE", None)
        else:
            os.environ["AUTOML_DEVICE"] = prev


def test_empty_override_falls_back_to_autodetect() -> None:
    prev = os.environ.get("AUTOML_DEVICE")
    try:
        os.environ["AUTOML_DEVICE"] = ""
        dev = get_device()
        if torch.cuda.is_available():
            expected = "cuda"
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            expected = "xpu"
        else:
            expected = "cpu"
        assert dev.type == expected, f"empty override must auto-detect (expected {expected}, got {dev.type})"
    finally:
        if prev is None:
            os.environ.pop("AUTOML_DEVICE", None)
        else:
            os.environ["AUTOML_DEVICE"] = prev


if __name__ == "__main__":
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
            n += 1
    print(f"all {n} tests passed")
