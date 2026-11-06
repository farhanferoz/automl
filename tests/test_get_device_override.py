"""Direct-runnable tests for the AUTOML_DEVICE override in get_device().

Run directly (repo-wide pytest collection is broken by an omegaconf conflict):
    .venv/bin/python tests/test_get_device_override.py
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


def test_tree_models_follow_the_device_override() -> None:
    """AUTO follows get_device() (and so AUTOML_DEVICE); an explicit choice wins; LightGBM's AUTO takes the GPU only from gpu_min_rows."""
    from automl_package.enums import ComputeDevice
    from automl_package.models.lightgbm_model import LightGBMModel
    from automl_package.utils.pytorch_utils import resolve_compute_device

    prev = os.environ.get("AUTOML_DEVICE")
    try:
        for forced, expected in (("cuda", ComputeDevice.GPU), ("cpu", ComputeDevice.CPU), ("xpu", ComputeDevice.CPU)):
            os.environ["AUTOML_DEVICE"] = forced
            assert resolve_compute_device(ComputeDevice.AUTO) is expected, f"AUTO under AUTOML_DEVICE={forced} must give {expected}"
            assert resolve_compute_device(ComputeDevice.CPU) is ComputeDevice.CPU, "an explicit CPU must not follow the override"
        os.environ["AUTOML_DEVICE"] = "cuda"
        auto = LightGBMModel(gpu_min_rows=1000)
        assert auto.fit_device(999) is ComputeDevice.CPU, "AUTO must keep a fit below gpu_min_rows on the CPU (faster there)"
        assert auto.fit_device(1000) is ComputeDevice.GPU, "AUTO must take the GPU from gpu_min_rows"
        assert LightGBMModel(compute_device=ComputeDevice.GPU, gpu_min_rows=1000).fit_device(10) is ComputeDevice.GPU, "an explicit GPU wins"
        os.environ["AUTOML_DEVICE"] = "cpu"
        assert LightGBMModel(gpu_min_rows=1000).fit_device(10**6) is ComputeDevice.CPU, "AUTOML_DEVICE=cpu must keep AUTO on the CPU"
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
