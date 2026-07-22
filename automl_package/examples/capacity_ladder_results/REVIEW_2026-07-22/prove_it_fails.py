"""Reproduces the WSEL-12 'prove-it-fails' demonstration that the width.md task report says was
run but does not show recorded output for on disk (grep of docs/plans/capacity_programme/ and
tests/test_nested_width_single_trunk.py turned up no captured failing-test output, only prose
asserting it happened). Monkeypatches an off-by-one mask mutant into SharedTrunkPerWidthHeadNet
(read-only against the repo -- patches the imported class in-process, writes nothing to disk) and
runs the actual landed pytest module against it.
"""
from __future__ import annotations

import os
import subprocess
import sys

REPO = "/home/ff235/dev/MLResearch/automl"
sys.path.insert(0, os.path.join(REPO, "automl_package", "examples"))

MUTANT_PATCH = r'''
import sys, os
sys.path.insert(0, os.path.join("{repo}", "automl_package", "examples"))
import nested_width_net as nwn
import torch

_orig = nwn.SharedTrunkPerWidthHeadNet.sampled_widths_forward

def _mutant_sampled_widths_forward(self, x, widths):
    h = self.hidden(x)
    means, logvars = [], []
    for k in widths:
        mask = torch.zeros_like(h)
        mask[:, : k - 1] = 1.0  # off-by-one mutant: drops the k-th unit
        mean = self.mean_heads[k - 1](h * mask)
        means.append(mean)
        logvars.append(torch.zeros_like(mean))
    return torch.cat(means, dim=1), torch.cat(logvars, dim=1)

nwn.SharedTrunkPerWidthHeadNet.sampled_widths_forward = _mutant_sampled_widths_forward
'''.format(repo=REPO)

conftest_path = os.path.join(os.path.dirname(__file__), "_mutant_conftest.py")
with open(conftest_path, "w") as f:
    f.write(MUTANT_PATCH)

env = dict(os.environ)
env["AUTOML_DEVICE"] = "cpu"
env["OMP_NUM_THREADS"] = "4"
env["PYTHONSTARTUP"] = ""
# Inject the mutant via -c import at collection time using pytest's -p / sitecustomize is fiddly;
# simplest robust route: run pytest with `python -c` wrapper that imports the mutant patch THEN
# invokes pytest.main so the patched class is what the test module imports.
runner = f"""
import sys
sys.path.insert(0, "{os.path.dirname(__file__)}")
import _mutant_conftest  # noqa: F401  (applies the monkeypatch before pytest collects)
import pytest
raise SystemExit(pytest.main(["{REPO}/tests/test_nested_width_single_trunk.py", "-q"]))
"""
result = subprocess.run(
    [os.path.expanduser("~/dev/.venv/bin/python"), "-c", runner],
    cwd=REPO,
    env=env,
    capture_output=True,
    text=True,
    timeout=90,
)
print("RETURN CODE:", result.returncode)
print("--- STDOUT (tail) ---")
print("\n".join(result.stdout.splitlines()[-40:]))
print("--- STDERR (tail) ---")
print("\n".join(result.stderr.splitlines()[-20:]))
