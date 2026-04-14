"""AutoML Package."""

from automl_package.utils.pytorch_utils import _disable_broken_triton  # noqa: F401

# Must run before any torch._dynamo import (triggered by Adam optimizer creation).
_disable_broken_triton()
