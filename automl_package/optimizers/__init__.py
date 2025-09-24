"""Factory for creating optimizer wrappers."""

from automl_package.enums import OptimizerType

from .base import OptimizerWrapper
from .standard_optimizers import AdamWrapper, HessianFreeWrapper


def get_optimizer_wrapper(optimizer_type: OptimizerType) -> OptimizerWrapper:
    """Returns an instance of the specified optimizer wrapper."""
    if optimizer_type == OptimizerType.ADAM:
        return AdamWrapper()
    if optimizer_type == OptimizerType.HESSIAN_FREE:
        return HessianFreeWrapper()
    raise ValueError(f"Unknown optimizer type: {optimizer_type}")
