"""Mixin for handling the middle class NLL penalty."""

import numpy as np


class MiddleClassPenaltyMixin:
    """A mixin for models that support the middle class NLL penalty."""

    def __init__(self, use_middle_class_nll_penalty: bool = False, **kwargs: dict) -> None:
        """Initializes the MiddleClassPenaltyMixin."""
        self.use_middle_class_nll_penalty = use_middle_class_nll_penalty
        self.middle_class_dist_params_ = None

    def _calculate_middle_class_dist_params(self, y_flat: np.ndarray, y_binned: np.ndarray, n_classes: int | None = None) -> None:
        """Calculates and stores the mean and std of the y values for the middle class."""
        if n_classes is None:
            n_classes = self.n_classes

        if self.use_middle_class_nll_penalty and n_classes % 2 != 0:
            middle_class_idx = int(n_classes / 2)
            middle_class_mask = y_binned == middle_class_idx
            middle_class_y = y_flat[middle_class_mask]
            self.middle_class_dist_params_ = {"mean": np.mean(middle_class_y), "std": np.std(middle_class_y)}
        else:
            self.middle_class_dist_params_ = None
