"""Proportional-odds (ordinal) output for the ProbabilisticRegressionModel classifier.

Searched first: nothing ordinal exists in the repo (grep "ordinal", "proportional", "threshold" in automl_package/models).

The classifier network emits ONE score eta(x) per row. Ordered thresholds theta_1 < ... < theta_{k-1} cut the score line into
the k ordered classes: ``P(class <= c) = sigmoid(theta_{c+1} - eta)`` and ``p_c = F_{c+1} - F_c`` with F_0 = 0, F_k = 1.
The module returns class LOG-probabilities, so a softmax over its output is the probabilities and the cross-entropy on it is
``-log p_y``: every consumer written for softmax logits works unchanged.
"""

import torch
import torch.nn as nn
import torch.nn.functional as f

MIN_CLASS_PROBABILITY = 1e-12  # floor under a class probability before the log, so a class the scores rule out gives a finite loss
MIN_CLASSES = 2  # an ordinal output needs at least two classes (one threshold)


def _softplus_inverse(value: torch.Tensor) -> torch.Tensor:
    """``log(exp(value) - 1)``, the raw value whose softplus is ``value`` (``value > 0``)."""
    return torch.log(torch.expm1(value))


class ProportionalOddsOutput(nn.Module):
    """Turns a score ``eta`` of shape ``(batch, 1)`` into class log-probabilities ``(batch, n_classes)``.

    Thresholds: ``theta_1 = first_threshold`` and ``theta_{j+1} = theta_j + softplus(raw_gap_j)`` (``n_classes - 2`` gaps), so they
    are strictly increasing whatever the values of the parameters. They start at ``logit(j / n_classes)``, ``j = 1..n_classes-1``
    (the classes are equal-share percentile bins).
    """

    def __init__(self, n_classes: int) -> None:
        """Initializes the thresholds at the logits of the equal-share cumulative fractions."""
        super().__init__()
        if n_classes < MIN_CLASSES:
            raise ValueError(f"A proportional-odds output needs at least {MIN_CLASSES} classes; got {n_classes}.")
        self.n_classes = n_classes
        cumulative = torch.arange(1, n_classes, dtype=torch.float32) / n_classes
        initial = torch.logit(cumulative)
        self.first_threshold = nn.Parameter(initial[:1].clone())
        self.raw_gaps = nn.Parameter(_softplus_inverse(initial[1:] - initial[:-1]))

    def thresholds(self) -> torch.Tensor:
        """The ordered thresholds ``theta_1 < ... < theta_{k-1}``, shape ``(n_classes - 1,)``."""
        return torch.cat([self.first_threshold, self.first_threshold + torch.cumsum(f.softplus(self.raw_gaps), dim=0)])

    def class_probabilities(self, eta: torch.Tensor) -> torch.Tensor:
        """Class probabilities ``p_c = F_{c+1} - F_c`` with ``F_j = sigmoid(theta_j - eta)``, shape ``(batch, n_classes)``."""
        cumulative = torch.sigmoid(self.thresholds().unsqueeze(0) - eta)
        padded = f.pad(cumulative, (1, 0), value=0.0)
        padded = f.pad(padded, (0, 1), value=1.0)
        return padded[:, 1:] - padded[:, :-1]

    def forward(self, eta: torch.Tensor) -> torch.Tensor:
        """Class log-probabilities, ``log(max(p, MIN_CLASS_PROBABILITY))``."""
        return torch.log(self.class_probabilities(eta).clamp_min(MIN_CLASS_PROBABILITY))


__all__ = ["MIN_CLASSES", "MIN_CLASS_PROBABILITY", "ProportionalOddsOutput"]
