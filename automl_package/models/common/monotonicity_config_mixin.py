"""Mixin for shared monotonicity configuration."""


class MonotonicityConfigMixin:
    """A mixin for models that share monotonicity configuration."""

    def __init__(self, use_monotonic_constraints: bool = False, constrain_middle_class: bool = True, **kwargs) -> None:
        """Initializes the MonotonicityConfigMixin."""
        super().__init__(**kwargs)
        self.use_monotonic_constraints = use_monotonic_constraints
        self.constrain_middle_class = constrain_middle_class
