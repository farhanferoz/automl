"""Optuna optimizer for hyperparameter tuning."""

from collections.abc import Callable
from typing import Any

import optuna


class OptunaOptimizer:
    """Manages hyperparameter optimization using Optuna."""

    def __init__(self, direction: str = "minimize", n_trials: int = 50, random_seed: int | None = None) -> None:
        """Initializes the Optuna optimizer.

        Args:
            direction (str): 'minimize' or 'maximize' the objective function.
            n_trials (int): Number of optimization trials.
            random_seed (int | None): Random seed for reproducibility.
        """
        self.direction = direction
        self.n_trials = n_trials
        self.random_seed = random_seed
        self.study: optuna.study.Study = None

    def optimize(self, objective_fn: Callable[[optuna.Trial], float], **kwargs: Any) -> optuna.study.Study:
        """Runs the Optuna optimization.

        Args:
            objective_fn (Callable[[optuna.Trial], float]): A function that takes an Optuna trial object
                                                            and returns the metric to be optimized.
            **kwargs: Additional keyword arguments for optuna.study.Study.optimize.
                      (e.g., timeout, callbacks).

        Returns:
            optuna.study.Study: The Optuna study object containing optimization results.
        """
        sampler = optuna.samplers.TPESampler(seed=self.random_seed) if self.random_seed is not None else optuna.samplers.TPESampler()

        self.study = optuna.create_study(
            direction=self.direction,
            sampler=sampler,
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=30),
        )

        # The objective_fn directly takes the trial object now.
        # `show_progress_bar=True` is useful for interactive sessions.
        self.study.optimize(objective_fn, n_trials=self.n_trials, show_progress_bar=True, **kwargs)
        return self.study
