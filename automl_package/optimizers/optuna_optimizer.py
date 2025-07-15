from typing import Callable
import optuna


class OptunaOptimizer:
    """
    Manages hyperparameter optimization using Optuna.
    """

    def __init__(self, direction: str = "minimize", n_trials: int = 50, seed: int = 42):
        """
        Initializes the Optuna optimizer.

        Args:
            direction (str): 'minimize' or 'maximize' the objective function.
            n_trials (int): Number of optimization trials.
            seed (int): Random seed for reproducibility.
        """
        self.direction = direction
        self.n_trials = n_trials
        self.seed = seed
        self.study = None

    def optimize(self, objective_fn: Callable[[optuna.Trial], float], **kwargs) -> optuna.study.Study:
        """
        Runs the Optuna optimization.

        Args:
            objective_fn (Callable[[optuna.Trial], float]): A function that takes an Optuna trial object
                                                            and returns the metric to be optimized.
            **kwargs: Additional keyword arguments for optuna.study.Study.optimize.
                      (e.g., timeout, callbacks).

        Returns:
            optuna.study.Study: The Optuna study object containing optimization results.
        """
        self.study = optuna.create_study(
            direction=self.direction, sampler=optuna.samplers.TPESampler(seed=self.seed), pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=30)
        )

        # The objective_fn directly takes the trial object now.
        # `show_progress_bar=True` is useful for interactive sessions.
        self.study.optimize(objective_fn, n_trials=self.n_trials, show_progress_bar=True, **kwargs)
        return self.study
