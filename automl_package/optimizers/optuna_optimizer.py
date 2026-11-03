"""Optuna optimizer for hyperparameter tuning."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import optuna
from optuna.trial import TrialState

from automl_package.enums import HpoSampler
from automl_package.logger import logger

FINISHED_STATES = (TrialState.COMPLETE, TrialState.PRUNED)
PRUNER_STARTUP_TRIALS = 5
PRUNER_WARMUP_STEPS = 30
SAMPLER_CLASSES = {HpoSampler.TPE: optuna.samplers.TPESampler, HpoSampler.RANDOM: optuna.samplers.RandomSampler}


class OptunaOptimizer:
    """Manages hyperparameter optimization using Optuna, optionally resumable from a journal file."""

    def __init__(
        self,
        direction: str = "minimize",
        n_trials: int = 50,
        random_seed: int | None = None,
        sampler: HpoSampler = HpoSampler.TPE,
        storage_path: Path | None = None,
        study_name: str | None = None,
        show_progress_bar: bool = True,
    ) -> None:
        """Initializes the Optuna optimizer.

        Args:
            direction (str): 'minimize' or 'maximize' the objective function.
            n_trials (int): Number of optimization trials (in total across resumes when ``storage_path`` is set).
            random_seed (int | None): Random seed for reproducibility.
            sampler (HpoSampler): TPE (default) or RANDOM sampling of trials.
            storage_path (Path | None): Journal file of the study. When set, finished trials are kept on disk and a new optimizer
                on the same file runs only the trials still missing. None keeps the study in memory.
            study_name (str | None): Name of the study inside the journal (default: Optuna's generated name).
            show_progress_bar (bool): Show Optuna's progress bar.
        """
        self.direction = direction
        self.n_trials = n_trials
        self.random_seed = random_seed
        self.sampler = sampler
        self.storage_path = storage_path
        self.study_name = study_name
        self.show_progress_bar = show_progress_bar
        self.study: optuna.study.Study = None

    def _storage(self) -> optuna.storages.JournalStorage | None:
        """The journal storage of ``storage_path`` (None when the study stays in memory)."""
        if self.storage_path is None:
            return None
        Path(self.storage_path).parent.mkdir(parents=True, exist_ok=True)
        return optuna.storages.JournalStorage(optuna.storages.journal.JournalFileBackend(str(self.storage_path)))

    def optimize(self, objective_fn: Callable[[optuna.Trial], float], **kwargs: Any) -> optuna.study.Study:
        """Runs the Optuna optimization.

        With a journal, trials finished (complete or pruned) by an earlier process count towards ``n_trials``; trials that process
        left running are ignored.

        Args:
            objective_fn (Callable[[optuna.Trial], float]): A function that takes an Optuna trial object
                                                            and returns the metric to be optimized.
            **kwargs: Additional keyword arguments for optuna.study.Study.optimize.
                      (e.g., timeout, callbacks).

        Returns:
            optuna.study.Study: The Optuna study object containing optimization results.
        """
        sampler = SAMPLER_CLASSES[self.sampler](seed=self.random_seed)

        self.study = optuna.create_study(
            direction=self.direction,
            sampler=sampler,
            pruner=optuna.pruners.MedianPruner(n_startup_trials=PRUNER_STARTUP_TRIALS, n_warmup_steps=PRUNER_WARMUP_STEPS),
            storage=self._storage(),
            study_name=self.study_name,
            load_if_exists=self.storage_path is not None,
        )
        finished = len(self.study.get_trials(deepcopy=False, states=FINISHED_STATES))
        stale = len(self.study.get_trials(deepcopy=False, states=(TrialState.RUNNING,)))
        remaining = max(self.n_trials - finished, 0)
        if self.storage_path is not None and (finished or stale):
            logger.info(
                f"Resuming study {self.study.study_name} from {self.storage_path}: {finished} trials finished, "
                f"{stale} stale running (ignored), {remaining} remaining of {self.n_trials}."
            )

        # The objective_fn directly takes the trial object now.
        # `show_progress_bar=True` is useful for interactive sessions.
        if remaining:
            self.study.optimize(objective_fn, n_trials=remaining, show_progress_bar=self.show_progress_bar, **kwargs)
        return self.study
