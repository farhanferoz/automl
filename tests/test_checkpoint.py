"""Pausable runs: the Optuna journal resumes a search, and UnitStore keeps finished units under their config hash."""

from pathlib import Path

import pandas as pd
import pytest
from optuna import Trial
from optuna.trial import TrialState

from automl_package.optimizers.optuna_optimizer import OptunaOptimizer
from automl_package.utils.checkpoint import ConfigConflictError, UnitStore, stable_hash

FIRST_TRIALS, ALL_TRIALS = 3, 5


def test_search_resumes_from_its_journal(tmp_path: Path) -> None:
    calls: list[int] = []

    def objective(trial: Trial) -> float:
        calls.append(trial.number)
        return (trial.suggest_float("x", -5, 5) - 1) ** 2

    journal = tmp_path / "search.journal"
    OptunaOptimizer(n_trials=FIRST_TRIALS, random_seed=0, storage_path=journal, study_name="s").optimize(objective)
    assert calls == [0, 1, 2]
    # a trial a killed process left running is neither counted nor best
    killed = OptunaOptimizer(n_trials=FIRST_TRIALS, random_seed=0, storage_path=journal, study_name="s")
    killed.optimize(objective)
    killed.study.ask().suggest_float("x", -5, 5)
    calls.clear()
    study = OptunaOptimizer(n_trials=ALL_TRIALS, random_seed=0, storage_path=journal, study_name="s").optimize(objective)
    assert len(calls) == ALL_TRIALS - FIRST_TRIALS
    complete = [t for t in study.trials if t.state == TrialState.COMPLETE]
    assert len(complete) == ALL_TRIALS
    assert study.best_value == min(t.value for t in complete)


def test_unit_store_round_trip_and_config_conflict(tmp_path: Path) -> None:
    store = UnitStore(tmp_path / "units")
    config_hash = stable_hash({"a": 1})
    assert not store.is_done("fold1", config_hash)
    frames = {"metrics": pd.DataFrame({"origin": pd.to_datetime(["2023-12-01"]), "rmse": [1.5]}), "predictions": pd.DataFrame({"y": [1.0, 2.0]})}
    store.save("fold1", config_hash, frames, {"n_trials": 3})
    assert store.is_done("fold1", config_hash, expect={"n_trials": 3})
    assert not store.is_done("fold1", config_hash, expect={"n_trials": 5})
    loaded, meta = store.load("fold1")
    assert meta == {"n_trials": 3}
    pd.testing.assert_frame_equal(loaded["metrics"], frames["metrics"])
    pd.testing.assert_frame_equal(loaded["predictions"], frames["predictions"])
    assert not list((tmp_path / "units").glob("*.tmp"))
    with pytest.raises(ConfigConflictError):
        store.is_done("fold1", stable_hash({"a": 2}))
