"""Experiment runner guards: one cell end to end, config validation, slot-in of a later variant, pairing within a fingerprint."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from automl_package.enums import MapperType, ModelName, OptimizerType
from automl_package.examples._toy_datasets import StudyToy
from automl_package.experiments import patience, report
from automl_package.experiments.cell import cell_status, fingerprint_id, load_record, record_path
from automl_package.experiments.config import (
    DEFAULT_RESULTS_ROOT,
    CellSpec,
    EvalSet,
    HpoConfig,
    ModelConfig,
    PredictorKind,
    ProtocolConfig,
    StudyConfig,
    ToyConfig,
    from_dict,
    to_dict,
)
from automl_package.experiments.registry import build_model
from automl_package.experiments.study import plan, run_cells

RESULTS_ROOT = Path(DEFAULT_RESULTS_ROOT)
TINY_TOY = ToyConfig(name=StudyToy.TOY_A, n_selection=300, n_test=300)
TINY_PROTOCOL = ProtocolConfig(seeds=[0], early_stopping_patience=2, max_epochs=2)
PLAIN_NN = ModelConfig(name=ModelName.PYTORCH_NEURAL_NETWORK, standardise_x=True, standardise_y=True)
ORACLE = ModelConfig(kind=PredictorKind.ORACLE)
OFFSET = ModelConfig(kind=PredictorKind.PLANTED_OFFSET)


def tiny_study(tmp_path, variants: dict[str, ModelConfig]) -> StudyConfig:
    """A one-toy, one-size, one-seed study writing under ``tmp_path``."""
    study = StudyConfig(name="tiny", card="t", variants=variants, toys={"toy_a": TINY_TOY}, sizes=[64], protocol=TINY_PROTOCOL)
    study.output.results_root = str(tmp_path)
    study.report.reference_variant = "oracle"
    return study


@pytest.fixture(autouse=True)
def _cpu(monkeypatch):
    monkeypatch.setenv("AUTOML_DEVICE", "cpu")


def test_one_cell_end_to_end_and_oracle_is_exact(tmp_path):
    """A plain network and the oracle each write a record with data hashes; the oracle's noise-free excess is exactly 0 on both held-out draws."""
    cells = {c.variant: c for c in plan(tiny_study(tmp_path, {"oracle": ORACLE, "plain_nn": PLAIN_NN}))}
    summary = run_cells(list(cells.values()))
    assert not summary.failed, summary.failed
    records = {v: load_record(record_path(c, fingerprint_id())) for v, c in cells.items()}
    assert set(records["plain_nn"]["data_hashes"]) == {"train", "validation", "selection", "test"}
    assert records["plain_nn"]["data_hashes"] == records["oracle"]["data_hashes"]
    assert records["plain_nn"]["n_parameters"] > 0
    for field in ("config_hash", "source_hash", "git"):
        assert records["plain_nn"][field]
    for eval_set in ("selection", "test"):
        assert records["oracle"]["metrics"][eval_set]["noise_free_excess"] == 0.0
        assert records["plain_nn"]["metrics"][eval_set]["noise_free_excess"] > 0.0


def test_config_validation_and_round_trip():
    """An unknown model key raises and names the valid keys; a string becomes the default's enum; a cell survives its JSON form."""
    with pytest.raises(ValueError, match="valid keys"):
        build_model(ModelConfig(name=ModelName.PYTORCH_NEURAL_NETWORK, params={"hiden_size": 8}), TINY_PROTOCOL, HpoConfig(n_trials=1), seed=0)
    model = build_model(ModelConfig(name=ModelName.PYTORCH_NEURAL_NETWORK, params={"optimizer_type": "ADAM"}), TINY_PROTOCOL, HpoConfig(n_trials=1), seed=0)
    assert model.optimizer_type is OptimizerType.ADAM
    with pytest.raises(ValueError, match="members"):
        build_model(ModelConfig(name=ModelName.PYTORCH_NEURAL_NETWORK, params={"optimizer_type": "NOPE"}), TINY_PROTOCOL, HpoConfig(n_trials=1), seed=0)
    cell = CellSpec(study="s", card="c", variant="v", model=PLAIN_NN, toy=TINY_TOY, n_train=64, protocol=TINY_PROTOCOL, results_dir="/x")
    assert from_dict(CellSpec, to_dict(cell)).config_hash() == cell.config_hash()
    # a composite model names its inner classifier in the config, and its own keys are checked too
    classreg = ModelConfig(name=ModelName.CLASSIFIER_REGRESSION, params={"base_classifier_class": "PYTORCH_NEURAL_NETWORK", "mapper_type": "LOOKUP_MEDIAN"})
    model = build_model(classreg, TINY_PROTOCOL, HpoConfig(n_trials=1), seed=0)
    assert model.base_classifier_class.__name__ == "PyTorchNeuralNetwork"
    assert model.mapper_type is MapperType.LOOKUP_MEDIAN
    assert model.random_seed == 0
    classreg.params["n_clases"] = 3
    with pytest.raises(ValueError, match="valid keys"):
        build_model(classreg, TINY_PROTOCOL, HpoConfig(n_trials=1), seed=0)


def test_finished_cells_keep_their_identity():
    """A cell composed from today's YAML hashes as the record it wrote did and finds it in the layout (a schema or path change would silently re-run every finished cell)."""
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from automl_package.experiments import run

    path = RESULTS_ROOT / "runs" / "main" / "probreg" / "toy_a" / "n250" / "s0.json"
    record = load_record(path)
    with initialize_config_dir(config_dir=run.CONF_DIR, version_base="1.3"):
        patience_override = f"study.protocol.early_stopping_patience={record['config']['protocol']['early_stopping_patience']}"
        cfg = compose(config_name="config", overrides=["+study=probreg_5b", "study.sizes=[250]", patience_override])
    (cell,) = [c for c in plan(OmegaConf.to_object(cfg).study) if c.variant == "probreg" and c.toy_label == "toy_a" and c.seed == 0]
    assert cell.config_hash() == record["config_hash"]
    assert record_path(cell, record["fingerprint_id"]) == path  # the configured layout finds the finished records


def test_later_variant_slots_in_without_rerunning(tmp_path):
    """Adding a variant and planning again runs only the new variant's cells, on the same draws as the first run."""
    first = tiny_study(tmp_path, {"oracle": ORACLE, "plain_nn": PLAIN_NN})
    assert not run_cells(plan(first)).failed
    second = tiny_study(tmp_path, {"oracle": ORACLE, "plain_nn": PLAIN_NN, "offset": OFFSET})
    summary = run_cells(plan(second))
    assert [s.split("/")[0] for s in summary.ran] == ["offset"]
    assert len(summary.skipped) == len(first.variants)
    cells = {c.variant: c for c in plan(second)}
    fp = fingerprint_id()
    assert all(cell_status(c, fp).value == "done" for c in cells.values())
    records = report.load_records(plan(first) + plan(second), EvalSet.SELECTION)  # two plans sharing records: each read once
    assert len(records) == len(second.variants)
    hashes = {v: load_record(record_path(c, fp))["data_hashes"] for v, c in cells.items()}
    assert hashes["offset"] == hashes["oracle"] == hashes["plain_nn"]


def test_report_pairs_only_within_one_fingerprint():
    """A variant record of one environment is never paired with the reference record of another, nor with one drawn from other data."""
    def rows(variant: str, fp: str, nfe: float, data: str = "same draws") -> pd.DataFrame:
        return pd.DataFrame({"variant": variant, "toy": "toy_a", "n": 64, "seed": [0, 1], "fp": fp, "nfe": nfe, "data": data, "path": "p"})

    noise = pd.DataFrame(columns=report.NOISE_COLUMNS)
    records = pd.concat([rows("ref", "aaaa", 1.0), rows("m", "aaaa", 0.5), rows("m", "bbbb", 9.0)])
    table = report.comparison_table(records, noise, "ref", "aaaa")
    assert table["mean_pct"].tolist() == [-50.0]
    assert report.comparison_table(records, noise, "ref", "bbbb").empty
    assert np.isnan(table["noise_pct"]).all()
    assert not table["greyed"].any()
    borrowed = pd.DataFrame({"toy": "toy_a", "n": 64, "fp": ["cccc", "dddd"], "noise": [0.9, 0.1], "n_records": [3, 5]})  # neither is the card's own environment
    filled = report.comparison_table(records, borrowed, "ref", "aaaa")
    assert filled["noise_fp"].tolist() == ["dddd"]  # the fingerprint with the most 5a records
    assert filled["noise_pct"].tolist() == [10.0]
    own = pd.concat([borrowed, pd.DataFrame({"toy": "toy_a", "n": 64, "fp": "aaaa", "noise": 0.2, "n_records": 1}, index=[0])])
    assert report.comparison_table(records, own, "ref", "aaaa")["noise_fp"].tolist() == ["aaaa"]  # its own wins over more records elsewhere
    with pytest.raises(ValueError, match="different data"):
        report.comparison_table(pd.concat([rows("ref", "aaaa", 1.0), rows("m", "aaaa", 0.5, data="other draws")]), noise, "ref", "aaaa")
    # several cards stack into one table only against one shared reference, and the table is named after every card
    card_t, card_u = (CellSpec(study="s", card=c, variant="v", model=PLAIN_NN, toy=TINY_TOY, n_train=64, protocol=TINY_PROTOCOL, results_dir="/x") for c in "tu")
    assert report.card_label([card_t, card_u, card_t]) == ("t+u", "Cards t+u")
    assert report.card_label([card_t]) == ("t", "Card t")
    card_u.report.reference_variant = "other"
    with pytest.raises(ValueError, match="reference variant"):
        report.build_report([card_t, card_u])


def test_patience_rule(tmp_path):
    """The patience is the longest wait for a new best, over every curve; hit-cap curves are counted, not dropped."""
    assert patience.longest_stretch_before_best(np.array([1.0, 2.0, 3.0])) == 0  # best at the first checkpoint
    assert patience.longest_stretch_before_best(np.array([9, 8, 8, 7, 6, 6, 6, 6, 6, 5, 6, 7.0])) == 5  # bests at 0, 1, 3, 4, 9; ties are not new bests
    assert patience.longest_stretch_before_best(np.array([5, 4, 4, 4, 3.0])) == 3
    cells = plan(tiny_study(tmp_path, {"oracle": ORACLE, "plain_nn": PLAIN_NN}))
    assert [c.variant for c in patience.missing_records(cells)] == ["oracle", "plain_nn"]
    assert not run_cells(cells).failed
    assert not patience.missing_records(cells)
    record = load_record(record_path(next(c for c in cells if c.variant == "plain_nn"), fingerprint_id()))
    expected = patience.longest_stretch_before_best(np.array([value for _, value in record["convergence"]["trajectory"]]))
    choice = patience.choose_patience(cells)  # the oracle has no curve and is skipped
    assert (choice.patience, choice.n_curves, choice.n_hit_cap, choice.max_epochs) == (expected, 1, int(record["convergence"]["hit_cap"]), 2)
