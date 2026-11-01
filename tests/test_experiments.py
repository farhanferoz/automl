"""Experiment runner guards: one cell end to end, config validation, slot-in of a later variant, pairing within a fingerprint."""

import numpy as np
import pandas as pd
import pytest

from automl_package.enums import ModelName, OptimizerType
from automl_package.examples._toy_datasets import StudyToy
from automl_package.experiments import report
from automl_package.experiments.cell import cell_status, fingerprint_id, load_record, record_path
from automl_package.experiments.config import CellSpec, HpoConfig, ModelConfig, PredictorKind, ProtocolConfig, StudyConfig, ToyConfig, from_dict, to_dict
from automl_package.experiments.registry import build_model
from automl_package.experiments.study import plan, run_cells

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
    hashes = {v: load_record(record_path(c, fp))["data_hashes"] for v, c in cells.items()}
    assert hashes["offset"] == hashes["oracle"] == hashes["plain_nn"]


def test_report_pairs_only_within_one_fingerprint():
    """A variant record of one environment is never paired with the reference record of another, nor with one drawn from other data."""
    def rows(variant: str, fp: str, nfe: float, data: str = "same draws") -> pd.DataFrame:
        return pd.DataFrame({"variant": variant, "toy": "toy_a", "n": 64, "seed": [0, 1], "fp": fp, "nfe": nfe, "data": data, "path": "p"})

    noise = pd.DataFrame(columns=["toy", "n", "fp", "noise"])
    records = pd.concat([rows("ref", "aaaa", 1.0), rows("m", "aaaa", 0.5), rows("m", "bbbb", 9.0)])
    table = report.comparison_table(records, noise, "ref", "aaaa")
    assert table["mean_pct"].tolist() == [-50.0]
    assert report.comparison_table(records, noise, "ref", "bbbb").empty
    assert np.isnan(table["noise_pct"]).all()
    assert not table["greyed"].any()
    with pytest.raises(ValueError, match="different data"):
        report.comparison_table(pd.concat([rows("ref", "aaaa", 1.0), rows("m", "aaaa", 0.5, data="other draws")]), noise, "ref", "aaaa")
