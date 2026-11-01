"""Hydra entry point: composes a study, writes its cells, reports what is pending.

    python -m automl_package.experiments.run +study=probreg_5b [study.sizes=[500] ...]

The cells go to ``<results_dir>/cells_<card>.jsonl``; ``execute.py`` (no Hydra) runs them. Every YAML under ``conf/`` extends a
schema registered here, so a wrong key or type fails at composition.

Reused: the structured-config layout of ``/home/ff235/dev/ae/automl_adapter/config.py`` and ``/home/ff235/dev/ae/conf``.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from automl_package.experiments.cell import CellStatus, cell_status, environment_fingerprint, fingerprint_id
from automl_package.experiments.config import HpoConfig, ModelConfig, ProtocolConfig, RootConfig, StudyConfig, ToyConfig
from automl_package.experiments.study import PLAN_FILE, plan, results_dir, write_plan

CONF_DIR = str(Path(__file__).resolve().parent / "conf")


def register_configs() -> None:
    """Registers the schemas the YAML files extend (root, and one per config group)."""
    store = ConfigStore.instance()
    store.store(name="root_schema", node=RootConfig)
    store.store(group="study", name="study_schema", node=StudyConfig)
    store.store(group="model", name="model_schema", node=ModelConfig)
    store.store(group="toy", name="toy_schema", node=ToyConfig)
    store.store(group="protocol", name="protocol_schema", node=ProtocolConfig)
    store.store(group="hpo", name="hpo_schema", node=HpoConfig)


register_configs()


@hydra.main(config_path=CONF_DIR, config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Writes the plan and prints the command that runs it."""
    root: RootConfig = OmegaConf.to_object(cfg)  # type: ignore[assignment]
    study = root.study
    cells = plan(study)
    path = results_dir(study) / PLAN_FILE.format(card=study.card)
    write_plan(cells, path)
    target = root.target_fingerprint or fingerprint_id()
    status = Counter(cell_status(c, target) for c in cells)
    print(f"{len(cells)} cells -> {path}")
    print(f"for fingerprint {target} (this machine: {fingerprint_id()} {environment_fingerprint()}): {status[CellStatus.PENDING]} pending, {status[CellStatus.DONE]} done")
    print(f"run: python -m automl_package.experiments.execute {path} --workers {study.n_workers}")


if __name__ == "__main__":
    main()
