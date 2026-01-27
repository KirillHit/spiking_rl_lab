"""CLI entry point for launching experiments via Hydra."""

import logging

import hydra
from omegaconf import DictConfig, OmegaConf

from spiking_rl_lab.trainer import Trainer
from spiking_rl_lab.utils import register_configs
from spiking_rl_lab.utils.config import BaseConfig

register_configs()

log = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra entry: receives the composed config and logs it for traceability."""
    cfg_obj = OmegaConf.to_object(cfg)
    if not isinstance(cfg_obj, BaseConfig):
        log.error("Invalid config type: expected BaseConfig, got %s", type(cfg_obj).__name__)
        return

    trainer = Trainer()
    trainer.run(cfg_obj)


if __name__ == "__main__":
    main()
