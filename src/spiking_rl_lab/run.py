"""CLI entry point for launching experiments via Hydra."""

import logging

import hydra
from omegaconf import DictConfig, OmegaConf

from spiking_rl_lab.utils import register_configs

register_configs()

log = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra entry: receives the composed config and logs it for traceability."""
    log.info(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    main()
