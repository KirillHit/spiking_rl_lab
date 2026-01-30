"""CLI entry point for launching experiments via Hydra."""

import logging

import hydra
from omegaconf import DictConfig, OmegaConf
from skrl.utils import set_seed

from spiking_rl_lab.scripts import evaluate, optimize, train
from spiking_rl_lab.utils import register_configs
from spiking_rl_lab.utils.config import BaseConfig

register_configs()

# Silence the default skrl handlers to avoid duplicate log output.
skrl_logger = logging.getLogger("skrl")
skrl_logger.handlers.clear()

log = logging.getLogger(__name__)


def run(cfg: BaseConfig) -> None:
    """Run the training loop using the provided configuration."""
    log.info("Starting SpikingRL Lab in mode: %s", cfg.trainer.mode)
    log.info("Experiment name: %s", cfg.trainer.experiment_name)

    set_seed(cfg.trainer.seed, deterministic=cfg.trainer.deterministic)

    match cfg.trainer.mode:
        case "train":
            train(cfg)
        case "evaluate":
            evaluate(cfg)
        case "optimize":
            optimize(cfg)
        case _:
            log.error("Unsupported mode: %s!", cfg.trainer.mode)

    log.info("SpikingRL Lab finished.")


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra entry: receives the composed config and logs it for traceability."""
    cfg_obj = OmegaConf.to_object(cfg)
    if not isinstance(cfg_obj, BaseConfig):
        log.error("Invalid config type: expected BaseConfig, got %s", type(cfg_obj).__name__)
        return

    run(cfg_obj)


if __name__ == "__main__":
    main()
