from jsonargparse import ArgumentParser, Namespace

from spike_rl import Trainer
from spike_rl.utils.config import Config
from spike_rl.utils.logger import console_logger


def prepare_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="RL-SNN trainer",
        default_config_files=["config/example.yaml"],
    )
    parser.add_argument(
        "--config",
        action="config",
        help="Path to a YAML configuration file",
    )
    parser.add_class_arguments(Config, "rl_params")
    return parser


def main():
    parser = prepare_parser()
    cfg = parser.parse_args()
    trainer_cfg: Namespace = cfg.rl_params

    console_logger.configure(trainer_cfg.logging.level, trainer_cfg.logging.cmd_log_path)

    trainer = Trainer()
    trainer.run(trainer_cfg)


if __name__ == "__main__":
    main()
