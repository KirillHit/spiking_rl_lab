"""..."""

import contextlib
import subprocess
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
from mlflow.exceptions import MlflowException
from stable_baselines3.common.logger import KVWriter, Logger


class MLflowOutputFormat(KVWriter):
    """Dumps key/value pairs into MLflow's numeric format."""

    def write(
        self,
        key_values: dict[str, Any],
        key_excluded: dict[str, str | tuple[str, ...]],
        step: int = 0,
    ) -> None:
        """..."""
        for (key, value), (_, excluded) in zip(
            sorted(key_values.items()),
            sorted(key_excluded.items()),
            strict=False,
        ):
            if excluded is not None and "mlflow" in excluded:
                continue

            if isinstance(value, np.ScalarType) and not isinstance(value, str):
                mlflow.log_metric(key, value, step)


sb3_logger = Logger(
    None,
    output_formats=[MLflowOutputFormat()],
)


def setup_mlflow(base_dir: Path, experiment_name: str | None = None) -> None:
    base_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(f"sqlite:///{base_dir / 'mlflow.db'}")

    with contextlib.suppress(MlflowException):
        mlflow.create_experiment(
            name=experiment_name,
            artifact_location=str(base_dir / "artifacts"),
        )

    mlflow.set_experiment(experiment_name)


def log_git_diff_artifact(folder: Path) -> None:
    """..."""
    folder.mkdir(exist_ok=True)

    diff_file = folder / "git_diff.txt"
    diff_file.write_text(subprocess.check_output(["git", "diff"], text=True))  # noqa: S607
    mlflow.log_artifact(str(diff_file))

    commit_file = folder / "git_commit.txt"
    commit_file.write_text(subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip())  # noqa: S607
    mlflow.log_artifact(str(commit_file))
