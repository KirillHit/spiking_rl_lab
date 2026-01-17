"""..."""

import subprocess
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
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


def log_git_diff_artifact(folder: Path) -> None:
    """..."""
    folder.mkdir(exist_ok=True)

    # git diff
    diff_file = folder / "git_diff.txt"
    diff_file.write_text(subprocess.check_output(["git", "diff"], text=True))
    mlflow.log_artifact(str(diff_file.resolve()))

    # git commit
    commit_file = folder / "git_commit.txt"
    commit_file.write_text(subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip())
    mlflow.log_artifact(str(commit_file.resolve()))
