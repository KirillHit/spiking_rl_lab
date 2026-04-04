"""MLflow utilities for logging metrics, git information, and configuring experiments."""

import json
import logging
import platform
import subprocess
import sys
from dataclasses import asdict
from enum import Enum
from pathlib import Path

import dagshub
import httpx
import mlflow
import psutil
import torch

from spiking_rl_lab.utils.config import BaseConfig

log = logging.getLogger(__name__)


def setup_mlflow(repo_owner: str, repo_name: str, experiment_name: str) -> None:
    """Initialize MLflow tracking against the configured DagsHub repository."""
    mlflow.enable_system_metrics_logging()

    try:
        dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
    except httpx.NetworkError:
        log.warning(
            "Failed to connect to DagsHub. Experiment logs will be stored locally: "
            "./experiments/mlflow.db",
        )
        mlflow.set_tracking_uri("sqlite:///experiments/mlflow.db")
    mlflow.set_experiment(experiment_name)


def log_git_diff_artifact(folder: Path) -> None:
    """Log the current git diff as an artifact and the current commit as a tag."""
    folder.mkdir(parents=True, exist_ok=True)

    try:
        subprocess.check_output(
            ["git", "rev-parse", "--is-inside-work-tree"],  # noqa: S607
            stderr=subprocess.DEVNULL,
        )
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        diff = subprocess.check_output(
            ["git", "diff"],  # noqa: S607
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return

    mlflow.set_tag("git_commit", commit)

    if not diff.strip():
        return

    diff_file = folder / "git_diff.txt"
    diff_file.write_text(diff)
    mlflow.log_artifact(str(diff_file))


def log_model_metadata(run: mlflow.ActiveRun, folder: Path) -> None:
    """Save the current MLflow run's parameters, metrics, and tags as a JSON artifact."""
    client = mlflow.tracking.MlflowClient()
    run_data = client.get_run(run.info.run_id)
    metadata = {
        "params": run_data.data.params,
        "metrics": run_data.data.metrics,
        "tags": run_data.data.tags,
    }
    metadata_path = folder / "run_metadata.json"
    with metadata_path.open("w") as f:
        json.dump(metadata, f, indent=4)
    mlflow.log_artifact(str(metadata_path))


def log_artifact_if_exists(path: Path) -> None:
    """Log an artifact file if it exists."""
    if not path.exists():
        log.warning("Artifact file does not exist and will not be logged to MLflow: %s", path)
        return
    mlflow.log_artifact(str(path))


def log_environment_packages(folder: Path) -> None:
    """Log installed Python packages for the current environment as an artifact."""
    folder.mkdir(parents=True, exist_ok=True)
    packages_path = folder / "pip_freeze.txt"

    try:
        packages = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        log.warning("Failed to collect installed packages with pip freeze")
        return

    packages_path.write_text(packages)
    mlflow.log_artifact(str(packages_path))


def log_hardware_info(folder: Path) -> None:
    """Log host hardware and runtime information as an artifact."""
    folder.mkdir(parents=True, exist_ok=True)
    hardware_path = folder / "hardware.json"

    hardware = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "ram_total_gb": round(psutil.virtual_memory().total / 1024**3, 2),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
    }

    if torch.cuda.is_available():
        hardware["gpus"] = [
            {
                "index": index,
                "name": props.name,
                "total_memory_gb": round(props.total_memory / 1024**3, 2),
                "multi_processor_count": props.multi_processor_count,
            }
            for index in range(torch.cuda.device_count())
            for props in [torch.cuda.get_device_properties(index)]
        ]

    hardware_path.write_text(json.dumps(hardware, indent=2))
    mlflow.log_artifact(str(hardware_path))


def config_to_dict(cfg: BaseConfig) -> dict:
    """Convert the Hydra config dataclass to a JSON-serializable dict."""

    def normalize(obj):  # noqa: ANN001, ANN202
        """Recursively normalize enums, paths, and containers."""
        if isinstance(obj, Enum):
            return obj.value
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, dict):
            return {k: normalize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [normalize(v) for v in obj]
        return obj

    return normalize(asdict(cfg))
