# SpikingRL Lab

Experimental framework for reinforcement learning with spiking neural network policies.

> [!Warning]
> This project is under development.

## Requirements

- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/)

## Setup

Install the project dependencies from `pyproject.toml` and `uv.lock`:

```bash
git clone https://github.com/KirillHit/spiking_rl_lab.git
cd spiking_rl_lab
uv sync
```

## Run

Start an experiment with the default configuration:

```bash
uv run spiking-rl-lab
```

Override configuration values from the command line:

```bash
uv run spiking-rl-lab env=pendulum_v1
```

Saved experiment configurations are stored in `src/spiking_rl_lab/configs/experiment/`. Launch one by passing its filename (without `.yaml`) as `experiment`:

```bash
uv run spiking-rl-lab experiment=<experiment_name>
```

## Logs and Artifacts

Local run logs and artifacts are written under `runs/`.

If DagsHub is unavailable, experiment metadata is stored locally in `experiments/mlflow.db`.

## Tracking

Models and experiment results are available on [DagsHub](https://dagshub.com/KirillHit/spiking_rl_lab).

Open the local MLflow UI for locally stored runs:

```bash
uv run mlflow ui --backend-store-uri sqlite:///experiments/mlflow.db
```
