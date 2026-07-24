# SpikingRL Lab

SpikingRL Lab is an experimental framework for researching and developing reinforcement-learning methods with spiking neural-network policies. It is designed to make environments, agents, policies, and network components easy to extend and combine.

The current implementation includes REINFORCE, discrete and continuous policies, Gymnasium environments through skrl, PyTorch/Norse networks, and MLflow tracking.

> [!WARNING]
> This project is under active development.

## Quick start

Requirements:

- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/)

Install the locked environment and launch the default CartPole experiment:

```bash
git clone --recurse-submodules https://github.com/KirillHit/spiking_rl_lab.git
cd spiking_rl_lab
uv sync
uv run spiking-rl-lab
```

Launch a retained experiment configuration:

```bash
uv run spiking-rl-lab experiment=reinforce_cartpole_ann
```

Hydra overrides can be combined:

```bash
uv run spiking-rl-lab \
  experiment=reinforce_cartpole_ann \
  runner.seed=7 \
  trainer.params.timesteps=50000
```

## Documentation

See the [project wiki](https://github.com/KirillHit/spiking_rl_lab/wiki) for usage, architecture, configuration, algorithms, and extension guidance.

## Tracking

Models and experiment results are available on [DagsHub](https://dagshub.com/KirillHit/spiking_rl_lab). Run artifacts are written under `runs/`. If DagsHub is unavailable, MLflow uses `experiments/mlflow.db`. Inspect local runs with:

```bash
uv run mlflow ui --backend-store-uri sqlite:///experiments/mlflow.db
```
