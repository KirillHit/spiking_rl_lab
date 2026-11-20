"""Learning-rate schedulers."""

import torch
from torch.optim.lr_scheduler import _LRScheduler


class KLAdaptiveLR(_LRScheduler):
    """Adapt learning rates to KL divergence, preserving relative group bounds."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        kl_threshold: float = 0.008,
        min_lr: float = 1e-6,
        max_lr: float = 1e-2,
        kl_factor: float = 2,
        lr_factor: float = 1.5,
        last_epoch: int = -1,
    ) -> None:
        """Follow skrl's KLAdaptiveLR with bounds relative to the first group."""
        super().__init__(optimizer, last_epoch)

        self.kl_threshold = kl_threshold
        self.min_lr = min_lr
        self.max_lr = max_lr
        self._kl_factor = kl_factor
        self._lr_factor = lr_factor

        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]

    def step(self, kl: torch.Tensor | float | None = None, *, epoch: int | None = None) -> None:
        """Adjust rates when KL is supplied, using proportional bounds per group."""
        if kl is not None:
            for group, initial_lr in zip(self.optimizer.param_groups, self.base_lrs, strict=True):
                ratio = initial_lr / self.base_lrs[0]
                if kl > self.kl_threshold * self._kl_factor:
                    group["lr"] = max(group["lr"] / self._lr_factor, self.min_lr * ratio)
                elif kl < self.kl_threshold / self._kl_factor:
                    group["lr"] = min(group["lr"] * self._lr_factor, self.max_lr * ratio)

            self._last_lr = [group["lr"] for group in self.optimizer.param_groups]
