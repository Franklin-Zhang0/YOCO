"""Learning rate scheduler utilities for training."""
import math
import torch
from torch.optim.lr_scheduler import LRScheduler


def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_steps: int = 0,
    peak_lr: float = 1e-3,
    final_lr: float = 0.0,
    start_lr: float = 0.0,
    last_epoch: int = -1,
) -> LRScheduler:
    """Build a warmup + cosine annealing learning rate scheduler.
    
    The scheduler assumes the optimizer's base learning rate is set to `peak_lr`.
    It returns a LambdaLR scheduler that modulates this base rate.
    
    Schedule behavior:
    - If warmup_steps >= total_steps: linear schedule from start_lr to final_lr over total_steps
    - If 0 < warmup_steps < total_steps:
        - Warmup phase (0 to warmup_steps): linear from start_lr to peak_lr
        - Cosine phase (warmup_steps to total_steps): cosine decay from peak_lr to final_lr
    - If warmup_steps == 0: begins immediately at peak_lr, then cosine decays to final_lr
    - Steps beyond total_steps are clamped to final_lr
    
    Args:
        optimizer: The optimizer to schedule.
        total_steps: Total number of training steps.
        warmup_steps: Number of warmup steps (default: 0).
        peak_lr: Peak learning rate after warmup (default: 1e-3).
        final_lr: Final learning rate after decay (default: 0.0).
        start_lr: Initial learning rate at step 0 (default: 0.0).
        last_epoch: Index of last step; used for resuming (default: -1).
    
    Returns:
        A torch LRScheduler instance.
    
    Example:
        >>> optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        >>> scheduler = build_warmup_cosine_scheduler(
        ...     optimizer, total_steps=10000, warmup_steps=1000, peak_lr=1e-3
        ... )
        >>> for step in range(10000):
        ...     optimizer.step()
        ...     scheduler.step()
    """
    warmup_steps = max(0, warmup_steps)
    
    def _lr_at_step(step: int) -> float:
        """Compute absolute learning rate at given step."""
        if warmup_steps >= total_steps:
            # Edge case: only linear schedule from start_lr to final_lr
            progress = step / max(total_steps, 1)
            return start_lr + (final_lr - start_lr) * min(progress, 1.0)
        elif step < warmup_steps:
            # Warmup phase: linear from start_lr to peak_lr
            if warmup_steps == 0:
                return peak_lr
            progress = step / warmup_steps
            return start_lr + (peak_lr - start_lr) * progress
        else:
            # Cosine decay phase: from peak_lr to final_lr
            cosine_steps = total_steps - warmup_steps
            progress = (step - warmup_steps) / max(cosine_steps, 1)
            if progress >= 1.0:
                return final_lr
            # Cosine annealing formula
            return final_lr + (peak_lr - final_lr) * 0.5 * (1 + math.cos(math.pi * progress))
    
    def lr_lambda(step: int) -> float:
        """Convert absolute LR to multiplicative factor for LambdaLR."""
        abs_lr = _lr_at_step(step)
        return abs_lr / max(peak_lr, 1e-10)  # Avoid division by zero
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)
