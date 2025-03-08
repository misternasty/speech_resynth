import torch


def shift_unit(unit: int) -> int:
    if unit < 94:
        return unit + 33
    else:
        return unit + 67


def get_lr_schedule(
    optimizer,
    total_steps: int,
    warmup_steps: int = 5000,
    base_lr: float = 1e-3,
    min_lr: float = 1e-4,
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_schedule(current_step: int) -> float:
        if current_step < warmup_steps:
            return (min_lr + (base_lr - min_lr) * current_step / warmup_steps) / base_lr
        else:
            progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
            return (min_lr + (base_lr - min_lr) * (1 - progress)) / base_lr

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)
