import json
from itertools import islice
from typing import Any, Dict

import torch
from torch.nn.utils.rnn import pad_sequence


def load_batch_from_json(file, batch_size: int) -> Dict[str, Any]:
    with open(file) as f:
        dataset = json.load(f)

    dataloader = iter(dataset.items())

    while True:
        batch = dict(islice(dataloader, batch_size))

        if not batch:
            break

        names = list(batch.keys())
        input_ids = [torch.tensor(value) + 1 for value in batch.values()]  # 0: pad
        input_ids = pad_sequence(input_ids, batch_first=True)

        yield {"names": names, "input_ids": input_ids}


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
