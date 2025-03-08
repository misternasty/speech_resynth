import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from transformers import LlamaConfig, LlamaForCausalLM

from .data import UnitDataset
from .utils import get_lr_schedule


@torch.amp.autocast("cuda", dtype=torch.bfloat16)
@torch.inference_mode()
def validate(dataloader, model, step: int, writer: SummaryWriter):
    torch.cuda.empty_cache()
    model.eval()

    losses = []
    for batch in dataloader:
        loss = model(
            input_ids=batch["input_ids"].cuda(),
            attention_mask=batch["attention_mask"].cuda(),
            labels=batch["labels"].cuda(),
        ).loss
        losses.append(loss)
    loss = torch.tensor(losses).mean().item()

    writer.add_scalar("dev/loss", loss, step)
    torch.cuda.empty_cache()


def train(config):
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    print(f"Start running DDP on rank {rank}.", flush=True)
    # create model and move it to GPU with id rank
    device_id = rank % torch.cuda.device_count()

    train_set = UnitDataset(config.dataset.train_file, config.dataset.units_per_sample)
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=config.dataloader.batch_size,
        shuffle=True,
        num_workers=config.dataloader.num_workers,
        persistent_workers=True,
    )

    if rank == 0:
        dev_set = UnitDataset(config.dataset.dev_file, config.dataset.units_per_sample)
        dev_loader = torch.utils.data.DataLoader(
            dev_set,
            batch_size=config.dataloader.batch_size,
            num_workers=config.dataloader.num_workers,
            persistent_workers=True,
        )
        writer = SummaryWriter(config.model.path)

    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=config.model.vocab_size + 1,
            hidden_size=config.model.hidden_size,
            intermediate_size=config.model.intermediate_size,
            num_hidden_layers=config.model.num_hidden_layers,
            num_attention_heads=config.model.num_attention_heads,
            pad_token_id=config.model.pad_token_id,
            bos_token_id=config.model.bos_token_id,
            eos_token_id=config.model.eos_token_id,
        )
    ).to(device_id)
    model = DDP(model, device_ids=[device_id])

    optimizer = torch.optim.AdamW(model.parameters(), config.optim.lr, (config.optim.beta1, config.optim.beta2))

    # learning rate scheduler
    lr_scheduler = get_lr_schedule(
        optimizer,
        config.optim.epoch * len(train_loader),
        config.optim.warmup_steps,
        config.optim.lr,
        config.optim.lr_min,
    )

    scaler = torch.amp.GradScaler("cuda", init_scale=1e24)

    last_epoch = 0
    step = 0

    # resume training
    checkpoint_path = Path(config.model.path) / "checkpoint"
    if checkpoint_path.is_file():
        ckpt = torch.load(checkpoint_path, weights_only=True)

        last_epoch = ckpt["epoch"]
        step = ckpt["step"]
        model.module.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        lr_scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])

        print(f"load from {checkpoint_path}", flush=True)
        del ckpt
        torch.cuda.empty_cache()

    for epoch in range(last_epoch + 1, config.optim.epoch + 1):
        model.train()

        for batch in train_loader:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss = model(
                    input_ids=batch["input_ids"].cuda(),
                    attention_mask=batch["attention_mask"].cuda(),
                    labels=batch["labels"].cuda(),
                ).loss
            scaler.scale(loss).backward()

            # gradient clipping
            if config.optim.max_norm is not None:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.optim.max_norm)

            # update model
            scaler.step(optimizer)
            scale = scaler.get_scale()
            scaler.update()
            optimizer.zero_grad()

            # update learning rate
            lr = lr_scheduler.get_last_lr()[0]
            lr_scheduler.step()

            step += 1

            # tensorboard log
            if step % config.optim.summary_interval == 0:
                writer.add_scalar("train/loss", loss.item(), step)
                writer.add_scalar("train/lr", lr, step)
                writer.add_scalar("train/scale", scale, step)
                if config.optim.max_norm is not None:
                    writer.add_scalar("train/grad_norm", grad_norm.item(), step)

        if rank == 0:
            validate(dev_loader, model, step, writer)

            # save model
            ckpt = {
                "epoch": epoch,
                "step": step,
                "model": model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": lr_scheduler.state_dict(),
                "scaler": scaler.state_dict(),
            }
            Path(config.model.path).parent.mkdir(parents=True, exist_ok=True)
            model.module.save_pretrained(config.model.path)
            torch.save(ckpt, checkpoint_path)

    torch.distributed.destroy_process_group()
