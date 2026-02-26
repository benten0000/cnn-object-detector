from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.targets import build_targets
from src.training.config import TrainConfig


def run_epoch(
    *,
    train: bool,
    epoch: int,
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    cfg: TrainConfig,
    device: torch.device,
    num_classes: int,
) -> float:
    model.train(train)
    total = 0.0
    ctx = torch.enable_grad() if train else torch.no_grad()
    amp_enabled = bool(cfg.amp and device.type == "cuda")
    amp_dtype = torch.bfloat16 if str(cfg.amp_dtype).lower() in {"bf16", "bfloat16"} else torch.float16

    with ctx:
        for step, (images, boxes, classes, valid_mask) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=True)
            boxes = boxes.to(device, non_blocking=True)
            classes = classes.to(device, non_blocking=True)
            valid_mask = valid_mask.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                predictions = model(images)
                targets = build_targets(
                    boxes_xywh_norm=boxes,
                    classes=classes,
                    S=cfg.S,
                    C=num_classes,
                    valid_mask=valid_mask,
                )
                losses = criterion(predictions, targets)

            if train:
                optimizer.zero_grad(set_to_none=True)
                losses["total"].backward()
                optimizer.step()

            total += float(losses["total"].item())
            if step % cfg.log_every == 0 or step == len(loader):
                if train:
                    print(
                        f"epoch {epoch:03d} train step {step:04d}/{len(loader):04d} "
                        f"total={losses['total'].item():.4f} "
                        f"coord={losses['coord'].item():.4f} "
                        f"obj={losses['obj'].item():.4f} "
                        f"noobj={losses['noobj'].item():.4f} "
                        f"cls={losses['cls'].item():.4f}"
                    )
                else:
                    print(f"epoch {epoch:03d} val   step {step:04d}/{len(loader):04d} total={losses['total'].item():.4f}")

    return total / max(1, len(loader))


def save_checkpoint(
    *,
    out_dir: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
    train_avg_loss: float,
    val_avg_loss: float,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "last.pt"
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": asdict(cfg),
            "train_avg_loss": train_avg_loss,
            "val_avg_loss": val_avg_loss,
        },
        ckpt_path,
    )
    return ckpt_path
