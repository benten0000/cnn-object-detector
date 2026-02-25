from pathlib import Path

import torch

from src.model import YOLOv1
from src.training.config import load_config
from src.training.data import make_dataset, make_loader
from src.training.engine import run_epoch, save_checkpoint
from src.training.loss import YoloV1Loss


def main() -> None:
    cfg = load_config()
    torch.manual_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = make_dataset(cfg, train=True)
    val_dataset = make_dataset(cfg, train=False)

    num_classes = int(cfg.C if cfg.C is not None else getattr(train_dataset, "num_classes", 1))
    cfg.C = num_classes

    train_loader = make_loader(train_dataset, cfg, shuffle=True, device=device)
    val_loader = make_loader(val_dataset, cfg, shuffle=False, device=device)

    model = YOLOv1(S=cfg.S, C=num_classes, base=cfg.base).to(device)
    criterion = YoloV1Loss(C=num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Training on {device} | train_samples={len(train_dataset)} | val_samples={len(val_dataset)} | "
        f"S={cfg.S} C={cfg.C} | batch={cfg.batch_size}"
    )

    for epoch in range(1, cfg.epochs + 1):
        train_avg = run_epoch(
            train=True,
            epoch=epoch,
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            cfg=cfg,
            device=device,
            num_classes=num_classes,
        )
        val_avg = run_epoch(
            train=False,
            epoch=epoch,
            model=model,
            loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            cfg=cfg,
            device=device,
            num_classes=num_classes,
        )

        ckpt_path = save_checkpoint(
            out_dir=out_dir,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            cfg=cfg,
            train_avg_loss=train_avg,
            val_avg_loss=val_avg,
        )
        print(
            f"epoch {epoch:03d} done | train_avg_loss={train_avg:.4f} "
            f"| val_avg_loss={val_avg:.4f} | saved={ckpt_path}"
        )


if __name__ == "__main__":
    main()

