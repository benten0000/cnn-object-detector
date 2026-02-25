import json
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
from torchvision.transforms.functional import resize

from src.model import YOLOv1
from src.targets import build_targets


class CocoDetectionDataset(Dataset):
    def __init__(
        self,
        images_dir: str,
        annotations_path: str,
        image_size: int = 416,
        max_samples: int | None = None,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.image_size = image_size

        with Path(annotations_path).open("r", encoding="utf-8") as f:
            coco = json.load(f)

        images = {img["id"]: img for img in coco["images"]}
        categories = sorted(cat["id"] for cat in coco["categories"])
        self.cat_to_idx = {cat_id: i for i, cat_id in enumerate(categories)}
        self.num_classes = len(self.cat_to_idx)

        anns_by_image: dict[int, list[dict[str, Any]]] = {}
        for ann in coco["annotations"]:
            if ann.get("iscrowd", 0) == 1:
                continue
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            if ann["image_id"] in images:
                anns_by_image.setdefault(ann["image_id"], []).append(ann)

        self.samples = [(images[image_id], anns) for image_id, anns in anns_by_image.items()]
        if max_samples is not None:
            self.samples = self.samples[:max_samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image_info, anns = self.samples[idx]
        image = read_image(str(self.images_dir / image_info["file_name"])).float().div_(255.0)
        image = resize(image, [self.image_size, self.image_size], antialias=True)

        h = float(image_info["height"])
        w = float(image_info["width"])

        boxes = []
        classes = []
        for ann in anns:
            x, y, bw, bh = ann["bbox"]
            boxes.append([(x + 0.5 * bw) / w, (y + 0.5 * bh) / h, bw / w, bh / h])
            classes.append(self.cat_to_idx[ann["category_id"]])

        return image, torch.tensor(boxes, dtype=torch.float32), torch.tensor(classes, dtype=torch.long)


class SyntheticDetectionDataset(Dataset):
    def __init__(self, n_samples: int = 128, image_size: int = 416, num_classes: int = 1) -> None:
        self.n_samples = n_samples
        self.image_size = image_size
        self.num_classes = num_classes

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image = torch.rand(3, self.image_size, self.image_size, dtype=torch.float32)
        n_obj = int(torch.randint(1, 4, (1,)).item())

        boxes = []
        classes = []
        for offset in range(n_obj):
            boxes.append(
                [
                    torch.rand(1).item() * 0.8 + 0.1,
                    torch.rand(1).item() * 0.8 + 0.1,
                    torch.rand(1).item() * 0.4 + 0.05,
                    torch.rand(1).item() * 0.4 + 0.05,
                ]
            )
            classes.append((idx + offset) % self.num_classes)

        return image, torch.tensor(boxes, dtype=torch.float32), torch.tensor(classes, dtype=torch.long)


def detection_collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    images, box_list, cls_list = zip(*batch)
    batch_size = len(batch)
    max_objects = max(b.shape[0] for b in box_list)

    boxes = torch.zeros((batch_size, max_objects, 4), dtype=torch.float32)
    classes = torch.zeros((batch_size, max_objects), dtype=torch.long)
    valid_mask = torch.zeros((batch_size, max_objects), dtype=torch.bool)

    for i, (b, c) in enumerate(zip(box_list, cls_list)):
        k = b.shape[0]
        boxes[i, :k] = b
        classes[i, :k] = c
        valid_mask[i, :k] = True

    return torch.stack(images, dim=0), boxes, classes, valid_mask


class YoloV1Loss(nn.Module):
    def __init__(self, C: int, lambda_coord: float = 5.0, lambda_noobj: float = 0.5) -> None:
        super().__init__()
        self.C = C
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    @staticmethod
    def _masked_sum(values: torch.Tensor, mask: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return values[mask].sum() if mask.any() else ref.new_zeros(())

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> dict[str, torch.Tensor]:
        n = predictions.shape[0]
        pred_xywh = torch.sigmoid(predictions[..., :4])
        pred_conf = predictions[..., 4]
        tgt_xywh = targets[..., :4]
        tgt_conf = targets[..., 4]

        obj_mask = tgt_conf > 0
        noobj_mask = ~obj_mask
        cell_obj_mask = obj_mask

        coord_loss = self._masked_sum(((pred_xywh - tgt_xywh) ** 2).sum(dim=-1), obj_mask, predictions)
        conf_loss = self.bce(pred_conf, tgt_conf)
        obj_loss = self._masked_sum(conf_loss, obj_mask, predictions)
        noobj_loss = self._masked_sum(conf_loss, noobj_mask, predictions)

        cls_loss_raw = self.bce(predictions[..., 5:], targets[..., 5:]).sum(dim=-1)
        cls_loss = self._masked_sum(cls_loss_raw, cell_obj_mask, predictions)

        total = self.lambda_coord * coord_loss + obj_loss + self.lambda_noobj * noobj_loss + cls_loss
        denom = max(1, n)
        return {
            "total": total / denom,
            "coord": coord_loss / denom,
            "obj": obj_loss / denom,
            "noobj": noobj_loss / denom,
            "cls": cls_loss / denom,
        }


@dataclass
class TrainConfig:
    images_dir: str = "data/coco/train2017"
    annotations: str = "data/coco/annotations/instances_train2017.json"
    val_images_dir: str = "data/coco/val2017"
    val_annotations: str = "data/coco/annotations/instances_val2017.json"
    out_dir: str = "runs/exp1"
    epochs: int = 10
    batch_size: int = 8
    num_workers: int = 2
    lr: float = 1e-4
    image_size: int = 416
    max_samples: int | None = None
    val_max_samples: int | None = None
    seed: int = 42
    log_every: int = 20
    base: int = 32
    S: int = 13
    C: int | None = None
    synthetic: bool = False


def _coerce_env(raw: str, current: Any) -> Any:
    if current is None:
        txt = raw.strip().lower()
        if txt in {"", "none", "null"}:
            return None
        for caster in (int, float):
            try:
                return caster(raw)
            except ValueError:
                pass
        return raw
    if isinstance(current, bool):
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(current, int):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    return raw


def load_config() -> TrainConfig:
    cfg = TrainConfig()
    cfg_path = Path(os.environ.get("TRAIN_CONFIG", "config/train.yaml"))

    if cfg_path.exists():
        data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        allowed = {f.name for f in fields(cfg)}
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise ValueError(f"Unknown config keys in {cfg_path}: {unknown}")
        for key, value in data.items():
            setattr(cfg, key, value)

    for f in fields(cfg):
        env_key = f"TRAIN_{f.name.upper()}"
        if env_key in os.environ:
            setattr(cfg, f.name, _coerce_env(os.environ[env_key], getattr(cfg, f.name)))

    return cfg


def make_dataset(cfg: TrainConfig, *, train: bool) -> CocoDetectionDataset | SyntheticDetectionDataset:
    if cfg.synthetic:
        n_default = 256 if train else 64
        n_samples = cfg.max_samples if train else cfg.val_max_samples
        return SyntheticDetectionDataset(
            n_samples=n_default if n_samples is None else n_samples,
            image_size=cfg.image_size,
            num_classes=cfg.C or 1,
        )

    images_dir = cfg.images_dir if train else cfg.val_images_dir
    annotations = cfg.annotations if train else cfg.val_annotations
    if not Path(images_dir).exists() or not Path(annotations).exists():
        split = "train" if train else "val"
        raise FileNotFoundError(f"COCO {split} paths not found: images={images_dir} annotations={annotations}")

    return CocoDetectionDataset(
        images_dir=images_dir,
        annotations_path=annotations,
        image_size=cfg.image_size,
        max_samples=cfg.max_samples if train else cfg.val_max_samples,
    )


def make_loader(dataset: Dataset, cfg: TrainConfig, *, shuffle: bool, device: torch.device) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=detection_collate_fn,
    )


def run_epoch(
    *,
    train: bool,
    epoch: int,
    model: YOLOv1,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: YoloV1Loss,
    cfg: TrainConfig,
    device: torch.device,
    num_classes: int,
) -> float:
    model.train(train)
    total = 0.0
    ctx = torch.enable_grad() if train else torch.no_grad()

    with ctx:
        for step, (images, boxes, classes, valid_mask) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=True)
            boxes = boxes.to(device, non_blocking=True)
            classes = classes.to(device, non_blocking=True)
            valid_mask = valid_mask.to(device, non_blocking=True)

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

        ckpt_path = out_dir / "last.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": asdict(cfg),
                "train_avg_loss": train_avg,
                "val_avg_loss": val_avg,
            },
            ckpt_path,
        )
        print(
            f"epoch {epoch:03d} done | train_avg_loss={train_avg:.4f} "
            f"| val_avg_loss={val_avg:.4f} | saved={ckpt_path}"
        )


if __name__ == "__main__":
    main()
