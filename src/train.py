import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
from torchvision.transforms.functional import resize

try:
    from src.model import YOLOv1
    from src.targets import build_targets
except ModuleNotFoundError:
    from model import YOLOv1
    from targets import build_targets


class CocoSingleObjectDataset(Dataset):
    """
    Minimal COCO dataset adapter for the current target builder.
    Uses exactly one GT bbox per image (largest area annotation).
    """

    def __init__(
        self,
        images_dir: str,
        annotations_path: str,
        image_size: int = 416,
        max_samples: int | None = None,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.annotations_path = Path(annotations_path)
        self.image_size = image_size

        with self.annotations_path.open("r", encoding="utf-8") as f:
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
            anns_by_image.setdefault(ann["image_id"], []).append(ann)

        self.samples: list[dict[str, Any]] = []
        for image_id, anns in anns_by_image.items():
            if image_id not in images:
                continue
            best_ann = max(anns, key=lambda a: float(a["bbox"][2] * a["bbox"][3]))
            self.samples.append({"image": images[image_id], "ann": best_ann})

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        image_info = sample["image"]
        ann = sample["ann"]

        img_path = self.images_dir / image_info["file_name"]
        image = read_image(str(img_path)).float() / 255.0  # [3, H, W]

        orig_h = int(image_info["height"])
        orig_w = int(image_info["width"])

        image = resize(image, [self.image_size, self.image_size], antialias=True)

        x, y, w, h = ann["bbox"]  # COCO px: top-left + width/height
        cx = (x + 0.5 * w) / orig_w
        cy = (y + 0.5 * h) / orig_h
        bw = w / orig_w
        bh = h / orig_h
        box_xywh = torch.tensor([cx, cy, bw, bh], dtype=torch.float32)

        cls = self.cat_to_idx[ann["category_id"]]
        class_idx = torch.tensor(cls, dtype=torch.long)

        return image, box_xywh, class_idx


class SyntheticDetectionDataset(Dataset):
    def __init__(self, n_samples: int = 128, image_size: int = 416, num_classes: int = 1) -> None:
        self.n_samples = n_samples
        self.image_size = image_size
        self.num_classes = num_classes

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image = torch.rand(3, self.image_size, self.image_size, dtype=torch.float32)
        cx = torch.rand(1).item() * 0.8 + 0.1
        cy = torch.rand(1).item() * 0.8 + 0.1
        bw = torch.rand(1).item() * 0.4 + 0.05
        bh = torch.rand(1).item() * 0.4 + 0.05
        box_xywh = torch.tensor([cx, cy, bw, bh], dtype=torch.float32)
        class_idx = torch.tensor(idx % self.num_classes, dtype=torch.long)
        return image, box_xywh, class_idx


class YoloV1Loss(nn.Module):
    def __init__(self, B: int, C: int, lambda_coord: float = 5.0, lambda_noobj: float = 0.5) -> None:
        super().__init__()
        self.B = B
        self.C = C
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> dict[str, torch.Tensor]:
        n = predictions.shape[0]

        pred_boxes = predictions[..., : self.B * 5].view(n, predictions.shape[1], predictions.shape[2], self.B, 5)
        tgt_boxes = targets[..., : self.B * 5].view(n, targets.shape[1], targets.shape[2], self.B, 5)

        pred_xywh = torch.sigmoid(pred_boxes[..., 0:4])
        tgt_xywh = tgt_boxes[..., 0:4]
        pred_conf = pred_boxes[..., 4]
        tgt_conf = tgt_boxes[..., 4]

        obj_mask = tgt_conf > 0
        noobj_mask = ~obj_mask

        coord_loss_raw = ((pred_xywh - tgt_xywh) ** 2).sum(dim=-1)
        coord_loss = coord_loss_raw[obj_mask].sum() if obj_mask.any() else predictions.new_zeros(())

        conf_loss_raw = self.bce(pred_conf, tgt_conf)
        obj_loss = conf_loss_raw[obj_mask].sum() if obj_mask.any() else predictions.new_zeros(())
        noobj_loss = conf_loss_raw[noobj_mask].sum() if noobj_mask.any() else predictions.new_zeros(())

        pred_cls = predictions[..., self.B * 5 :]
        tgt_cls = targets[..., self.B * 5 :]
        cell_obj_mask = obj_mask.any(dim=-1)
        cls_loss_raw = self.bce(pred_cls, tgt_cls).sum(dim=-1)
        cls_loss = cls_loss_raw[cell_obj_mask].sum() if cell_obj_mask.any() else predictions.new_zeros(())

        total = self.lambda_coord * coord_loss + obj_loss + self.lambda_noobj * noobj_loss + cls_loss
        normalizer = max(1, n)
        total = total / normalizer
        coord_loss = coord_loss / normalizer
        obj_loss = obj_loss / normalizer
        noobj_loss = noobj_loss / normalizer
        cls_loss = cls_loss / normalizer

        return {
            "total": total,
            "coord": coord_loss,
            "obj": obj_loss,
            "noobj": noobj_loss,
            "cls": cls_loss,
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
    B: int = 2
    C: int | None = None
    synthetic: bool = False


def _coerce_from_text(value: str, default: Any) -> Any:
    if default is None:
        txt = value.strip().lower()
        if txt in {"none", "null", ""}:
            return None
        try:
            return int(value)
        except ValueError:
            return value
    if isinstance(default, bool):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(default, int):
        return int(value)
    if isinstance(default, float):
        return float(value)
    return value


def load_config() -> TrainConfig:
    config_path = Path(os.environ.get("TRAIN_CONFIG", "config/train.yaml"))
    config = TrainConfig()

    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f) or {}
        for key, value in config_data.items():
            if not hasattr(config, key):
                raise ValueError(f"Unknown config key in {config_path}: {key}")
            setattr(config, key, value)

    for key, default in asdict(config).items():
        env_key = f"TRAIN_{key.upper()}"
        if env_key not in os.environ:
            continue
        setattr(config, key, _coerce_from_text(os.environ[env_key], default))

    return config


def make_dataset(config: TrainConfig) -> CocoSingleObjectDataset | SyntheticDetectionDataset:
    if config.synthetic:
        c = 1 if config.C is None else config.C
        return SyntheticDetectionDataset(
            n_samples=256 if config.max_samples is None else config.max_samples,
            image_size=config.image_size,
            num_classes=c,
        )

    images_dir = Path(config.images_dir)
    annotations = Path(config.annotations)
    if not images_dir.exists() or not annotations.exists():
        raise FileNotFoundError(
            "COCO paths not found. Extract zip files first, then run with "
            f"images_dir={config.images_dir} annotations={config.annotations}."
        )

    return CocoSingleObjectDataset(
        images_dir=str(images_dir),
        annotations_path=str(annotations),
        image_size=config.image_size,
        max_samples=config.max_samples,
    )


def make_val_dataset(config: TrainConfig) -> CocoSingleObjectDataset | SyntheticDetectionDataset:
    if config.synthetic:
        c = 1 if config.C is None else config.C
        return SyntheticDetectionDataset(
            n_samples=64 if config.val_max_samples is None else config.val_max_samples,
            image_size=config.image_size,
            num_classes=c,
        )

    images_dir = Path(config.val_images_dir)
    annotations = Path(config.val_annotations)
    if not images_dir.exists() or not annotations.exists():
        raise FileNotFoundError(
            "COCO val paths not found. Extract zip files first, then run with "
            f"val_images_dir={config.val_images_dir} val_annotations={config.val_annotations}."
        )

    return CocoSingleObjectDataset(
        images_dir=str(images_dir),
        annotations_path=str(annotations),
        image_size=config.image_size,
        max_samples=config.val_max_samples,
    )


def main() -> None:
    config = load_config()
    torch.manual_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = make_dataset(config)
    val_dataset = make_val_dataset(config)

    if config.C is None:
        inferred_c = getattr(train_dataset, "num_classes", 1)
        config.C = inferred_c
    num_classes = cast(int, config.C)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = YOLOv1(S=config.S, B=config.B, C=num_classes, base=config.base).to(device)
    criterion = YoloV1Loss(B=config.B, C=num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Training on {device} | train_samples={len(train_dataset)} | val_samples={len(val_dataset)} | "
        f"S={config.S} B={config.B} C={config.C} | batch={config.batch_size}"
    )

    for epoch in range(1, config.epochs + 1):
        model.train()
        running_total = 0.0

        for step, (images, boxes_xywh, classes) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            boxes_xywh = boxes_xywh.to(device, non_blocking=True)
            classes = classes.to(device, non_blocking=True)

            predictions = model(images)
            targets = build_targets(
                boxes_xywh_norm=boxes_xywh,
                classes=classes,
                S=config.S,
                B=config.B,
                C=num_classes,
                predictions=predictions.detach(),
            )

            losses = criterion(predictions, targets)
            loss = losses["total"]

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_total += float(loss.item())
            if step % config.log_every == 0 or step == len(train_loader):
                print(
                    f"epoch {epoch:03d} train step {step:04d}/{len(train_loader):04d} "
                    f"total={losses['total'].item():.4f} "
                    f"coord={losses['coord'].item():.4f} "
                    f"obj={losses['obj'].item():.4f} "
                    f"noobj={losses['noobj'].item():.4f} "
                    f"cls={losses['cls'].item():.4f}"
                )

        train_avg_loss = running_total / max(1, len(train_loader))

        model.eval()
        val_running_total = 0.0
        with torch.no_grad():
            for step, (images, boxes_xywh, classes) in enumerate(val_loader, start=1):
                images = images.to(device, non_blocking=True)
                boxes_xywh = boxes_xywh.to(device, non_blocking=True)
                classes = classes.to(device, non_blocking=True)

                predictions = model(images)
                targets = build_targets(
                    boxes_xywh_norm=boxes_xywh,
                    classes=classes,
                    S=config.S,
                    B=config.B,
                    C=num_classes,
                    predictions=predictions,
                )
                losses = criterion(predictions, targets)
                val_running_total += float(losses["total"].item())

                if step % config.log_every == 0 or step == len(val_loader):
                    print(
                        f"epoch {epoch:03d} val   step {step:04d}/{len(val_loader):04d} "
                        f"total={losses['total'].item():.4f}"
                    )

        val_avg_loss = val_running_total / max(1, len(val_loader))

        ckpt_path = out_dir / "last.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": asdict(config),
                "train_avg_loss": train_avg_loss,
                "val_avg_loss": val_avg_loss,
            },
            ckpt_path,
        )
        print(
            f"epoch {epoch:03d} done | train_avg_loss={train_avg_loss:.4f} "
            f"| val_avg_loss={val_avg_loss:.4f} | saved={ckpt_path}"
        )


if __name__ == "__main__":
    main()
