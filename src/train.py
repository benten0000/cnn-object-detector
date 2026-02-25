import argparse
import json
import tomllib
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a tiny YOLOv1-style detector.")
    parser.add_argument("--config", type=str, default="config/train.toml")
    parser.add_argument("--images-dir", type=str, default="data/coco/train2017")
    parser.add_argument("--annotations", type=str, default="data/coco/annotations/instances_train2017.json")
    parser.add_argument("--val-images-dir", type=str, default="data/coco/val2017")
    parser.add_argument("--val-annotations", type=str, default="data/coco/annotations/instances_val2017.json")
    parser.add_argument("--out-dir", type=str, default="runs/exp1")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=416)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--val-max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--base", type=int, default=32)
    parser.add_argument("--S", type=int, default=13)
    parser.add_argument("--B", type=int, default=2)
    parser.add_argument("--C", type=int, default=None)
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data for quick pipeline check.")
    return parser


def parse_args() -> argparse.Namespace:
    parser = _build_parser()
    pre_args, _ = parser.parse_known_args()

    cfg_path = Path(pre_args.config)
    if cfg_path.exists():
        with cfg_path.open("rb") as f:
            config_data = tomllib.load(f)
        parser.set_defaults(**config_data)

    return parser.parse_args()


def make_dataset(args: argparse.Namespace) -> CocoSingleObjectDataset | SyntheticDetectionDataset:
    if args.synthetic:
        c = 1 if args.C is None else args.C
        return SyntheticDetectionDataset(
            n_samples=256 if args.max_samples is None else args.max_samples,
            image_size=args.image_size,
            num_classes=c,
        )

    images_dir = Path(args.images_dir)
    annotations = Path(args.annotations)
    if not images_dir.exists() or not annotations.exists():
        raise FileNotFoundError(
            "COCO paths not found. Extract zip files first, then run with "
            f"--images-dir {args.images_dir} --annotations {args.annotations}."
        )

    return CocoSingleObjectDataset(
        images_dir=str(images_dir),
        annotations_path=str(annotations),
        image_size=args.image_size,
        max_samples=args.max_samples,
    )


def make_val_dataset(args: argparse.Namespace) -> CocoSingleObjectDataset | SyntheticDetectionDataset:
    if args.synthetic:
        c = 1 if args.C is None else args.C
        return SyntheticDetectionDataset(
            n_samples=64 if args.val_max_samples is None else args.val_max_samples,
            image_size=args.image_size,
            num_classes=c,
        )

    images_dir = Path(args.val_images_dir)
    annotations = Path(args.val_annotations)
    if not images_dir.exists() or not annotations.exists():
        raise FileNotFoundError(
            "COCO val paths not found. Extract zip files first, then run with "
            f"--val-images-dir {args.val_images_dir} --val-annotations {args.val_annotations}."
        )

    return CocoSingleObjectDataset(
        images_dir=str(images_dir),
        annotations_path=str(annotations),
        image_size=args.image_size,
        max_samples=args.val_max_samples,
    )


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = make_dataset(args)
    val_dataset = make_val_dataset(args)

    if args.C is None:
        inferred_c = getattr(train_dataset, "num_classes", 1)
        args.C = inferred_c

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = YOLOv1(S=args.S, B=args.B, C=args.C, base=args.base).to(device)
    criterion = YoloV1Loss(B=args.B, C=args.C)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Training on {device} | train_samples={len(train_dataset)} | val_samples={len(val_dataset)} | "
        f"S={args.S} B={args.B} C={args.C} | batch={args.batch_size}"
    )

    for epoch in range(1, args.epochs + 1):
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
                S=args.S,
                B=args.B,
                C=args.C,
                predictions=predictions.detach(),
            )

            losses = criterion(predictions, targets)
            loss = losses["total"]

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_total += float(loss.item())
            if step % args.log_every == 0 or step == len(train_loader):
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
                    S=args.S,
                    B=args.B,
                    C=args.C,
                    predictions=predictions,
                )
                losses = criterion(predictions, targets)
                val_running_total += float(losses["total"].item())

                if step % args.log_every == 0 or step == len(val_loader):
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
                "args": vars(args),
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
