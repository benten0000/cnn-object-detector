import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
from torchvision.transforms.functional import resize

from src.training.config import TrainConfig


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

        anns_by_image: dict[int, dict[int, dict[str, Any]]] = {}
        for ann_idx, ann in enumerate(coco["annotations"]):
            if ann.get("iscrowd", 0) == 1:
                continue
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            if ann["image_id"] in images:
                image_anns = anns_by_image.setdefault(ann["image_id"], {})
                ann_id = int(ann.get("id", ann_idx))
                while ann_id in image_anns:
                    ann_id += 1
                image_anns[ann_id] = ann

        self.samples = [(images[image_id], anns) for image_id, anns in anns_by_image.items()]
        if max_samples is not None:
            self.samples = self.samples[:max_samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image_info, anns = self.samples[idx]
        image = read_image(str(self.images_dir / image_info["file_name"])).float().div_(255.0)
        if image.ndim == 2:
            image = image.unsqueeze(0)
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        elif image.shape[0] > 3:
            image = image[:3]
        elif image.shape[0] < 3:
            image = torch.cat([image, image[:1].repeat(3 - image.shape[0], 1, 1)], dim=0)
        image = resize(image, [self.image_size, self.image_size], antialias=True)

        h = float(image_info["height"])
        w = float(image_info["width"])

        boxes = []
        classes = []
        for ann in anns.values():
            x, y, bw, bh = ann["bbox"]
            boxes.append([(x + 0.5 * bw) / w, (y + 0.5 * bh) / h, bw / w, bh / h])
            classes.append(self.cat_to_idx[ann["category_id"]])

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


def make_dataset(cfg: TrainConfig, *, train: bool) -> CocoDetectionDataset:
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
