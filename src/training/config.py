import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml


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

