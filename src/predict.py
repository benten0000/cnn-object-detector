import argparse
from pathlib import Path

import torch
from torchvision.io import read_image, write_png
from torchvision.ops import nms
from torchvision.transforms.functional import resize
from torchvision.utils import draw_bounding_boxes

from src.model import YOLOv1


def _to_3ch(image: torch.Tensor) -> torch.Tensor:
    if image.ndim == 2:
        image = image.unsqueeze(0)
    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)
    elif image.shape[0] > 3:
        image = image[:3]
    elif image.shape[0] < 3:
        image = torch.cat([image, image[:1].repeat(3 - image.shape[0], 1, 1)], dim=0)
    return image


def decode_predictions(
    pred: torch.Tensor,  # [S, S, 5+C]
    score_thresh: float,
    iou_thresh: float,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    S = int(pred.shape[0])
    xywh = torch.sigmoid(pred[..., :4])
    obj = torch.sigmoid(pred[..., 4])
    cls_prob = torch.sigmoid(pred[..., 5:])
    cls_score, cls_idx = cls_prob.max(dim=-1)
    score = obj * cls_score

    keep = score >= score_thresh
    if not keep.any():
        empty_boxes = pred.new_zeros((0, 4))
        empty_scores = pred.new_zeros((0,))
        empty_cls = torch.zeros((0,), dtype=torch.long, device=pred.device)
        return empty_boxes, empty_scores, empty_cls

    gy, gx = torch.where(keep)
    sel = xywh[gy, gx]
    cx = (sel[:, 0] + gx.float()) / float(S)
    cy = (sel[:, 1] + gy.float()) / float(S)
    w = sel[:, 2].clamp(1e-6, 1.0)
    h = sel[:, 3].clamp(1e-6, 1.0)

    x1 = (cx - 0.5 * w).clamp(0.0, 1.0)
    y1 = (cy - 0.5 * h).clamp(0.0, 1.0)
    x2 = (cx + 0.5 * w).clamp(0.0, 1.0)
    y2 = (cy + 0.5 * h).clamp(0.0, 1.0)
    boxes = torch.stack([x1, y1, x2, y2], dim=-1)
    scores = score[gy, gx]
    classes = cls_idx[gy, gx]

    keep_nms = nms(boxes, scores, iou_thresh=iou_thresh)
    if topk > 0:
        keep_nms = keep_nms[:topk]

    return boxes[keep_nms], scores[keep_nms], classes[keep_nms]


def load_model(args: argparse.Namespace, device: torch.device) -> tuple[YOLOv1, int, int]:
    S = args.S
    C = args.C
    base = args.base
    model = None

    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
        cfg = ckpt.get("config", {})
        S = int(cfg.get("S", S))
        cfg_c = cfg.get("C", C)
        C = int(cfg_c) if cfg_c is not None else int(C)
        base = int(cfg.get("base", base))
        model = YOLOv1(S=S, C=C, base=base).to(device)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
    else:
        model = YOLOv1(S=S, C=C, base=base).to(device)

    model.eval()
    return model, S, C


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple YOLOv1 image inference and bbox drawing.")
    parser.add_argument("--image", type=str, required=True, help="Input image path.")
    parser.add_argument("--out", type=str, default="prediction.png", help="Output image path.")
    parser.add_argument("--ckpt", type=str, default=None, help="Checkpoint path. Omit for untrained model.")
    parser.add_argument("--image-size", type=int, default=416, help="Inference resize size.")
    parser.add_argument("--S", type=int, default=13, help="Grid size used when no checkpoint is loaded.")
    parser.add_argument("--C", type=int, default=80, help="Number of classes used when no checkpoint is loaded.")
    parser.add_argument("--base", type=int, default=32, help="Backbone base channels when no checkpoint is loaded.")
    parser.add_argument("--score-thresh", type=float, default=0.35, help="Score threshold.")
    parser.add_argument("--iou-thresh", type=float, default=0.50, help="NMS IoU threshold.")
    parser.add_argument("--topk", type=int, default=50, help="Keep top-K after NMS.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, _ = load_model(args, device)

    image_u8 = read_image(args.image)
    image_u8 = _to_3ch(image_u8)
    image_f = image_u8.float().div(255.0)
    image_f = resize(image_f, [args.image_size, args.image_size], antialias=True)
    input_tensor = image_f.unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(input_tensor)[0]  # [S, S, 5+C]
        boxes_norm, scores, classes = decode_predictions(
            pred=pred,
            score_thresh=args.score_thresh,
            iou_thresh=args.iou_thresh,
            topk=args.topk,
        )

    vis = (image_f * 255.0).clamp(0, 255).to(torch.uint8).cpu()
    if boxes_norm.numel() > 0:
        h, w = vis.shape[1], vis.shape[2]
        boxes_px = boxes_norm.cpu().clone()
        boxes_px[:, [0, 2]] *= float(w)
        boxes_px[:, [1, 3]] *= float(h)
        labels = [f"c{int(c.item())}:{float(s.item()):.2f}" for c, s in zip(classes.cpu(), scores.cpu())]
        vis = draw_bounding_boxes(vis, boxes=boxes_px, labels=labels, width=2, colors="red")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_png(vis, str(out_path))
    print(f"saved={out_path} | boxes={int(boxes_norm.shape[0])} | ckpt={'yes' if args.ckpt else 'no'}")


if __name__ == "__main__":
    main()
