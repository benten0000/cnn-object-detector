import torch


def _xywh_to_xyxy(box_xywh: torch.Tensor) -> torch.Tensor:
    # box_xywh: [..., 4] with (cx, cy, w, h), all normalized to [0, 1]
    cx, cy, w, h = (
        box_xywh[..., 0],
        box_xywh[..., 1],
        box_xywh[..., 2],
        box_xywh[..., 3],
    )
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)


def _iou_one_to_many(gt_xyxy: torch.Tensor, boxes_xyxy: torch.Tensor) -> torch.Tensor:
    # gt_xyxy: [4], boxes_xyxy: [B, 4] -> IoU: [B]
    inter_x1 = torch.maximum(gt_xyxy[0], boxes_xyxy[:, 0])
    inter_y1 = torch.maximum(gt_xyxy[1], boxes_xyxy[:, 1])
    inter_x2 = torch.minimum(gt_xyxy[2], boxes_xyxy[:, 2])
    inter_y2 = torch.minimum(gt_xyxy[3], boxes_xyxy[:, 3])

    inter_w = (inter_x2 - inter_x1).clamp(min=0.0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0.0)
    inter_area = inter_w * inter_h

    gt_area = ((gt_xyxy[2] - gt_xyxy[0]).clamp(min=0.0) * (gt_xyxy[3] - gt_xyxy[1]).clamp(min=0.0))
    box_areas = ((boxes_xyxy[:, 2] - boxes_xyxy[:, 0]).clamp(min=0.0) * (boxes_xyxy[:, 3] - boxes_xyxy[:, 1]).clamp(min=0.0))
    union = gt_area + box_areas - inter_area
    return inter_area / union.clamp(min=1e-6)


def build_targets(
    boxes_xywh_norm: torch.Tensor,  # [N, 4] (cx, cy, w, h) normalized in [0..1]
    classes: torch.Tensor,          # [N] class indices
    S: int,
    B: int,
    C: int,
    predictions: torch.Tensor | None = None,  # [N, S, S, B*5+C] raw model output
) -> torch.Tensor:
    """
    Target shape: [N, S, S, (B*5 + C)]
    YOLOv1-style assignment: only one bbox slot per object is responsible.
    Responsible slot is selected by max IoU inside the ground-truth cell.
    """

    if predictions is not None:
        device = predictions.device
    else:
        device = boxes_xywh_norm.device

    boxes_xywh_norm = boxes_xywh_norm.to(device=device, dtype=torch.float32)  # [N, 4]
    classes = classes.to(device=device, dtype=torch.long)  # [N]

    N = boxes_xywh_norm.shape[0]  # scalar
    t = torch.zeros((N, S, S, B * 5 + C), device=device, dtype=torch.float32)  # [N, S, S, B*5+C]

    cx = boxes_xywh_norm[:, 0].clamp(0, 1 - 1e-6)  # [N]
    cy = boxes_xywh_norm[:, 1].clamp(0, 1 - 1e-6)  # [N]
    w = boxes_xywh_norm[:, 2].clamp(1e-6, 1)       # [N]
    h = boxes_xywh_norm[:, 3].clamp(1e-6, 1)       # [N]

    gi = torch.floor(cx * S).long()  # [N] x-cell index in [0, S-1]
    gj = torch.floor(cy * S).long()  # [N] y-cell index in [0, S-1]

    x_off = cx * S - gi.float()  # [N] x offset inside selected cell
    y_off = cy * S - gj.float()  # [N] y offset inside selected cell

    n_idx = torch.arange(N, device=device)  # [N] batch indices

    # YOLOv1 responsible predictor: set GT only for one bbox slot (max IoU), others keep conf=0.
    for n in range(N):
        gti = int(gi[n].item())
        gtj = int(gj[n].item())
        gt_box_xywh = torch.stack([cx[n], cy[n], w[n], h[n]])  # [4]
        gt_box_xyxy = _xywh_to_xyxy(gt_box_xywh)  # [4]

        if predictions is None:
            responsible_b = 0
        else:
            pred_cell = predictions[n, gtj, gti]  # [B*5 + C]
            pred_boxes_xywh = []
            for b in range(B):
                base = b * 5
                # Interpret prediction head values as normalized box params for assignment.
                px_off = torch.sigmoid(pred_cell[base + 0]).clamp(0.0, 1.0)
                py_off = torch.sigmoid(pred_cell[base + 1]).clamp(0.0, 1.0)
                pw = torch.sigmoid(pred_cell[base + 2]).clamp(1e-6, 1.0)
                ph = torch.sigmoid(pred_cell[base + 3]).clamp(1e-6, 1.0)

                pcx = (gti + px_off) / S
                pcy = (gtj + py_off) / S
                pred_boxes_xywh.append(torch.stack([pcx, pcy, pw, ph]))

            pred_boxes_xywh = torch.stack(pred_boxes_xywh, dim=0)  # [B, 4]
            pred_boxes_xyxy = _xywh_to_xyxy(pred_boxes_xywh)  # [B, 4]
            ious = _iou_one_to_many(gt_box_xyxy, pred_boxes_xyxy)  # [B]
            responsible_b = int(torch.argmax(ious).item())

        base = responsible_b * 5
        t[n, gtj, gti, base + 0] = x_off[n]
        t[n, gtj, gti, base + 1] = y_off[n]
        t[n, gtj, gti, base + 2] = w[n]
        t[n, gtj, gti, base + 3] = h[n]
        t[n, gtj, gti, base + 4] = 1.0

    # One-hot class target in the responsible cell.
    t[n_idx, gj, gi, B * 5 + classes] = 1.0  # indexed write of shape [N]

    return t
