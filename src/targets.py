import torch


def build_targets(
    boxes_xywh_norm: torch.Tensor,  # [N, 4] (cx, cy, w, h) v [0..1]
    classes: torch.Tensor,          # [N] int
    S: int,
    B: int,
    C: int,
) -> torch.Tensor:
    """
    Target shape: [N, S, S, (B*5 + C)]
    V YOLOv1 je isti GT box običajno skopiran v oba B slot-a (enostaven start).
    """

    boxes_xywh_norm = boxes_xywh_norm.to(device=torch.device("cuda"), dtype=torch.float32)
    classes = classes.to(device=torch.device("cuda"), dtype=torch.long)

    N = boxes_xywh_norm.shape[0]
    t = torch.zeros((N, S, S, B * 5 + C), device=torch.device("cuda"), dtype=torch.float32)

    cx = boxes_xywh_norm[:, 0].clamp(0, 1 - 1e-6)
    cy = boxes_xywh_norm[:, 1].clamp(0, 1 - 1e-6)
    w = boxes_xywh_norm[:, 2].clamp(1e-6, 1)
    h = boxes_xywh_norm[:, 3].clamp(1e-6, 1)

    gi = torch.floor(cx * S).long()  # x index
    gj = torch.floor(cy * S).long()  # y index

    x_off = cx * S - gi.float()
    y_off = cy * S - gj.float()

    n_idx = torch.arange(N, device=torch.device("cuda"))
    box_values = torch.stack(
        [x_off, y_off, w, h, torch.ones_like(x_off)], dim=-1
    )  # [N, 5]

    # Fill all B bbox slots with the same GT box (simple YOLOv1 start).
    for b in range(B):
        base = b * 5
        t[n_idx, gj, gi, base : base + 5] = box_values

    # One-hot class target in the responsible cell.
    t[n_idx, gj, gi, B * 5 + classes] = 1.0

    return t
