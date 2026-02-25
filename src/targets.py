import torch


def build_targets(
    boxes_xywh_norm: torch.Tensor,  # [N, M, 4]
    classes: torch.Tensor,  # [N, M]
    S: int,
    C: int,
    valid_mask: torch.Tensor | None = None,  # [N, M]
) -> torch.Tensor:
    device = boxes_xywh_norm.device
    boxes = boxes_xywh_norm.to(device=device, dtype=torch.float32)
    cls = classes.to(device=device, dtype=torch.long)
    mask = (
        torch.ones(cls.shape, device=device, dtype=torch.bool)
        if valid_mask is None
        else valid_mask.to(device=device, dtype=torch.bool)
    )

    n = boxes.shape[0]
    targets = torch.zeros((n, S, S, 5 + C), device=device, dtype=torch.float32)
    best_area = torch.zeros((n, S, S), device=device, dtype=torch.float32)

    for i in range(n):
        for j in torch.nonzero(mask[i], as_tuple=False).flatten().tolist():
            cx, cy, w, h = boxes[i, j]
            cx = cx.clamp(0.0, 1.0 - 1e-6)
            cy = cy.clamp(0.0, 1.0 - 1e-6)
            w = w.clamp(1e-6, 1.0)
            h = h.clamp(1e-6, 1.0)

            gx = int(torch.floor(cx * S).item())
            gy = int(torch.floor(cy * S).item())
            area = float((w * h).item())
            if area <= float(best_area[i, gy, gx].item()):
                continue

            cell = targets[i, gy, gx]
            cell.zero_()
            cell[0] = cx * S - gx
            cell[1] = cy * S - gy
            cell[2] = w
            cell[3] = h
            cell[4] = 1.0
            cell[5 + int(cls[i, j].item())] = 1.0
            best_area[i, gy, gx] = area

    return targets
