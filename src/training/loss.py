import torch
import torch.nn as nn


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

