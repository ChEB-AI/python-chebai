from typing import Optional

import torch
import torch.nn.functional as F


class BCELogitLossWithValidLabels(torch.nn.BCEWithLogitsLoss):
    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        valid_label_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if valid_label_mask is None:
            return super().forward(input, target)

        loss_mat = F.binary_cross_entropy_with_logits(
            input,
            target,
            weight=self.weight,
            pos_weight=self.pos_weight,
            reduction="none",
        )
        loss_mat = loss_mat.masked_fill(~valid_label_mask, 0.0)

        if self.reduction == "mean":
            return loss_mat.sum() / valid_label_mask.sum().clamp_min(1)
        elif self.reduction == "sum":
            return loss_mat.sum()
        elif self.reduction == "none":
            return loss_mat
        raise ValueError(f"Invalid reduction mode: {self.reduction}")
