from typing import Optional

import torch


class BCELogitLossWithValidLabels(torch.nn.BCEWithLogitsLoss):
    def __init__(self, **kwargs):
        kwargs["reduction"] = "none"
        super().__init__(**kwargs)

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        valid_label_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        loss_mat = super().forward(input, target)

        if valid_label_mask is None:
            return loss_mat.mean()

        loss_mat = torch.where(
            valid_label_mask,
            loss_mat,
            torch.zeros_like(loss_mat),
        )

        return loss_mat.sum() / valid_label_mask.sum().clamp_min(1)
