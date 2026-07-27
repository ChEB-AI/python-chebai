from contextlib import nullcontext

import torch
import torch.nn as nn

# Asymmetric Loss (ASL) for multi-label classification, following the official
# implementation by Ridnik et al. (https://github.com/Alibaba-MIIL/ASL), with the
# reduction changed from 'sum' to 'mean' for consistency with the other loss functions
# in this framework.


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for multi-label classification.

    ASL addresses the imbalance between positive and negative labels by decoupling the
    focusing parameters for the two cases and by clipping away the contribution of very
    easy negatives:

        L_ASL = -(1 - p_t)^gamma_t * log(p_t)

    with the shifted and clipped probability p_m = max(p - clip, 0) and

        p_t     = p        if y = 1,   1 - p_m  if y = 0
        gamma_t = gamma_pos if y = 1,  gamma_neg if y = 0

    Negatives whose predicted probability does not exceed ``clip`` therefore contribute
    exactly zero to the loss and to the gradient. With ``gamma_pos = gamma_neg = 0`` the
    loss reduces to binary cross-entropy with probability margin clipping, and with
    ``clip = 0`` it further reduces to plain binary cross-entropy.

    The defaults correspond to the best-performing configuration found for ChEBI
    classification (gamma_pos = gamma_neg = 0, clip = 0.15), where probability margin
    clipping alone outperformed every focusing configuration.

    Args:
        gamma_pos (float, optional): Focusing parameter gamma+ for positive labels. Default is 0.0.
        gamma_neg (float, optional): Focusing parameter gamma- for negative labels. Default is 0.0.
        clip (float, optional): Probability margin m used to shift and clip the negative
            probabilities. Default is 0.15.
        eps (float, optional): Lower clamp applied inside the logarithm for numerical
            stability. Default is 1e-8.
        disable_torch_grad_focal_loss (bool, optional): If True, the modulating factor is
            computed without tracking gradients (as in the official implementation).
            Default is True.
        reduction (str, optional): Specifies the reduction method: 'none' | 'mean' | 'sum'.
            Default is 'mean'.
    """

    def __init__(
        self,
        gamma_pos: float = 0.0,
        gamma_neg: float = 0.0,
        clip: float = 0.15,
        eps: float = 1e-8,
        disable_torch_grad_focal_loss: bool = True,
        reduction: str = "mean",
    ):
        super().__init__()
        if reduction not in ("none", "mean", "sum"):
            raise ValueError(
                f"Unsupported reduction '{reduction}'. Use 'none', 'mean' or 'sum'."
            )
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.eps = eps
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.reduction = reduction

    def forward(
        self, input: torch.Tensor, target: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """
        Forward pass for the loss calculation.

        Args:
            input (torch.Tensor): The input tensor (logits), shape (batch_size, num_classes).
            target (torch.Tensor): The target tensor (multi-hot labels), same shape as input.

        Returns:
            torch.Tensor: The computed loss.
        """
        # Additional kwargs (e.g. current_epoch) are not used by this loss
        target = target.float()

        probs = torch.sigmoid(input)
        xs_pos = probs
        xs_neg = 1 - probs

        # Probability margin clipping: 1 - max(p - clip, 0)
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic cross-entropy calculation
        loss = target * torch.log(xs_pos.clamp(min=self.eps)) + (
            1 - target
        ) * torch.log(xs_neg.clamp(min=self.eps))

        # Asymmetric focusing
        if self.gamma_pos > 0 or self.gamma_neg > 0:
            # The official implementation toggles torch.set_grad_enabled here, which
            # unconditionally re-enables grad tracking and would leak it into a
            # surrounding torch.no_grad() block (e.g. during validation). A no_grad
            # context is numerically identical but restores the ambient state.
            grad_ctx = (
                torch.no_grad() if self.disable_torch_grad_focal_loss else nullcontext()
            )
            with grad_ctx:
                p_t = xs_pos * target + xs_neg * (1 - target)
                gamma_t = self.gamma_pos * target + self.gamma_neg * (1 - target)
                modulating_factor = torch.pow(1 - p_t, gamma_t)
            loss = loss * modulating_factor

        loss = -loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
