import torch
from torch import nn


class MixedDataLoss(nn.Module):
    """
    A wrapper for applying a base loss function to a subset of input data.

    This class allows for selective application of a loss function based on the provided
    non-null labels.

    Args:
        base_loss (nn.Module): The base loss function to be applied.
    """

    def __init__(self, base_loss: nn.Module):
        super().__init__()
        self.base_loss = base_loss

    def forward(
        self, input: torch.Tensor, target: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """
        Forward pass for applying the base loss function.

        Args:
            input (torch.Tensor): The input tensor (predictions).
            target (torch.Tensor): The target tensor (labels).
            **kwargs: Additional keyword arguments. The 'non_null_labels' key can be used
                      to specify the indices of the non-null labels.

        Returns:
            torch.Tensor: The computed loss.
        """
        nnl = kwargs.pop("non_null_labels", None)
        if nnl:
            inp = input[nnl]
        else:
            inp = input
        return self.base_loss(inp, target, **kwargs)
