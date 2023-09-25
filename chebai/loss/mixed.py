from torch import nn


class MixedDataLoss(nn.Module):

    def __init__(self, base_loss: nn.Module):
        super().__init__()
        self.base_loss = base_loss

    def forward(self, input, target, **kwargs):
        nnl = kwargs.pop("non_null_labels", None)
        if nnl:
            inp = input[nnl]
        else:
            inp = input
        return self.base_loss(inp, target, **kwargs)