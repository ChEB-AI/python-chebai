import torch
import extras.weight_loader as f


class BCE_Boosting(torch.nn.BCEWithLogitsLoss):
    
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(reduction='none',**kwargs)

    def forward(
            self,
            input: torch.Tensor,
            target: torch.Tensor,
            **kwargs
            
    )-> torch.Tensor:
        weights = kwargs['weights']
        loss = super().forward(input=input,target=target)
        loss_scaled = loss * weights
        return torch.mean(loss_scaled)