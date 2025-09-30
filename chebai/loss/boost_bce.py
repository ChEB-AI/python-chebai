import torch
import sys
sys.path.insert(1,'/home/programmer/Bachelorarbeit/python-chebai')

import extras.weight_loader as f


class BCE_Point_boosting(torch.nn.BCEWithLogitsLoss):
    
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(reduction=None,**kwargs)

    def forward(
            self,
            input: torch.Tensor,
            target: torch.Tensor,
            **kwargs
            
    )-> torch.Tensor:
        weights = kwargs['weights']
        loss = super().forward(input=input,target=target)
        weights_tensor = f.create_weight_tensor(weights)
        loss_scaled = torch.matmul(weights_tensor,loss)
        return torch.mean(loss_scaled)

        
        
        
        
