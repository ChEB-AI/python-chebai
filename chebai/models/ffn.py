from typing import Dict, Any, Tuple

from chebai.models import ChebaiBaseNet
import torch
from torch import Tensor


class FFN(ChebaiBaseNet):

    NAME = "FFN"

    def __init__(
        self,
        input_size: int = 1000,
        num_hidden_layers: int = 3,
        hidden_size: int = 128,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.layers = torch.nn.ModuleList()
        self.layers.append(torch.nn.Linear(input_size, hidden_size))
        for _ in range(num_hidden_layers):
            self.layers.append(torch.nn.Linear(hidden_size, hidden_size))
        self.layers.append(torch.nn.Linear(hidden_size, self.out_dim))

    def _get_prediction_and_labels(self, data, labels, model_output):
        d = model_output["logits"]
        loss_kwargs = data.get("loss_kwargs", dict())
        if "non_null_labels" in loss_kwargs:
            n = loss_kwargs["non_null_labels"]
            d = data[n]
        return torch.sigmoid(d), labels.int() if labels is not None else None

    def _process_for_loss(
        self,
        model_output: Dict[str, Tensor],
        labels: Tensor,
        loss_kwargs: Dict[str, Any],
    ) -> Tuple[Tensor, Tensor, Dict[str, Any]]:
        """
        Process the model output for calculating the loss.

        Args:
            model_output (Dict[str, Tensor]): The output of the model.
            labels (Tensor): The target labels.
            loss_kwargs (Dict[str, Any]): Additional loss arguments.

        Returns:
            tuple: A tuple containing the processed model output, labels, and loss arguments.
        """
        kwargs_copy = dict(loss_kwargs)
        if labels is not None:
            labels = labels.float()
        return model_output["logits"], labels, kwargs_copy

    def forward(self, data, **kwargs):
        x = data["features"]
        for layer in self.layers:
            x = torch.relu(layer(x))
        return {"logits": x}
