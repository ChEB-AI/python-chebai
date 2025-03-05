from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn

from chebai.models import ChebaiBaseNet


class FFN(ChebaiBaseNet):
    # Reference: https://github.com/bio-ontology-research-group/deepgo2/blob/main/deepgo/models.py#L121-L139

    NAME = "FFN"

    def __init__(
        self,
        input_size: int,
        hidden_layers: List[int] = [
            1024,
        ],
        **kwargs
    ):
        super().__init__(**kwargs)

        layers = []
        current_layer_input_size = input_size
        for hidden_dim in hidden_layers:
            layers.append(MLPBlock(current_layer_input_size, hidden_dim))
            layers.append(Residual(MLPBlock(hidden_dim, hidden_dim)))
            current_layer_input_size = hidden_dim

        layers.append(torch.nn.Linear(current_layer_input_size, self.out_dim))
        layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)

    def _get_prediction_and_labels(self, data, labels, model_output):
        d = model_output["logits"]
        loss_kwargs = data.get("loss_kwargs", dict())
        if "non_null_labels" in loss_kwargs:
            n = loss_kwargs["non_null_labels"]
            d = d[n]
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
        return {"logits": self.model(x)}


class Residual(nn.Module):
    """
    A residual layer that adds the output of a function to its input.

    Args:
        fn (nn.Module): The function to be applied to the input.

    References:
        https://github.com/bio-ontology-research-group/deepgo2/blob/main/deepgo/base.py#L6-L35
    """

    def __init__(self, fn):
        """
        Initialize the Residual layer with a given function.

        Args:
            fn (nn.Module): The function to be applied to the input.
        """
        super().__init__()
        self.fn = fn

    def forward(self, x):
        """
        Forward pass of the Residual layer.

        Args:
            x: Input tensor.

        Returns:
            torch.Tensor: The input tensor added to the result of applying the function `fn` to it.
        """
        return x + self.fn(x)


class MLPBlock(nn.Module):
    """
    A basic Multi-Layer Perceptron (MLP) block with one fully connected layer.

    Args:
        in_features (int): The number of input features.
        output_size (int): The number of output features.
        bias (boolean): Add bias to the linear layer
        layer_norm (boolean): Apply layer normalization
        dropout (float): The dropout value
        activation (nn.Module): The activation function to be applied after each fully connected layer.

    References:
        https://github.com/bio-ontology-research-group/deepgo2/blob/main/deepgo/base.py#L38-L73

    Example:
    ```python
    # Create an MLP block with 2 hidden layers and ReLU activation
    mlp_block = MLPBlock(input_size=64, output_size=10, activation=nn.ReLU())

    # Apply the MLP block to an input tensor
    input_tensor = torch.randn(32, 64)
    output = mlp_block(input_tensor)
    ```
    """

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        layer_norm=True,
        dropout=0.1,
        activation=nn.ReLU,
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias)
        self.activation = activation()
        self.layer_norm: Optional[nn.LayerNorm] = (
            nn.LayerNorm(out_features) if layer_norm else None
        )
        self.dropout: Optional[nn.Dropout] = nn.Dropout(dropout) if dropout else None

    def forward(self, x):
        x = self.activation(self.linear(x))
        if self.layer_norm:
            x = self.layer_norm(x)
        if self.dropout:
            x = self.dropout(x)
        return x
