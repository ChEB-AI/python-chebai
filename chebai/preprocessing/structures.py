from typing import Any, Tuple, Union

import networkx as nx
import torch


class XYData(torch.utils.data.Dataset):
    """
    A dataset class for handling pairs of data (x, y).

    Args:
        x: Input data.
        y: Target data.
        kwargs: Additional fields to store in the dataset.
    """

    def __init__(
        self, x: Union[torch.Tensor, Tuple[Any, ...]], y: torch.Tensor, **kwargs
    ):
        super().__init__()
        self.additional_fields = kwargs
        self.x = x
        self.y = y

    def __getitem__(self, index: int):
        """Returns the data and target at the given index."""
        return self.x[index], self.y[index]

    def __len__(self) -> int:
        """Returns the size of the dataset."""
        return len(self.x)

    def to_x(self, device: torch.device) -> Union[torch.Tensor, Tuple[Any, ...]]:
        """
        Moves the input data to the specified device.

        Args:
            device: The device to move the data to.

        Returns:
            The input data on the specified device.
        """
        if isinstance(self.x, tuple):
            res = []
            for elem in self.x:
                if isinstance(elem, dict):
                    for k, v in elem.items():
                        elem[k] = v.to(device) if v is not None else None
                else:
                    elem = elem.to(device)
                res.append(elem)
            return tuple(res)
        return self.x.to(device)

    def to_y(self, device: torch.device) -> torch.Tensor:
        """
        Moves the target data to the specified device.

        Args:
            device: The device to move the data to.

        Returns:
            The target data on the specified device.
        """
        return self.y.to(device)

    def _to_if_tensor(self, obj: Any, device: torch.device) -> Any:
        """
        Recursively moves the object to the specified device if it is a tensor.

        Args:
            obj: The object to move.
            device: The device to move the object to.

        Returns:
            The object on the specified device.
        """
        if isinstance(obj, torch.Tensor):
            return obj.to(device)
        elif isinstance(obj, dict):
            return {k: self._to_if_tensor(v, device) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._to_if_tensor(v, device) for v in obj]
        else:
            return obj

    def to(self, device: torch.device) -> "XYData":
        """
        Moves the dataset to the specified device.

        Args:
            device: The device to move the dataset to.

        Returns:
            A new dataset on the specified device.
        """
        x = self.to_x(device)
        if self.y is not None:
            y = self.to_y(device)
        else:
            y = None
        return XYData(
            x,
            y,
            **{
                k: self._to_if_tensor(v, device)
                for k, v in self.additional_fields.items()
            },
        )


class XYMolData(XYData):
    """
    A dataset class for handling molecular data represented as NetworkX graphs.

    Args:
        x: Input molecular graphs.
        y: Target data.
        kwargs: Additional fields to store in the dataset.
    """

    def to_x(self, device: torch.device) -> Tuple[nx.Graph, ...]:
        """
        Moves the node attributes of the molecular graphs to the specified device.

        Args:
            device: The device to move the data to.

        Returns:
            A tuple of molecular graphs with node attributes on the specified device.
        """
        l = []
        for g in self.x:
            graph = g.copy()
            nx.set_node_attributes(
                graph,
                {k: v.to(device) for k, v in nx.get_node_attributes(g, "x").items()},
                "x",
            )
            l.append(graph)
        return tuple(l)
