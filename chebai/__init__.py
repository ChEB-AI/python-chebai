import os
from typing import Any

import torch

# Get the absolute path of the current file's directory
MODULE_PATH = os.path.abspath(os.path.dirname(__file__))


class CustomTensor(torch.Tensor):
    """
    A custom tensor class inheriting from `torch.Tensor`.

    This class allows for the creation of tensors using the provided data.

    Attributes:
        data (Any): The data to be converted into a tensor.
    """

    def __new__(cls, data: Any) -> "CustomTensor":
        """
        Creates a new instance of CustomTensor.

        Args:
            data (Any): The data to be converted into a tensor.

        Returns:
            CustomTensor: A tensor containing the provided data.
        """
        return torch.tensor(data)
