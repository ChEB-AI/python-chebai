import os

import torch

MODULE_PATH = os.path.abspath(os.path.dirname(__file__))


class CustomTensor(torch.Tensor):
    def __new__(cls, data):
        return torch.tensor(data)
