import typing

import torch

FeatureType = typing.TypeVar("FeatureType")
LabelType = typing.TypeVar("LabelType")


class StrOntEx(torch.Module):
    def __init__(self, computation_graph):
        pass
