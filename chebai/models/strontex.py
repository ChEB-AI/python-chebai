import abc
import typing

import networkx as nx
import numpy as np
import torch

FeatureType = typing.TypeVar("FeatureType")
LabelType = typing.TypeVar("LabelType")


class StrOntEx(torch.Module):
    def __init__(self, computation_graph):
        pass
