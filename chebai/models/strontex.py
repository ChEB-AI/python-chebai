import abc
import torch
import typing
import numpy as np
import networkx as nx


FeatureType = typing.TypeVar("FeatureType")
LabelType = typing.TypeVar("LabelType")


class StrOntEx(torch.Module):
    def __init__(self, computation_graph):
        pass
