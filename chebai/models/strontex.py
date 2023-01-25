import abc
import torch
import typing
from chebai.models.electra import Electra
from chebai.models.lnn_model import LNN
from chebai.models.base import JCIBaseNet


FeatureType = typing.TypeVar("FeatureType")
LabelType = typing.TypeVar("LabelType")


class ElectraLNN(JCIBaseNet):
    NAME = "Electra+LNN"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layer2 = Electra(**kwargs["electra_kwargs"])
        self.layer3 = LNN(**kwargs["lnn_kwargs"])


    def forward(self, data, **kwargs):
        out_l2 = self.layer2(data, **kwargs)
        out_l3 = self.layer3(dict(features=torch.sigmoid(out_l2["logits"])))
        return dict(
            predictions=out_l3,
        )

