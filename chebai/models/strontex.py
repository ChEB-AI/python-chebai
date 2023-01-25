import abc
import torch
import typing
from chebai.models.electra import Electra
from chebai.models.lnn_model import LNN
from chebai.models.base import JCIBaseNet
from lnn import Loss as LNNLoss

FeatureType = typing.TypeVar("FeatureType")
LabelType = typing.TypeVar("LabelType")


class SplitLoss:
    def __init__(self):
        self.bce_wl = torch.nn.BCEWithLogitsLoss()

    def __call__(self, target, input):
        lnn_loss_fn = input["lnn_loss_fn"]
        electra_loss = self.bce_wl(target=target, input=input["electra_logits"])
        lnn_loss = lnn_loss_fn(target=target, input=input["lnn_predictions"])
        return electra_loss + lnn_loss[0]

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
            electra_logits=out_l2["logits"],
            lnn_predictions = out_l3
        )

    def _get_data_for_loss(self, model_output, labels):
        return dict(input=model_output, target=labels.float(), lnn_loss_fn=self.layer3.lnn.loss_fn([LNNLoss.SUPERVISED]))

    def _get_prediction_and_labels(self, data, labels, output):
        return output["lnn_predictions"], labels


