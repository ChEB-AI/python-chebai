from tempfile import TemporaryDirectory
import logging

from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import (
    ElectraConfig,
    ElectraForMultipleChoice,
    ElectraForPreTraining,
    ElectraForSequenceClassification,
    PretrainedConfig,
)
import torch

from chebai.models.base import JCIBaseNet

logging.getLogger("pysmiles").setLevel(logging.CRITICAL)


class ElectraPre(JCIBaseNet):
    NAME = "ElectraPre"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.config = ElectraConfig(**kwargs["config"])
        self.electra = ElectraForPreTraining(self.config)

    def forward(self, data):
        x = data.x
        x = self.electra(x)
        return x.logits


class Electra(JCIBaseNet):
    NAME = "Electra"

    def __init__(self, **kwargs):
        # Remove this property in order to prevent it from being stored as a
        # hyper parameter
        pretrained_checkpoint = kwargs.pop("pretrained_checkpoint") or None
        super().__init__(**kwargs)
        self.config = ElectraConfig(**kwargs["config"], output_attentions=True)

        if pretrained_checkpoint:
            elpre = ElectraPre.load_from_checkpoint(kwargs["pretrained_checkpoint"])
            with TemporaryDirectory() as td:
                elpre.electra.save_pretrained(td)
                self.electra = ElectraForSequenceClassification.from_pretrained(
                    td, config=self.config
                )
                in_d = elpre.config.hidden_size
        else:
            self.electra = ElectraForSequenceClassification(config=self.config)
            in_d = self.config.hidden_size

    def forward(self, data):
        electra = self.electra(data.x)
        d = torch.sum(electra.last_hidden_state, dim=1)
        return self.output(d)
