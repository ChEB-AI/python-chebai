from transformers import ElectraConfig, ElectraForPreTraining

from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import logging
from chem.models.base import JCIBaseNet

logging.getLogger("pysmiles").setLevel(logging.CRITICAL)


class ElectraPre(JCIBaseNet):
    NAME = "Electra"

    def __init__(self, config=None, **kwargs):
        super().__init__(**kwargs)
        config = ElectraConfig(**config)
        self.electra = ElectraForPreTraining(config)

    def forward(self, data):
        x = data.x
        x = self.electra(x)
        return x.logits
