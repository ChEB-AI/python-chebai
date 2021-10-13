from transformers import ElectraConfig, ElectraForPreTraining

from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import logging
from chem.models.base import JCIBaseNet

logging.getLogger('pysmiles').setLevel(logging.CRITICAL)

# TODO: Make params dynamic and load them from dataset
VOCAB_SIZE = 1400
MAX_LEN = 1800

class ElectraPre(JCIBaseNet):
    NAME = "Electra"
    def __init__(self, in_d, out_d, num_classes, **kwargs):
        super().__init__(num_classes, **kwargs)
        config = ElectraConfig(vocab_size=VOCAB_SIZE,
            max_position_embeddings=MAX_LEN,
            num_attention_heads=8,
            num_hidden_layers=6,
            type_vocab_size=1,)
        self.electra = ElectraForPreTraining(config)

    def forward(self, data):
        x = data.x
        x = self.electra(x)
        return x.logits
