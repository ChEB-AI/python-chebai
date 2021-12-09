from tempfile import TemporaryDirectory
import logging
import random

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
        super().__init__(**kwargs, p=0.2)
        self._p = 0.2
        self.config = ElectraConfig(**kwargs["config"])
        self.electra = ElectraForPreTraining(self.config)

    def _get_data_and_labels(self, batch, batch_idx):
        vocab_w0 = torch.cat(torch.unbind(batch.x))
        vocab = vocab_w0[torch.nonzero(vocab_w0, as_tuple=False)].squeeze(-1)
        labels_rnd = torch.rand(batch.x.shape, device=self.device)
        subs = vocab[torch.randint(0, len(vocab), batch.x.shape, device=self.device)]
        equals = torch.eq(batch.x, subs).int()

        # exclude those indices where the replacement yields the same token
        labels = (labels_rnd < self._p).int() * (1 - equals)
        features = (batch.x * (1 - labels)) + (subs * labels)

        return features, labels

    def forward(self, data):
        x = data
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
