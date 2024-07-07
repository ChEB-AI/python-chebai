import logging
import random
from tempfile import TemporaryDirectory

import torch
from torch import nn
from torch.nn.functional import one_hot
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import (
    RobertaConfig,
    RobertaForMaskedLM,
    RobertaModel,
    RobertaTokenizer,
)

from chebai.models.base import ChebaiBaseNet

logging.getLogger("pysmiles").setLevel(logging.CRITICAL)
MAX_LEN = 1800


class ChembertaPre(ChebaiBaseNet):
    NAME = "ChembertaPre"

    def __init__(self, p=0.2, **kwargs):
        super().__init__(**kwargs)
        self._p = p
        self.config = RobertaConfig(**kwargs["config"])
        self.model = RobertaForMaskedLM(self.config)

    def _process_batch(self, batch, batch_idx):
        masked = (
            torch.rand([batch.x.shape[0]], device=self.device)
            * torch.tensor(batch.lens, device=self.device)
        ).long()
        labels = one_hot(
            torch.gather(batch.x, 1, masked.unsqueeze(-1)).squeeze(-1),
            self.config.vocab_size,
        )
        features = 1 + batch.x
        features = features * (1 - one_hot(masked, batch.x.shape[-1]))
        return features, labels

    def forward(self, data):
        x = self.model(data)
        return {"logits": torch.sum(x.logits, dim=1)}


class Chemberta(ChebaiBaseNet):
    NAME = "Chemberta"

    def __init__(self, **kwargs):
        # Remove this property in order to prevent it from being stored as a
        # hyper parameter
        pretrained_checkpoint = (
            kwargs.pop("pretrained_checkpoint")
            if "pretrained_checkpoint" in kwargs
            else None
        )
        super().__init__(**kwargs)
        self.config = RobertaConfig(
            **kwargs["config"], output_attentions=True, num_labels=self.out_dim
        )

        if pretrained_checkpoint:
            elpre = RobertaModel.load_from_checkpoint(pretrained_checkpoint)
            with TemporaryDirectory() as td:
                elpre.electra.save_pretrained(td)
                self.electra = RobertaModel.from_pretrained(td, config=self.config)
                in_d = elpre.config.hidden_size
        else:
            self.electra = RobertaModel(config=self.config)
            in_d = self.config.hidden_size

    def forward(self, data):
        electra = self.electra(data)
        return dict(logits=electra.logits, attentions=electra.attentions)
