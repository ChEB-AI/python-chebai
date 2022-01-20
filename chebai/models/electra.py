from tempfile import TemporaryDirectory
import logging
import random

from torch import nn
from torch.nn.utils.rnn import (
    pack_padded_sequence,
    pad_packed_sequence,
    pad_sequence,
)
from transformers import (
    ElectraConfig,
    ElectraModel,
    ElectraForPreTraining,
    ElectraForSequenceClassification,
    PretrainedConfig,
)
import torch

from chebai.models.base import JCIBaseNet

logging.getLogger("pysmiles").setLevel(logging.CRITICAL)


class ElectraPre(JCIBaseNet):
    NAME = "ElectraPre"

    def __init__(self, p=0.2, **kwargs):
        super().__init__(**kwargs)
        self._p = p
        self.config = ElectraConfig(**kwargs["config"])
        self.electra = ElectraForPreTraining(self.config)

    def _get_data_and_labels(self, batch, batch_idx):
        vocab_w0 = torch.cat(torch.unbind(batch.x))
        vocab = vocab_w0[torch.nonzero(vocab_w0, as_tuple=False)].squeeze(-1)
        labels_rnd = torch.rand(batch.x.shape, device=self.device)
        mask = pad_sequence([torch.ones(l, device=self.device) for l in batch.lens]).T
        subs = vocab[torch.randint(0, len(vocab), batch.x.shape, device=self.device)]
        equals = torch.eq(batch.x, subs).int()

        # exclude those indices where the replacement yields the same token
        labels = torch.logical_and(
            (labels_rnd < self._p) * mask, torch.logical_not(equals)
        ).int()
        features = (batch.x * (1 - labels)) + (subs * labels)

        return dict(
            features=features, labels=labels, model_kwargs=dict(attention_mask=mask)
        )

    def forward(self, data, **kwargs):
        x = self.electra(data, **kwargs)
        return {"logits": x.logits}


class Electra(JCIBaseNet):
    NAME = "Electra"

    def _get_data_and_labels(self, batch, batch_idx):
        mask = pad_sequence([torch.ones(l, device=self.device) for l in batch.lens]).T
        return dict(
            features=batch.x, labels=batch.y, model_kwargs=dict(attention_mask=mask)
        )

    def __init__(self, **kwargs):
        # Remove this property in order to prevent it from being stored as a
        # hyper parameter
        pretrained_checkpoint = (
            kwargs.pop("pretrained_checkpoint")
            if "pretrained_checkpoint" in kwargs
            else None
        )
        super().__init__(**kwargs)
        if not "num_labels" in kwargs["config"]:
            kwargs["config"]["num_labels"] = self.out_dim
        self.config = ElectraConfig(**kwargs["config"], output_attentions=True)

        if pretrained_checkpoint:
            elpre = ElectraPre.load_from_checkpoint(pretrained_checkpoint)
            with TemporaryDirectory() as td:
                elpre.electra.save_pretrained(td)
                self.electra = ElectraForSequenceClassification.from_pretrained(
                    td, config=self.config
                )
                in_d = elpre.config.hidden_size
        else:
            self.electra = ElectraForSequenceClassification(config=self.config)
            in_d = self.config.hidden_size

    def forward(self, data, **kwargs):
        electra = self.electra(data, **kwargs)
        return dict(logits=electra.logits, attentions=electra.attentions)


class ElectraLegacy(JCIBaseNet):
    NAME = "ElectraLeg"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.config = ElectraConfig(**kwargs["config"], output_attentions=True)

        if "pretrained_checkpoint" in kwargs:
            elpre = ElectraPre.load_from_checkpoint(kwargs["pretrained_checkpoint"])
            with TemporaryDirectory() as td:
                elpre.electra.save_pretrained(td)
                self.electra = ElectraModel.from_pretrained(td, config=self.config)
                in_d = elpre.config.hidden_size
        else:
            self.electra = ElectraModel(config=self.config)
            in_d = self.config.hidden_size

        self.output = nn.Sequential(
            nn.Linear(in_d, in_d),
            nn.ReLU(),
            nn.Linear(in_d, in_d),
            nn.ReLU(),
            nn.Linear(in_d, in_d),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_d, 500),
        )

    def forward(self, data):
        electra = self.electra(data)
        d = torch.sum(electra.last_hidden_state, dim=1)
        return dict(logits=self.output(d), attentions=electra.attentions)