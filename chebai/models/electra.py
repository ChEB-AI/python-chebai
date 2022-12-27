import random
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
    ElectraForMaskedLM,
    ElectraForMultipleChoice,
    ElectraForPreTraining,
    ElectraForSequenceClassification,
    ElectraModel,
    PretrainedConfig,
)
from chebai.preprocessing.reader import MASK_TOKEN_INDEX
import torch

from chebai.models.base import JCIBaseNet

logging.getLogger("pysmiles").setLevel(logging.CRITICAL)

class ElectraPre(JCIBaseNet):
    NAME = "ElectraPre"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        config = kwargs["config"]
        self.generator_config = ElectraConfig(**config["generator"])
        self.generator = ElectraForMaskedLM(self.generator_config)
        self.discriminator_config = ElectraConfig(**config["discriminator"])
        self.discriminator = ElectraForPreTraining(self.discriminator_config)
        self.replace_p = 0.1

    @property
    def as_pretrained(self):
        return self.discriminator

    def _get_data_and_labels(self, batch, batch_idx):

        return dict(features=batch.x, labels=None, mask=batch.mask)

    def forward(self, data):
        features = data["features"]
        self.batch_size = batch_size = features.shape[0]
        max_seq_len = features.shape[1]

        mask = data["mask"]
        with torch.no_grad():
            dis_tar = (torch.rand((batch_size,), device=self.device) * torch.sum(mask, dim=-1)).int()
            disc_tar_one_hot = torch.eq(torch.arange(max_seq_len, device=self.device)[None, :],
                                        dis_tar[:, None])
            gen_tar = features[disc_tar_one_hot]
            #x[disc_tar_one_hot] = MASK_TOKEN_INDEX
            gen_tar_one_hot = torch.eq(torch.arange(self.generator_config.vocab_size, device=self.device)[None, :],
                                       gen_tar[:, None])

        raw_gen_out = torch.mean(self.generator((features * ~disc_tar_one_hot) + MASK_TOKEN_INDEX*disc_tar_one_hot, attention_mask=mask).logits, dim=1)
        gen_best_guess = raw_gen_out.argmax(dim=-1)

        with torch.no_grad():
            correct_mask = (features[disc_tar_one_hot] == gen_best_guess)
            random_tokens = torch.randint(self.generator_config.vocab_size, (batch_size,), device=self.device)
            replacements = gen_best_guess * ~correct_mask + random_tokens*correct_mask

        disc_out = torch.softmax(self.discriminator(features * ~disc_tar_one_hot + replacements[:,None] * disc_tar_one_hot, attention_mask=mask).logits, dim=-1)
        return (torch.softmax(raw_gen_out, dim=-1), disc_out), (gen_tar_one_hot.float(), disc_tar_one_hot.float())

    def _get_prediction_and_labels(self, batch, labels, output):
        return output[0][1], output[1][1].int()

    def _get_data_for_loss(self, model_output, labels):
        return dict(input=model_output, target=None)


class ElectraPreLoss:
    def __init__(self):
        self.bce_log = torch.nn.BCEWithLogitsLoss()
        self.bce = torch.nn.BCELoss()

    def __call__(self, target, input):
        t, p = input
        gen_pred, disc_pred = t
        gen_tar, disc_tar = p
        gen_loss = self.bce(target=gen_tar, input=gen_pred)
        with_differences = torch.any(disc_tar, dim=-1)
        if torch.any(with_differences):
            disc_loss = self.bce(target=disc_tar[with_differences], input=disc_pred[with_differences])
        else:
            disc_loss = 0
        return gen_loss + disc_loss


class Electra(JCIBaseNet):
    NAME = "Electra"

    def _get_data_and_labels(self, batch, batch_idx):
        mask = pad_sequence([torch.ones(l, device=self.device) for l in batch.lens]).T
        return dict(
            features=batch.x, labels=batch.y, model_kwargs=dict(attention_mask=mask), target_mask=batch.target_mask
        )

    @property
    def as_pretrained(self):
        return self.electra.electra

    def __init__(self, **kwargs):
        # Remove this property in order to prevent it from being stored as a
        # hyper parameter
        pretrained_checkpoint = (
            kwargs.pop("pretrained_checkpoint")
            if "pretrained_checkpoint" in kwargs
            else None
        )
        super().__init__(**kwargs)
        if not "num_labels" in kwargs["config"] and self.out_dim is not None:
            kwargs["config"]["num_labels"] = self.out_dim
        self.config = ElectraConfig(**kwargs["config"], output_attentions=True)

        if pretrained_checkpoint:
            elpre = ElectraPre.load_from_checkpoint(pretrained_checkpoint)
            with TemporaryDirectory() as td:
                elpre.as_pretrained.save_pretrained(td)
                self.electra = ElectraModel.from_pretrained(
                    td, config=self.config
                )
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

    def _get_data_for_loss(self, model_output, labels):
        mask = model_output.get("target_mask")
        if mask is not None:
            d = model_output["logits"] * mask - 100 * ~mask
        else:
            d = model_output["logits"]
        return dict(input=d, target=labels.float())

    def _get_prediction_and_labels(self, data, labels, model_output):
        mask = model_output.get("target_mask")
        if mask is not None:
            d = model_output["logits"]*mask - 100 * ~mask
        else:
            d = model_output["logits"]
        return torch.sigmoid(d), labels.int()

    def forward(self, data, **kwargs):
        self.batch_size = data["features"].shape[0]
        inp = data["features"]
        electra = self.electra(inp, **kwargs)
        d = torch.mean(electra.last_hidden_state, dim=1)
        return dict(logits=self.output(d), attentions=electra.attentions, target_mask=data.get("target_mask"))


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
