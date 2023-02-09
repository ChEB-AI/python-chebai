import random
from tempfile import TemporaryDirectory
import logging
import random
from math import pi
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
from chebai.preprocessing.reader import MASK_TOKEN_INDEX, CLS_TOKEN
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
            dis_tar = (
                torch.rand((batch_size,), device=self.device) * torch.sum(mask, dim=-1)
            ).int()
            disc_tar_one_hot = torch.eq(
                torch.arange(max_seq_len, device=self.device)[None, :], dis_tar[:, None]
            )
            gen_tar = features[disc_tar_one_hot]
            gen_tar_one_hot = torch.eq(
                torch.arange(self.generator_config.vocab_size, device=self.device)[
                    None, :
                ],
                gen_tar[:, None],
            )

        raw_gen_out = torch.mean(
            self.generator(
                (features * ~disc_tar_one_hot) + MASK_TOKEN_INDEX * disc_tar_one_hot,
                attention_mask=mask,
            ).logits,
            dim=1,
        )

        with torch.no_grad():
            gen_best_guess = raw_gen_out.argmax(dim=-1)
            correct_mask = features[disc_tar_one_hot] == gen_best_guess
            random_tokens = torch.randint(
                self.generator_config.vocab_size, (batch_size,), device=self.device
            )
            replacements = gen_best_guess * ~correct_mask + random_tokens * correct_mask

        disc_out = self.discriminator(
            features * ~disc_tar_one_hot + replacements[:, None] * disc_tar_one_hot,
            attention_mask=mask,
        ).logits
        return (raw_gen_out, disc_out), (gen_tar_one_hot, disc_tar_one_hot)

    def _get_prediction_and_labels(self, batch, labels, output):
        return torch.softmax(output[0][1], dim=-1), output[1][1].int()

    def _get_data_for_loss(self, model_output, labels):
        return dict(input=model_output, target=None)


class ElectraPreLoss:
    def __init__(self):
        self.ce = torch.nn.CrossEntropyLoss()

    def __call__(self, target, input):
        t, p = input
        gen_pred, disc_pred = t
        gen_tar, disc_tar = p
        gen_loss = self.ce(target=torch.argmax(gen_tar.int(), dim=-1), input=gen_pred)
        disc_loss = self.ce(
            target=torch.argmax(disc_tar.int(), dim=-1), input=disc_pred
        )
        return gen_loss + disc_loss


class Electra(JCIBaseNet):
    NAME = "Electra"

    def _get_data_and_labels(self, batch, batch_idx):
        mask = pad_sequence(
            [torch.ones(l + 1, device=self.device) for l in batch.lens],
            batch_first=True,
        )
        cls_tokens = (
            torch.ones(batch.x.shape[0], dtype=torch.int, device=self.device).unsqueeze(
                -1
            )
            * CLS_TOKEN
        )
        return dict(
            features=torch.cat((cls_tokens, batch.x), dim=1),
            labels=batch.y,
            model_kwargs=dict(attention_mask=mask),
            target_mask=batch.target_mask,
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
        self.word_dropout = nn.Dropout(kwargs["config"].get("word_dropout", 0))
        model_prefix = kwargs.get("load_prefix", None)
        if pretrained_checkpoint:
            with open(pretrained_checkpoint, "rb") as fin:
                model_dict = torch.load(fin,map_location=self.device)
                if model_prefix:
                    state_dict = {str(k)[len(model_prefix):]:v for k,v in model_dict["state_dict"].items() if str(k).startswith(model_prefix)}
                else:
                    state_dict = model_dict["state_dict"]
                self.electra = ElectraModel.from_pretrained(None, state_dict=state_dict, config=self.config)
        else:
            self.electra = ElectraModel(config=self.config)

        in_d = self.config.hidden_size

        self.output = nn.Sequential(
            nn.Dropout(self.config.hidden_dropout_prob),
            nn.Linear(in_d, in_d),
            nn.GELU(),
            nn.Dropout(self.config.hidden_dropout_prob),
            nn.Linear(in_d, self.config.num_labels),
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
            d = model_output["logits"] * mask - 100 * ~mask
        else:
            d = model_output["logits"]
        return torch.sigmoid(d), labels.int()

    def forward(self, data, **kwargs):
        self.batch_size = data["features"].shape[0]
        inp = self.electra.embeddings.forward(data["features"])
        inp = self.word_dropout(inp)
        electra = self.electra(inputs_embeds=inp, **kwargs)
        d = electra.last_hidden_state[:, 0, :]
        return dict(
            logits=self.output(d),
            attentions=electra.attentions,
            target_mask=data.get("target_mask"),
        )


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

class ConeElectra(JCIBaseNet):
    NAME = "ConeElectra"

    def _get_data_and_labels(self, batch, batch_idx):
        mask = pad_sequence(
            [torch.ones(l + 1, device=self.device) for l in batch.lens],
            batch_first=True,
        )
        cls_tokens = (
            torch.ones(batch.x.shape[0], dtype=torch.int, device=self.device).unsqueeze(
                -1
            )
            * CLS_TOKEN
        )
        return dict(
            features=torch.cat((cls_tokens, batch.x), dim=1),
            labels=batch.y,
            model_kwargs=dict(attention_mask=mask),
            target_mask=batch.target_mask,
        )

    @property
    def as_pretrained(self):
        return self.electra.electra

    def __init__(self, cone_dimensions=20, **kwargs):
        # Remove this property in order to prevent it from being stored as a
        # hyper parameter
        pretrained_checkpoint = (
            kwargs.pop("pretrained_checkpoint")
            if "pretrained_checkpoint" in kwargs
            else None
        )

        self.cone_dimensions = cone_dimensions

        super().__init__(**kwargs)
        if not "num_labels" in kwargs["config"] and self.out_dim is not None:
            kwargs["config"]["num_labels"] = self.out_dim
        self.config = ElectraConfig(**kwargs["config"], output_attentions=True)
        self.word_dropout = nn.Dropout(kwargs["config"].get("word_dropout", 0))
        model_prefix = kwargs.get("load_prefix", None)
        if pretrained_checkpoint:
            with open(pretrained_checkpoint, "rb") as fin:
                model_dict = torch.load(fin,map_location=self.device)
                if model_prefix:
                    state_dict = {str(k)[len(model_prefix):]:v for k,v in model_dict["state_dict"].items() if str(k).startswith(model_prefix)}
                else:
                    state_dict = model_dict["state_dict"]
                self.electra = ElectraModel.from_pretrained(None, state_dict=state_dict, config=self.config)
        else:
            self.electra = ElectraModel(config=self.config)

        in_d = self.config.hidden_size

        self.line_embedding = nn.Sequential(
            nn.Dropout(self.config.hidden_dropout_prob),
            nn.Linear(in_d, in_d),
            nn.GELU(),
            nn.Dropout(self.config.hidden_dropout_prob),
            nn.Linear(in_d, self.cone_dimensions),
        )

        self.cone_axes = nn.Parameter(2*pi*torch.rand((1, self.config.num_labels, self.cone_dimensions)))
        self.cone_arcs = nn.Parameter(pi*(1-2*torch.rand((1, self.config.num_labels, self.cone_dimensions))))

    def _get_data_for_loss(self, model_output, labels):
        mask = model_output.get("target_mask")
        d = model_output["predicted_vectors"]
        return dict(input=dict(predicted_vectors=d,
                               cone_axes = self.cone_axes,
                               cone_arcs = self.cone_arcs),
                    target=labels.float())

    def _get_prediction_and_labels(self, data, labels, model_output):
        mask = model_output.get("target_mask")
        d = model_output["predicted_vectors"]

        d = 1- ConeLoss.cal_logit_cone(d, self.cone_axes, self.cone_arcs)

        return d, labels.int()

    def forward(self, data, **kwargs):
        self.batch_size = data["features"].shape[0]
        inp = self.electra.embeddings.forward(data["features"])
        inp = self.word_dropout(inp)
        electra = self.electra(inputs_embeds=inp, **kwargs)
        d = electra.last_hidden_state[:, 0, :]
        return dict(
            predicted_vectors=self.line_embedding(d),
            attentions=electra.attentions,
            target_mask=data.get("target_mask"),
        )

def softabs(x, eps=0.01):
    return (x**2+eps)**0.5-eps**0.5

def in_cone_parts(vectors, cone_axes, cone_arcs):
        theta_L = cone_axes + cone_arcs
        theta_R = cone_axes - cone_arcs
        return ((softabs(vectors - theta_L) + softabs(vectors - theta_R)) - cone_arcs)/(2*pi-cone_arcs)

class ConeLoss:

    def __init__(self, center_scaling=0.1):
        self.center_scaling = center_scaling

    def negate(self, ax, arc):
        offset = pi*torch.ones_like(ax)
        offset[ax >= 0] *= -1
        return ax + offset, pi - arc

    def __call__(self, target, input):
        predicted_vectors = input["predicted_vectors"]
        cone_axes = input["cone_axes"]
        cone_arcs = input["cone_arcs"]
        memberships =  1 - in_cone_parts(predicted_vectors, cone_axes, cone_arcs)
        loss = torch.nn.functional.binary_cross_entropy(memberships, target)
        return loss