from math import pi
from tempfile import TemporaryDirectory
import logging

from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    ElectraConfig,
    ElectraForMaskedLM,
    ElectraForPreTraining,
    ElectraModel,
)
import torch

from chebai.loss.pretraining import ElectraPreLoss  # noqa
from chebai.models.base import ChebaiBaseNet
from chebai.preprocessing.reader import CLS_TOKEN, MASK_TOKEN_INDEX
import pytorch_lightning as pl

logging.getLogger("pysmiles").setLevel(logging.CRITICAL)


class ElectraPre(ChebaiBaseNet):
    NAME = "ElectraPre"

    def __init__(self, config=None, **kwargs):
        super().__init__(config=config, **kwargs)
        self.generator_config = ElectraConfig(**config["generator"])
        self.generator = ElectraForMaskedLM(self.generator_config)
        self.discriminator_config = ElectraConfig(**config["discriminator"])
        self.discriminator = ElectraForPreTraining(self.discriminator_config)
        self.replace_p = 0.1

    @property
    def as_pretrained(self):
        return self.discriminator

    def _process_labels_in_batch(self, batch):
        return None

    def forward(self, data, **kwargs):
        features = data["features"]
        features = features.long()
        self.batch_size = batch_size = features.shape[0]
        max_seq_len = features.shape[1]

        mask = kwargs["mask"]
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


def filter_dict(d, filter_key):
    return {
        str(k)[len(filter_key) :]: v
        for k, v in d.items()
        if str(k).startswith(filter_key)
    }


class ElectraBasedModel(ChebaiBaseNet):
    def _process_batch(self, batch, batch_idx):
        model_kwargs = dict()
        loss_kwargs = batch.additional_fields["loss_kwargs"]
        if "lens" in batch.additional_fields["model_kwargs"]:
            model_kwargs["attention_mask"] = pad_sequence(
                [
                    torch.ones(l + 1, device=self.device)
                    for l in batch.additional_fields["model_kwargs"]["lens"]
                ],
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
            model_kwargs=model_kwargs,
            loss_kwargs=loss_kwargs,
            idents=batch.additional_fields["idents"],
        )

    def __init__(
        self, config=None, pretrained_checkpoint=None, load_prefix=None, **kwargs
    ):
        # Remove this property in order to prevent it from being stored as a
        # hyper parameter

        super().__init__(**kwargs)
        if config is None:
            config = dict()
        if not "num_labels" in config and self.out_dim is not None:
            config["num_labels"] = self.out_dim
        self.config = ElectraConfig(**config, output_attentions=True)
        self.word_dropout = nn.Dropout(config.get("word_dropout", 0))

        in_d = self.config.hidden_size

        if pretrained_checkpoint:
            with open(pretrained_checkpoint, "rb") as fin:
                model_dict = torch.load(fin, map_location=self.device)
                if load_prefix:
                    state_dict = filter_dict(model_dict["state_dict"], load_prefix)
                else:
                    state_dict = model_dict["state_dict"]
                self.electra = ElectraModel.from_pretrained(
                    None, state_dict=state_dict, config=self.config
                )
        else:
            self.electra = ElectraModel(config=self.config)

    def _process_for_loss(self, model_output, labels, loss_kwargs):
        kwargs_copy = dict(loss_kwargs)
        mask = kwargs_copy.pop("target_mask", None)
        if mask is not None:
            d = model_output["logits"] * mask - 100 * ~mask
        else:
            d = model_output["logits"]
        if labels is not None:
            labels = labels.float()
        return d, labels, kwargs_copy

    def _get_prediction_and_labels(self, data, labels, model_output):
        mask = model_output.get("target_mask")
        if mask is not None:
            d = model_output["logits"] * mask - 100 * ~mask
        else:
            d = model_output["logits"]
        loss_kwargs = data.get("loss_kwargs", dict())
        if "non_null_labels" in loss_kwargs:
            n = loss_kwargs["non_null_labels"]
            d = d[n]
        return torch.sigmoid(d), labels.int()

    def forward(self, data, **kwargs):
        self.batch_size = data["features"].shape[0]
        try:
            inp = self.electra.embeddings.forward(data["features"].int())
        except RuntimeError as e:
            print(f"RuntimeError at forward: {e}")
            print(f'data[features]: {data["features"]}')
            raise Exception
        inp = self.word_dropout(inp)
        electra = self.electra(inputs_embeds=inp, **kwargs)
        d = electra.last_hidden_state[:, 0, :]
        return dict(
            output=d,
            attentions=electra.attentions,
            target_mask=data.get("target_mask"),
        )


class Electra(ElectraBasedModel):
    NAME = "Electra"

    def __init__(
        self, config=None, pretrained_checkpoint=None, load_prefix=None, **kwargs
    ):
        # Remove this property in order to prevent it from being stored as a
        # hyper parameter

        super().__init__(**kwargs)

        in_d = self.config.hidden_size
        self.output = nn.Sequential(
            nn.Dropout(self.config.hidden_dropout_prob),
            nn.Linear(in_d, in_d),
            nn.GELU(),
            nn.Dropout(self.config.hidden_dropout_prob),
            nn.Linear(in_d, self.config.num_labels),
        )

    def forward(self, data, **kwargs):
        d = super().forward(data, **kwargs)
        return dict(
            logits=self.output(d["output"]),
            attentions=d["attentions"],
            target_mask=d["target_mask"],
        )


class ElectraLegacy(ChebaiBaseNet):
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


def gbmf(x, l, r, b=6):
    a = (r - l) + 1e-3
    c = l + (r - l) / 2
    return 1 / (1 + (torch.abs((x - c) / a) ** (2 * b)))


def normal(sigma, mu, x):
    v = (x - mu) / sigma
    return torch.exp(-0.5 * v * v)


class ChebiBoxWithMemberships(ElectraBasedModel):
    NAME = "ChebiBoxWithMemberships"

    def __init__(
        self, membership_method="normal", dimension_aggregation="lukaziewisz", **kwargs
    ):
        super().__init__(**kwargs)
        self.in_dim = self.config.hidden_size
        self.hidden_dim = self.config.embeddings_to_points_hidden_size
        self.out_dim = self.config.embeddings_dimensions
        self.boxes = nn.Parameter(
            torch.rand((self.config.num_labels, self.out_dim, 2)) * 3
        )
        self.membership_method = membership_method
        self.dimension_aggregation = dimension_aggregation

        self.embeddings_to_points = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.out_dim),
        )

    def _prod_agg(self, memberships, dim=-1):
        return torch.relu(
            torch.sum(memberships, dim=dim) - (memberships.shape[dim] - 1)
        )

    def _min_agg(self, memberships, dim=-1):
        return torch.relu(
            torch.sum(memberships, dim=dim) - (memberships.shape[dim] - 1)
        )

    def _lukaziewisz_agg(self, memberships, dim=-1):
        return torch.relu(
            torch.sum(memberships, dim=dim) - (memberships.shape[dim] - 1)
        )

    def _forward_gbmf_membership(self, points, left_corners, right_corners, **kwargs):
        return gbmf(points, left_corners, right_corners)

    def _forward_normal_membership(self, points, left_corners, right_corners, **kwargs):
        widths = 0.1 * (right_corners - left_corners)
        max_distance_per_dim = nn.functional.relu(
            left_corners - points + widths**0.5
        ) + nn.functional.relu(points - right_corners + widths**0.5)
        return normal(widths**0.5, 0, max_distance_per_dim)

    def forward(self, data, **kwargs):
        d = super().forward(data, **kwargs)
        points = self.embeddings_to_points(d["output"]).unsqueeze(1)

        b = self.boxes.unsqueeze(0)
        l = torch.min(b, dim=-1)[0]
        r = torch.max(b, dim=-1)[0]

        if self.membership_method == "normal":
            m = self._forward_normal_membership(points, l, r)
        elif self.membership_method == "gbmf":
            m = self._forward_gbmf_membership(points, l, r)
        else:
            raise Exception("Unknown membership function:", self.membership_method)

        if self.dimension_aggregation == "prod":
            m = self._prod_agg(m)
        elif self.dimension_aggregation == "lukaziewisz":
            m = self._lukaziewisz_agg(m)
        elif self.dimension_aggregation == "min":
            m = self._prod_min(m)
        else:
            raise Exception("Unknown aggregation function:", self.dimension_aggregation)

        return dict(
            boxes=b,
            embedded_points=points,
            logits=m,
            attentions=d["attentions"],
            target_mask=d["target_mask"],
        )


class BoxLoss(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.criteria = nn.BCELoss()

    def forward(self, outputs, targets, **kwargs):
        criterion = self.criteria
        bce_loss = criterion(outputs, targets)

        total_loss = bce_loss

        return total_loss


class ConeElectra(ChebaiBaseNet):
    NAME = "ConeElectra"

    def _process_batch(self, batch, batch_idx):
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
                model_dict = torch.load(fin, map_location=self.device)
                if model_prefix:
                    state_dict = {
                        str(k)[len(model_prefix) :]: v
                        for k, v in model_dict["state_dict"].items()
                        if str(k).startswith(model_prefix)
                    }
                else:
                    state_dict = model_dict["state_dict"]
                self.electra = ElectraModel.from_pretrained(
                    None, state_dict=state_dict, config=self.config
                )
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

        self.cone_axes = nn.Parameter(
            2 * pi * torch.rand((1, self.config.num_labels, self.cone_dimensions))
        )
        self.cone_arcs = nn.Parameter(
            pi * (1 - 2 * torch.rand((1, self.config.num_labels, self.cone_dimensions)))
        )

    def _get_data_for_loss(self, model_output, labels):
        mask = model_output.get("target_mask")
        d = model_output["predicted_vectors"]
        return dict(
            input=dict(
                predicted_vectors=d, cone_axes=self.cone_axes, cone_arcs=self.cone_arcs
            ),
            target=labels.float(),
        )

    def _get_prediction_and_labels(self, data, labels, model_output):
        mask = model_output.get("target_mask")
        d = model_output["predicted_vectors"].unsqueeze(1)

        d = in_cone_parts(d, self.cone_axes, self.cone_arcs)

        return torch.mean(d, dim=-1), labels.int()

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
    return (x**2 + eps) ** 0.5 - eps**0.5


def anglify(x):
    return torch.tanh(x) * pi


def turn(vector, angle):
    v = vector - angle
    return v - (v > pi) * 2 * pi + (v < -pi) * 2 * pi


def in_cone_parts(vectors, cone_axes, cone_arcs):
    """
    # trap between -pi and pi
    cone_ax_ang = anglify(cone_axes)
    v = anglify(vectors)

    # trap between 0 and pi
    cone_arc_ang = (torch.tanh(cone_arcs)+1)*pi/2
    theta_L = cone_ax_ang - cone_arc_ang/2
    #theta_L = theta_L - (theta_L > 2*pi) * 2 * pi + (theta_L < 0) *2*pi
    theta_R = cone_ax_ang + cone_arc_ang/2
    #theta_R = theta_R - (theta_R > 2 * pi) * 2 * pi + (theta_R < 0) * 2 * pi
    dis = (torch.abs(turn(v, theta_L)) + torch.abs(turn(v, theta_R)) - cone_arc_ang)/(2*pi-cone_arc_ang)
    return dis
    """
    a = cone_axes - cone_arcs**2
    b = cone_axes + cone_arcs**2
    bigger_than_a = torch.sigmoid(vectors - a)
    smaller_than_b = torch.sigmoid(b - vectors)
    return bigger_than_a * smaller_than_b


class ConeLoss:
    def __init__(self, center_scaling=0.1):
        self.center_scaling = center_scaling

    def negate(self, ax, arc):
        offset = pi * torch.ones_like(ax)
        offset[ax >= 0] *= -1
        return ax + offset, pi - arc

    def __call__(self, target, input):
        predicted_vectors = input["predicted_vectors"].unsqueeze(1)
        cone_axes = input["cone_axes"]
        cone_arcs = input["cone_arcs"]
        memberships = (1 - 1e-6) * (
            in_cone_parts(predicted_vectors, cone_axes, cone_arcs)
        )
        loss = torch.nn.functional.binary_cross_entropy(
            memberships, target.unsqueeze(-1).expand(-1, -1, 20)
        )
        return loss
