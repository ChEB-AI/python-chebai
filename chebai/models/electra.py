import os.path
import pickle
import random
from tempfile import TemporaryDirectory
import logging
from typing import Dict
from math import pi

import torchmetrics
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
from chebai.preprocessing.datasets.chebi import extract_class_hierarchy
import torch
import csv
import pytorch_lightning as pl
from chebai.models.base import ChebaiBaseNet


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

    def _process_batch(self, batch, batch_idx):
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


class ElectraPreLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, input, target):
        t, p = input
        gen_pred, disc_pred = t
        gen_tar, disc_tar = p
        gen_loss = self.ce(target=torch.argmax(gen_tar.int(), dim=-1), input=gen_pred)
        disc_loss = self.ce(
            target=torch.argmax(disc_tar.int(), dim=-1), input=disc_pred
        )
        return gen_loss + disc_loss


def filter_dict(d, filter_key):
    return {
        str(k)[len(filter_key) :]: v
        for k, v in d.items()
        if str(k).startswith(filter_key)
    }


class Electra(ChebaiBaseNet):
    NAME = "Electra"

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

    @property
    def as_pretrained(self):
        return self.electra.electra

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
        self.output = nn.Sequential(
            nn.Dropout(self.config.hidden_dropout_prob),
            nn.Linear(in_d, in_d),
            nn.GELU(),
            nn.Dropout(self.config.hidden_dropout_prob),
            nn.Linear(in_d, self.config.num_labels),
        )
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
        mask = model_output.get("target_mask")
        if mask is not None:
            d = model_output["logits"] * mask - 100 * ~mask
        else:
            d = model_output["logits"]
        if labels is not None:
            labels = labels.float()
        return d, labels, loss_kwargs

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
        inp = self.electra.embeddings.forward(data["features"])
        inp = self.word_dropout(inp)
        electra = self.electra(inputs_embeds=inp, **kwargs)
        d = electra.last_hidden_state[:, 0, :]
        return dict(
            logits=self.output(d),
            attentions=electra.attentions,
            target_mask=data.get("target_mask"),
        )


IMPLICATION_CACHE_FILE = "chebi.cache"


def _load_label_names(path_to_label_names):
    with open(path_to_label_names) as fin:
        label_names = [int(line.strip()) for line in fin]
    return label_names


def _load_implications(path_to_chebi, implication_cache=IMPLICATION_CACHE_FILE):
    if os.path.isfile(implication_cache):
        with open(implication_cache, "rb") as fin:
            hierarchy = pickle.load(fin)
    else:
        hierarchy = extract_class_hierarchy(path_to_chebi)
        with open(implication_cache, "wb") as fout:
            pickle.dump(hierarchy, fout)
    return hierarchy


def _build_implication_filter(label_names, hierarchy):
    return torch.tensor(
        [
            (i1, i2)
            for i1, l1 in enumerate(label_names)
            for i2, l2 in enumerate(label_names)
            if l2 in hierarchy.pred[l1]
        ]
    )


def _build_disjointness_filter(path_to_disjointedness, label_names, hierarchy):
    disjoints = set()
    label_dict = dict(map(reversed, enumerate(label_names)))

    with open(path_to_disjointedness, "rt") as fin:
        reader = csv.reader(fin)
        for l1_raw, r1_raw in reader:
            l1 = int(l1_raw)
            r1 = int(r1_raw)
            disjoints.update(
                {
                    (label_dict[l2], label_dict[r2])
                    for r2 in hierarchy.succ[r1]
                    if r2 in label_names
                    for l2 in hierarchy.succ[l1]
                    if l2 in label_names and l2 < r2
                }
            )

    dis_filter = torch.tensor(list(disjoints))
    return dis_filter[:, 0], dis_filter[:, 1]


class ElectraChEBILoss(nn.Module):
    def __init__(
        self, path_to_chebi, path_to_label_names, base_loss: torch.nn.Module = None
    ):
        super().__init__()
        self.base_loss = base_loss
        label_names = _load_label_names(path_to_label_names)
        hierarchy = _load_implications(path_to_chebi)
        implication_filter = _build_implication_filter(label_names, hierarchy)
        self.implication_filter_l = implication_filter[:, 0]
        self.implication_filter_r = implication_filter[:, 1]

    def forward(self, input, target, **kwargs):
        if "non_null_labels" in kwargs:
            n = kwargs["non_null_labels"]
            inp = input[n]
        else:
            inp = input
        if target is not None:
            base_loss = self.base_loss(inp, target.float())
        else:
            base_loss = 0
        pred = torch.sigmoid(input)
        l = pred[:, self.implication_filter_l]
        r = pred[:, self.implication_filter_r]
        # implication_loss = torch.sqrt(torch.mean(torch.sum(l*(1-r), dim=-1), dim=0))
        implication_loss = torch.mean(torch.mean(torch.relu(l - r), dim=-1), dim=0)
        return base_loss + implication_loss


class ElectraChEBIDisjointLoss(ElectraChEBILoss):
    def __init__(
        self,
        path_to_chebi,
        path_to_label_names,
        path_to_disjointedness,
        base_loss: torch.nn.Module = None,
    ):
        super().__init__(path_to_chebi, path_to_label_names, base_loss)
        label_names = _load_label_names(path_to_label_names)
        hierarchy = _load_implications(path_to_chebi)
        self.disjoint_filter_l, self.disjoint_filter_r = _build_disjointness_filter(
            path_to_disjointedness, label_names, hierarchy
        )

    def forward(self, input, target, **kwargs):
        loss = super().forward(input, target, **kwargs)
        pred = torch.sigmoid(input)
        l = pred[:, self.disjoint_filter_l]
        r = pred[:, self.disjoint_filter_r]
        disjointness_loss = torch.mean(
            torch.mean(torch.relu(l - (1 - r)), dim=-1), dim=0
        )
        return loss + disjointness_loss


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


class ChebiBox(Electra):
    NAME = "ChebiBox"

    
    def __init__(self, dimensions=3, hidden_size=2000, **kwargs):
        super().__init__(**kwargs)
        self.dimensions = dimensions

        #self.boxes = nn.Parameter(
        #   3 - torch.rand((self.config.num_labels, self.dimensions, 2)) * 6
        #) 
        
        self.boxes = nn.Parameter( torch.rand((self.config.num_labels, self.dimensions, 2)) )

        self.embeddings_to_points = nn.Sequential(
            nn.Linear(self.electra.config.hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.dimensions)
        )

        #self.criterion = BoxLoss()

    def forward(self, data, **kwargs):
        self.batch_size = data["features"].shape[0]
        inp = self.electra.embeddings.forward(data["features"])
        inp = self.word_dropout(inp)
        electra = self.electra(inputs_embeds=inp, **kwargs)
        d = electra.last_hidden_state[:, 0, :]

        points = self.embeddings_to_points(d)

        b = self.boxes.expand(self.batch_size, -1, -1, -1)
        l = torch.min(b, dim=-1)[0]
        r = torch.max(b, dim=-1)[0]
        p = points.expand(self.config.num_labels, -1, -1).transpose(1, 0)
        max_distance_per_dim = torch.max(torch.stack((nn.functional.relu(l - p), nn.functional.relu(p - r))), dim=0)[0]
        
        # min might be replaced
        #m = torch.min(membership_per_dim, dim=-1)[0]
        #m = torch.mean(membership_per_dim, dim=-1)
        
        m = torch.mean(max_distance_per_dim, dim=-1)
        s = 2 - ( 2 * (torch.sigmoid(m)) )
        l = torch.logit( (s * 0.99) + 0.001 )
        
        return dict(
            boxes=b,
            embedded_points=points,
            logits=l,
            attentions=electra.attentions,
            target_mask=data.get("target_mask"),
        )

class BoxLoss(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, outputs, targets, model, **kwargs):
        """
        boxes = model.boxes
        dim = model.dimensions

        corner_1 = boxes[:, :, 0]
        corner_2 = boxes[:, :, 1]

        box_sizes_per_dim = torch.abs(corner_1 - corner_2)
        box_sizes = box_sizes_per_dim.prod(1)

        min_box_size_value = 2
        max_box_size_value = 100

        mask_min_box_size = (box_sizes < min_box_size_value)
        small_boxes = box_sizes[mask_min_box_size]
        diff_for_small_boxes = min_box_size_value - small_boxes

        min_box_size_penalty = 0
        if diff_for_small_boxes.nelement() != 0:
            min_box_size_penalty = torch.mean(diff_for_small_boxes) ** (1 / dim)

        mask_max_box_size = (box_sizes > max_box_size_value)
        large_boxes = box_sizes[mask_max_box_size]
        diff_for_large_boxes = large_boxes - max_box_size_value
        max_box_size_penalty = 0
        if diff_for_large_boxes.nelement() != 0:
            max_box_size_penalty = torch.mean(diff_for_large_boxes) ** (1 / dim)

        criterion = nn.BCEWithLogitsLoss()
        bce_loss = criterion(outputs, targets)

        #total_loss = bce_loss + (0.1 * min_box_size_penalty) + (0.1 * max_box_size_penalty)
        total_loss = bce_loss + (0.1 * max_box_size_penalty)
        """

        boxes = model.boxes
        dim = model.dimensions

        corner_1 = boxes[:, :, 0]
        corner_2 = boxes[:, :, 1]

        box_sizes_per_dim = torch.abs(corner_1 - corner_2)
        box_sizes = box_sizes_per_dim.prod(1)

        min_box_size_value = 1
        max_box_size_value = 0 

        diff_for_small_boxes = torch.relu(min_box_size_value - box_sizes)
        min_box_size_penalty = torch.mean(diff_for_small_boxes) ** (1 / dim)

        diff_for_large_boxes = torch.relu(box_sizes - max_box_size_value)
        max_box_size_penalty = torch.mean(diff_for_large_boxes) ** (1 / dim) * 0.001

        criterion = nn.BCEWithLogitsLoss()
        bce_loss = criterion(outputs, targets)

        total_loss = bce_loss + max_box_size_penalty 
        
        model.log(
            "min_box_size_penalty",
            (0.01 * min_box_size_penalty),
            batch_size=10,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            )

        model.log(
            "max_box_size_penalty",
            (0.01 * max_box_size_penalty),
            batch_size=10,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            )
        return total_loss

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
