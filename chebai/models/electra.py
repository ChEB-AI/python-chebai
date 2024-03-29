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

logging.getLogger("pysmiles").setLevel(logging.CRITICAL)


class ElectraPre(ChebaiBaseNet):
    """
    ElectraPre class represents a pre-trained Electra model for pre-training inherited from ChebaiBaseNet.

    Args:
        config (dict): Configuration parameters for the Electra model.
        **kwargs: Additional keyword arguments.

    Attributes:
        NAME (str): Name of the ElectraPre model.
        generator_config (ElectraConfig): Configuration for the generator model.
        generator (ElectraForMaskedLM): Generator model for masked language modeling.
        discriminator_config (ElectraConfig): Configuration for the discriminator model.
        discriminator (ElectraForPreTraining): Discriminator model for pre-training.
        replace_p (float): Probability of replacing tokens during training.

    """

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
        """
        Returns the discriminator model as a pre-trained model.

        Returns:
            ElectraForPreTraining: The discriminator model.

        """
        return self.discriminator

    def _process_labels_in_batch(self, batch):
        return None

    def forward(self, data, **kwargs):
        """
        Forward pass of the ElectraPre model.

        Args:
            data (dict): Input data.
            **kwargs: Additional keyword arguments.

        Returns:
            tuple: A tuple containing the raw generator output and discriminator output.
            The generator output is a tensor of shape (batch_size, max_seq_len, vocab_size).
            The discriminator output is a tensor of shape (batch_size, max_seq_len).

        """
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


class Electra(ChebaiBaseNet):
    """
    Electra model implementation inherited from ChebaiBaseNet.

    Args:
        config (dict, optional): Configuration parameters for the Electra model. Defaults to None.
        pretrained_checkpoint (str, optional): Path to the pretrained checkpoint file. Defaults to None.
        load_prefix (str, optional): Prefix to filter the state_dict keys from the pretrained checkpoint. Defaults to None.
        **kwargs: Additional keyword arguments.

    Attributes:
        NAME (str): Name of the Electra model.

    """

    NAME = "Electra"

    def _process_batch(self, batch, batch_idx):
        """
        Process a batch of data.

        Args:
            batch (XYData): The input batch of data.
            batch_idx (int): The index of the batch.

        Returns:
            dict: A dictionary containing the processed batch.

        """
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
        """
        Get the pretrained Electra model.

        Returns:
            ElectraModel: The pretrained Electra model.

        """
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

        # Load pretrained checkpoint if provided
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
        """
        Process the model output for calculating the loss.

        Args:
            model_output (dict): The output of the model.
            labels (Tensor): The target labels.
            loss_kwargs (dict): Additional loss arguments.

        Returns:
            tuple: A tuple containing the processed model output, labels, and loss arguments.

        """
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
        """
        Get the predictions and labels from the model output.

        Args:
            data (dict): The input data.
            labels (Tensor): The target labels.
            model_output (dict): The output of the model.

        Returns:
            tuple: A tuple containing the predictions and labels.

        """
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
        """
        Forward pass of the Electra model.

        Args:
            data (dict): The input data.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: A dictionary containing the model output.

        """
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
            logits=self.output(d),
            attentions=electra.attentions,
            target_mask=data.get("target_mask"),
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
