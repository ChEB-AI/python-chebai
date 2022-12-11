from tempfile import TemporaryDirectory
import logging

from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import (
    ElectraConfig,
    ElectraForMaskedLM,
    ElectraForMultipleChoice,
    ElectraForPreTraining,
    ElectraModel,
    PretrainedConfig,
)
import torch

from chebai.models.base import JCIBaseNet

logging.getLogger("pysmiles").setLevel(logging.CRITICAL)


class ElectraPre(JCIBaseNet):
    NAME = "ElectraPre"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        config = kwargs["config"]
        self.generator_config = ElectraConfig(**config["generator"])
        self.generator = ElectraModel(self.generator_config)
        self.generator_head = torch.nn.Linear(510, 128)
        self.discriminator_config = ElectraConfig(**config["discriminator"])
        self.discriminator = ElectraForPreTraining(self.discriminator_config)
        self.replace_p = 0.1

    def forward(self, data):
        x = data.x
        self.batch_size = x.shape[0]
        embs = self.generator.embeddings(x)
        gen_out = self.generator_head(self.generator(inputs_embeds=embs).last_hidden_state)
        with torch.no_grad():
            replace = torch.rand(x.shape) < self.replace_p
            disc_input = replace.unsqueeze(-1)*gen_out + (~replace.unsqueeze(-1))*embs
            replaced_by_different = torch.any(torch.ne(disc_input, embs), dim=-1)
        disc_out = self.discriminator(inputs_embeds=disc_input)
        return disc_out.logits, replaced_by_different.float()

    def _get_prediction_and_labels(self, batch, output):
        return output[0], output[1]

class Electra(JCIBaseNet):
    NAME = "Electra"

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
        electra = self.electra(data.x)
        d = torch.sum(electra.last_hidden_state, dim=1)
        return self.output(d)
