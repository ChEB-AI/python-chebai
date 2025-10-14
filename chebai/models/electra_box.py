import torch
import torch.nn as nn
from transformers import ElectraConfig, ElectraModel

from chebai.models.base import ChebaiBaseNet
from chebai.models.electra import ElectraProcessingMixIn, filter_dict


class ElectraBox(ElectraProcessingMixIn, ChebaiBaseNet):
    NAME = "ElectraBox"

    def __init__(
        self, config=None, pretrained_checkpoint=None, load_prefix=None, **kwargs
    ):
        super().__init__(**kwargs)
        if config is None:
            config = dict()
        if "num_labels" not in config and self.out_dim is not None:
            config["num_labels"] = self.out_dim
        self.config = ElectraConfig(**config, output_attentions=True)
        self.word_dropout = nn.Dropout(config.get("word_dropout", 0))

        self.in_dim = self.config.hidden_size
        self.hidden_dim = self.config.embeddings_to_points_hidden_size
        self.out_dim = self.config.embeddings_dimensions
        self.boxes = nn.Parameter(torch.rand((self.config.num_labels, self.out_dim, 2)))
        self.embeddings_to_points = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.out_dim),
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

    def forward(self, data, **kwargs):
        self.batch_size = data["features"].shape[0]
        inp = self.electra.embeddings.forward(data["features"])
        inp = self.word_dropout(inp)
        electra = self.electra(inputs_embeds=inp)
        d = electra.last_hidden_state[:, 0, :]

        points = self.embeddings_to_points(d)

        b = self.boxes.expand(self.batch_size, -1, -1, -1)
        raw_l = torch.min(b, dim=-1)[0]
        raw_r = torch.max(b, dim=-1)[0]

        left = raw_l + ((raw_r - raw_l) * 0.2)
        right = raw_r - ((raw_r - raw_l) * 0.2)

        p = points.expand(self.config.num_labels, -1, -1).transpose(1, 0)
        max_distance_per_dim = torch.max(
            torch.stack((nn.functional.relu(left - p), nn.functional.relu(p - right))),
            dim=0,
        )[0]

        m = torch.sum(max_distance_per_dim, dim=-1)
        s = 2 - (2 * (torch.sigmoid(m)))
        logits = torch.logit((s * 0.99) + 0.001)

        return dict(
            boxes=b,
            embedded_points=points,
            logits=logits,
            attentions=electra.attentions,
            target_mask=data.get("target_mask"),
        )


if __name__ == "__main__":
    model = ElectraBox(
        config={
            "vocab_size": 4400,
            "max_position_embeddings": 1800,
            "num_attention_heads": 8,
            "num_hidden_layers": 6,
            "type_vocab_size": 1,
            "hidden_size": 256,
            "embeddings_to_points_hidden_size": 1200,
            "embeddings_dimensions": 16,
        },
        out_dim=120,
        input_dim=1800,
    )
    import torch

    print(
        model._process_for_loss(
            torch.randint(0, 4400, (2, 1800)), torch.randint(0, 2, (2, 120))
        )
    )
