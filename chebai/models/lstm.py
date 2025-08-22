import logging

from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from chebai.models.base import ChebaiBaseNet

logging.getLogger("pysmiles").setLevel(logging.CRITICAL)


class ChemLSTM(ChebaiBaseNet):
    def __init__(self, out_d, in_d, num_classes, criterion : nn.Module=None, **kwargs):
        super().__init__(
            out_dim=out_d,
            input_dim=in_d,
            criterion=criterion,
            num_classes=num_classes,
            **kwargs
        )
        self.lstm = nn.LSTM(in_d, out_d, batch_first=True, dropout=0.2, bidirectional=True)
        self.embedding = nn.Embedding(1400, in_d)
        self.output = nn.Sequential(
            nn.Linear(out_d * 2, in_d),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_d, num_classes),
        )

    def forward(self, data, *args, **kwargs):
        x = data["features"]
        x_lens = data["model_kwargs"]["lens"]
        x = self.embedding(x)
        x = pack_padded_sequence(x, x_lens, batch_first=True, enforce_sorted=False)
        x = self.lstm(x)[1][0]
        # = pad_packed_sequence(x, batch_first=True)[0]
        x = self.output(x)
        return x.squeeze(0)
