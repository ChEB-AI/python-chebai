import logging
import sys

from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from chebai.models.base import ChebaiBaseNet

logging.getLogger("pysmiles").setLevel(logging.CRITICAL)


class ChemLSTM(ChebaiBaseNet):
    NAME = "LSTM"

    def __init__(self, in_d, out_d, num_classes, **kwargs):
        super().__init__(num_classes, **kwargs)
        self.lstm = nn.LSTM(in_d, out_d, batch_first=True)
        self.embedding = nn.Embedding(800, 100)
        self.output = nn.Sequential(
            nn.Linear(out_d, in_d),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_d, num_classes),
        )

    def forward(self, data):
        x = data.x
        x_lens = data.lens
        x = self.embedding(x)
        x = pack_padded_sequence(x, x_lens, batch_first=True, enforce_sorted=False)
        x = self.lstm(x)[1][0]
        # = pad_packed_sequence(x, batch_first=True)[0]
        x = self.output(x)
        return x.squeeze(0)
