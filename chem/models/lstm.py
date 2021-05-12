from torch import nn
from chem.data import JCIExtendedData
import logging
import sys
from base import JCIBaseNet

logging.getLogger('pysmiles').setLevel(logging.CRITICAL)

class ChemLSTM(JCIBaseNet):

    def __init__(self, in_d, out_d, num_classes, weights):
        super().__init__(num_classes, weights)
        self.lstm = nn.LSTM(100, 300, batch_first=True)
        self.embedding = nn.Embedding(800, 100)
        self.output = nn.Sequential(nn.Linear(300, 1000), nn.ReLU(), nn.Dropout(0.2), nn.Linear(1000, num_classes))

    def forward(self, x):
        x = self.embedding(x)
        x = self.lstm(x)[1][0]
        x = self.output(x)
        return x.squeeze(0)


if __name__ == "__main__":
    data = JCIExtendedData(batch_size=int(sys.argv[1]))
    ChemLSTM.run(data, "lstm", model_args=[100, 500, 500])
