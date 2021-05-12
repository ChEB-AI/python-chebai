import os
from torch import nn
import torch
from sklearn.metrics import f1_score
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.metrics import F1
from pytorch_lightning.metrics import MeanSquaredError
from torch.nn.utils.rnn import pad_sequence
from data import JCIExtendedData, JCIData
import logging
import sys

logging.getLogger('pysmiles').setLevel(logging.CRITICAL)

class ChemLSTM(pl.LightningModule):

    def __init__(self, in_d, out_d, num_classes, weights):
        super().__init__(num_classes, weights)
        self.lstm = nn.LSTM(100, 300, batch_first=True)
        self.embedding = nn.Embedding(800, 100)
        self.output = nn.Sequential(nn.Linear(300, 1000), nn.ReLU(), nn.Dropout(0.2), nn.Linear(1000, num_classes))
        self.loss = nn.BCEWithLogitsLoss()
        self.f1 = F1(num_classes, threshold=0.5, average="micro")
        self.mse = MeanSquaredError()

    def forward(self, x):
        x = self.embedding(x)
        x = self.lstm(x)[1][0]
        x = self.output(x)
        return x.squeeze(0)


if __name__ == "__main__":
    data = JCIExtendedData(batch_size=int(sys.argv[1]))
    ChemLSTM.run(data, model_args=[100, 500, 500])
