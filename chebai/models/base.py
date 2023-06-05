import itertools
import logging
import os
import sys

from pytorch_lightning import loggers as pl_loggers
from lightning.pytorch.core.module import LightningModule
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.tuner.tuning import Tuner
from sklearn.metrics import f1_score
from torch import nn
from torchmetrics import MeanSquaredError
from torchmetrics import F1Score, MeanSquaredError
from torchmetrics import functional as tmf
import pytorch_lightning as pl
import torch
import torchmetrics
import tqdm

from chebai.preprocessing.datasets import XYBaseDataModule

logging.getLogger("pysmiles").setLevel(logging.CRITICAL)

_MODEL_REGISTRY = dict()

class ChebaiBaseNet(LightningModule):
    NAME = None
    LOSS = torch.nn.BCEWithLogitsLoss

    def __init__(self, criterion: torch.nn.Module, out_dim=None, **kwargs):
        super().__init__()
        self.criterion = criterion
        self.save_hyperparameters()
        self.out_dim = out_dim
        self.optimizer_kwargs = kwargs.get("optimizer_kwargs", dict())

    def __init_subclass__(cls, **kwargs):
        if cls.NAME in _MODEL_REGISTRY:
            raise ValueError(f"Model {cls.NAME} does already exist")
        else:
            _MODEL_REGISTRY[cls.NAME] = cls

    def _get_prediction_and_labels(self, data, labels, output):
        return output, labels

    def _get_data_and_labels(self, batch, batch_idx):
        return dict(features=batch.x, labels=batch.y.float())

    def training_step(self, batch, batch_idx):
        return self._execute(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self._execute(batch, batch_idx)

    def _execute(self, batch, batch_idx):
        data = self._get_data_and_labels(batch, batch_idx)
        labels = data["labels"]
        model_output = self(data, **data.get("model_kwargs", dict()))
        loss = self.criterion(model_output, labels)
        return dict(data=data, labels=labels, output=model_output, loss=loss)

    def forward(self, x):
        raise NotImplementedError

    def configure_optimizers(self, **kwargs):
        return torch.optim.Adamax(self.parameters(), **self.optimizer_kwargs)
