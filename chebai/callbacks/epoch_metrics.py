from typing import Any

from lightning.pytorch.callbacks import Callback
import numpy as np
from lightning.pytorch.utilities.types import STEP_OUTPUT
import lightning as pl
from torchmetrics.classification import MultilabelF1Score


class _EpochLevelMetric(Callback):
    """Applies a metric to data from a whole training epoch, instead of batch-wise (the default in Lightning)"""

    def __init__(self, num_labels):
        self.train_labels, self.val_labels = None, None
        self.train_preds, self.val_preds = None, None
        self.num_labels = num_labels

    @property
    def metric_name(self):
        raise NotImplementedError

    def apply_metric(self, target, pred):
        raise NotImplementedError

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.train_labels = np.empty(shape=(0,), dtype=int)
        self.train_preds = np.empty(shape=(0,), dtype=int)

    def on_train_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT,
                           batch: Any, batch_idx: int) -> None:
        self.train_labels = np.concatenate((self.train_labels, outputs['labels'].int(),))
        self.train_preds = np.concatenate((self.train_preds, outputs['preds'],))

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        pl_module.log(f'train_{self.metric_name}', self.apply_metric(self.train_labels, self.train_preds))

    def on_validation_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.val_labels = np.empty(shape=(0,), dtype=int)
        self.val_preds = np.empty(shape=(0,), dtype=int)

    def on_validation_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT,
                                batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        self.val_labels = np.concatenate((self.val_labels, outputs['labels'].int(),))
        self.val_preds = np.concatenate((self.val_preds, outputs['preds'],))

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        pl_module.log(f'val_{self.metric_name}', self.apply_metric(self.val_labels, self.val_preds))


class EpochLevelMacroF1(_EpochLevelMetric):

    @property
    def metric_name(self):
        return 'ep_macro-f1'

    def apply_metric(self, target, pred):
        f1 = MultilabelF1Score(num_labels=self.num_labels, average='macro')
        return f1(target, pred)
