from typing import Any

from lightning.pytorch.callbacks import Callback
import numpy as np
from lightning.pytorch.utilities.types import STEP_OUTPUT
import lightning as pl
from torchmetrics.classification import MultilabelF1Score
import torch


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

    def on_train_epoch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        device = pl_module.device
        self.train_labels = torch.empty(size=(0,), dtype=torch.int, device=device)
        self.train_preds = torch.empty(size=(0,), dtype=torch.int, device=device)

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        self.train_labels = torch.concatenate(
            (
                self.train_labels,
                outputs["labels"],
            )
        )
        self.train_preds = torch.concatenate(
            (
                self.train_preds,
                outputs["preds"],
            )
        )

    def on_train_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        pl_module.log(
            f"train_{self.metric_name}",
            self.apply_metric(self.train_labels, self.train_preds),
        )

    def on_validation_epoch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        self.val_labels = torch.empty(
            size=(0,), dtype=torch.int, device=pl_module.device
        )
        self.val_preds = torch.empty(
            size=(0,), dtype=torch.int, device=pl_module.device
        )

    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self.val_labels = torch.concatenate(
            (
                self.val_labels,
                outputs["labels"],
            )
        )
        self.val_preds = torch.concatenate(
            (
                self.val_preds,
                outputs["preds"],
            )
        )

    def on_validation_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        pl_module.log(
            f"val_{self.metric_name}",
            self.apply_metric(self.val_labels, self.val_preds),
        )


class EpochLevelMacroF1(_EpochLevelMetric):
    @property
    def metric_name(self):
        return "ep_macro-f1"

    def apply_metric(self, target, pred):
        f1 = MultilabelF1Score(num_labels=self.num_labels, average="macro")
        if target.get_device() != -1:  # -1 == CPU
            f1 = f1.to(device=target.get_device())
        return f1(pred, target)
