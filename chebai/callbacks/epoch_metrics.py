from typing import Any

import torchmetrics
from lightning.pytorch.callbacks import Callback
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
        self.val_macro_adjust, self.train_macro_adjust = (
            None,
            None,
        )  # factor to compensate for not present classes

    @property
    def metric_name(self):
        raise NotImplementedError

    def apply_metric(self, target, pred, mode="test"):
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

    def _calculate_macro_adjust(self, labels):
        classes_present = torch.sum(torch.sum(labels, dim=0) > 0).item()
        total_classes = labels.shape[1]
        if classes_present > 0:
            macro_adjust = total_classes / classes_present
        else:
            macro_adjust = 1
            print(f"Warning: true-label for any class in dataset")
        return macro_adjust

    def on_train_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        # if self.train_macro_adjust is None:
        self.train_macro_adjust = self._calculate_macro_adjust(self.train_labels)
        if self.train_macro_adjust != 1:
            print(
                f"some classes are missing in train set, calculating macro-scores with adjustment factor {self.train_macro_adjust}"
            )

        pl_module.log(
            f"train_{self.metric_name}",
            self.apply_metric(self.train_labels, self.train_preds, mode="train"),
            sync_dist=True,
        )
        pl_module.log(f"train_macro_adjust", self.train_macro_adjust, sync_dist=True)

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
        self.val_macro_adjust = self._calculate_macro_adjust(self.val_labels)
        if self.val_macro_adjust != 1:
            print(
                f"some classes are missing in val set, calculating macro-scores with adjustment factor {self.val_macro_adjust}"
            )

        pl_module.log(
            f"val_{self.metric_name}",
            self.apply_metric(self.val_labels, self.val_preds, mode="val"),
            sync_dist=True,
        )
        pl_module.log(f"val_macro_adjust", self.val_macro_adjust, sync_dist=True)


class EpochLevelMacroF1(_EpochLevelMetric):
    @property
    def metric_name(self):
        return "ep_macro-f1"

    def apply_metric(self, target, pred, mode="train"):
        f1 = MultilabelF1Score(num_labels=self.num_labels, average="macro")
        if target.get_device() != -1:  # -1 == CPU
            f1 = f1.to(device=target.get_device())
        if mode == "train":
            return f1(pred, target) * self.train_macro_adjust
        elif mode == "val":
            return f1(pred, target) * self.val_macro_adjust
        else:
            return f1(pred, target)


class MacroF1(torchmetrics.Metric):
    def __init__(self, n_labels, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state(
            "true_positives", default=torch.empty((0, n_labels)), dist_reduce_fx="sum"
        )
        self.add_state(
            "positive_predictions",
            default=torch.empty((0, n_labels)),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "positive_labels", default=torch.empty((0, n_labels)), dist_reduce_fx="sum"
        )

    def update(self, preds: torch.Tensor, labels: torch.Tensor):
        self.true_positives += torch.sum(torch.logical_and(preds, labels), dim=1)
        self.positive_predictions += torch.sum(preds, dim=1)
        self.positive_labels += torch.sum(labels, dim=1)

    def compute(self):
        mask = torch.logical_and(
            self.positive_predictions > 0, self.positive_labels > 0
        )
        precision = self.true_positives[mask] / self.positive_predictions[mask]
        recall = self.true_positives[mask] / self.positive_labels[mask]
        classwise_f1 = (2 * precision * recall / (precision + recall)).nan_to_num()
        return torch.mean(classwise_f1)
