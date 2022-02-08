import itertools
import logging
import os
import sys

from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch import nn
from torchmetrics import F1, MeanSquaredError
from torchmetrics import functional as tmf
import pytorch_lightning as pl
import torch
import torchmetrics
import tqdm

from chebai.preprocessing.datasets import XYBaseDataModule

logging.getLogger("pysmiles").setLevel(logging.CRITICAL)


class JCIBaseNet(pl.LightningModule):
    NAME = None

    def __init__(self, out_dim=None, **kwargs):
        super().__init__()
        self.out_dim = out_dim
        weights = kwargs.get("weights", None)
        if weights is not None:
            self.loss = nn.BCEWithLogitsLoss(pos_weight=weights)
        else:
            self.loss = nn.BCEWithLogitsLoss()
        thres = kwargs.get("threshold", 0.5)
        self.f1 = F1(threshold=thres)
        self.mse = MeanSquaredError()
        self.lr = kwargs.get("lr", 1e-4)

        for metric in ["F1", "Precision", "Recall"]:
            for agg in ["micro", "samples", "macro", "weighted"]:
                setattr(
                    self,
                    metric + agg,
                    getattr(torchmetrics, metric)(
                        threshold=thres, average=agg, num_classes=500
                    ),
                )

    def _execute(self, batch, batch_idx):
        data = self._get_data_and_labels(batch, batch_idx)
        labels = data["labels"]
        pred = self(data["features"], **data.get("model_kwargs", dict()))["logits"]
        labels = labels.float()
        return pred, labels

    def _get_data_and_labels(self, batch, batch_idx):
        return dict(features=batch.x, labels=batch.y.float())

    def calculate_metrics(self, pred, labels):
        loss = self.loss(pred, labels)
        f1 = self.f1(target=labels.int(), preds=torch.sigmoid(pred))
        mse = self.mse(labels, torch.sigmoid(pred))
        return loss, f1, mse

    def training_step(self, *args, **kwargs):
        loss, f1, mse = self.calculate_metrics(*self._execute(*args, **kwargs))
        self.log(
            "train_loss",
            loss.detach().item(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "train_f1",
            f1.detach().item(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "train_mse",
            mse.detach().item(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def validation_step(self, *args, **kwargs):
        with torch.no_grad():
            loss, f1, mse = self.calculate_metrics(*self._execute(*args, **kwargs))
            self.log(
                "val_loss",
                loss.detach().item(),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            self.log(
                "val_f1",
                f1.detach().item(),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            self.log(
                "val_mse",
                mse.detach().item(),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            return loss

    def test_step(self, *args, **kwargs):
        with torch.no_grad():
            pred, labels = self._execute(*args, **kwargs)
            l = labels.int()
            p = torch.sigmoid(pred)
            for name in ["F1", "Precision", "Recall"]:
                for agg in ["micro", "samples", "macro", "weighted"]:
                    metric = getattr(self, name + agg)
                    self.log(
                        name + "_" + agg,
                        metric(preds=p, target=l),
                        on_step=False,
                        on_epoch=True,
                        prog_bar=True,
                        logger=True,
                    )
            return self.loss(p, l.float())

    def forward(self, x):
        raise NotImplementedError

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    @classmethod
    def test(cls, data, name, checkpoint_path):
        data.prepare_data()
        data.setup()
        name += "__" + "_".join(data.full_identifier)
        test_data = data.test_dataloader()

        # Calculate weights per class

        model = cls.load_from_checkpoint(checkpoint_path)

        trainer = pl.Trainer(
            replace_sampler_ddp=False,
        )

        test = trainer.test(model, test_data)
        print(test)

    @classmethod
    def run(
        cls,
        data,
        name,
        model_args: list = None,
        model_kwargs: dict = None,
        weighted=False,
    ):
        if model_args is None:
            model_args = []
        if model_kwargs is None:
            model_kwargs = {}
        data.prepare_data()
        data.setup()
        name += "__" + "_".join(data.full_identifier)
        train_data = data.train_dataloader()
        val_data = data.val_dataloader()

        if weighted:
            weights = model_kwargs.get("weights")
            if weights is None:
                weights = 1 + torch.sum(
                    torch.cat([data.y for data in train_data]).float(), dim=0
                )
                weights = torch.mean(weights) / weights
                name += "__weighted"
            model_kwargs["weights"] = weights
        else:
            try:
                model_kwargs.pop("weights")
            except KeyError:
                pass

        if torch.cuda.is_available():
            trainer_kwargs = dict(gpus=-1, accelerator="ddp")
        else:
            trainer_kwargs = dict(gpus=0)

        tb_logger = pl_loggers.TensorBoardLogger("logs/", name=name)
        best_checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(tb_logger.log_dir, "best_checkpoints"),
            filename="{epoch}-{val_f1:.7f}",
            save_top_k=5,
            monitor="val_f1",
            mode="max",
        )
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(tb_logger.log_dir, "periodic_checkpoints"),
            filename="{epoch}-{val_f1:.7f}",
            every_n_epochs=5,
            save_top_k=-1,
            save_last=True,
            verbose=True,
        )

        # Calculate weights per class

        net = cls(*model_args, **model_kwargs)

        # Early stopping seems to be bugged right now with ddp accelerator :(
        es = EarlyStopping(
            monitor="val_f1", patience=10, min_delta=0.00, verbose=False, mode="max"
        )

        trainer = pl.Trainer(
            logger=tb_logger,
            min_epochs=model_kwargs.get("epochs", 100),
            callbacks=[best_checkpoint_callback, checkpoint_callback, es],
            replace_sampler_ddp=False,
            **trainer_kwargs
        )
        trainer.fit(net, train_data, val_dataloaders=val_data)

    def pred(self, feature, batch_index=0):
        return (
            torch.sigmoid(self.predict_step(feature.to(self.device), batch_index))
            .detach()
            .cpu()
        )
