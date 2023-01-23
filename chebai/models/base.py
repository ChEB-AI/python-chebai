import itertools
import logging
import os
import sys

from pytorch_lightning import loggers as pl_loggers
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


class JCIBaseNet(pl.LightningModule):
    NAME = None
    LOSS = torch.nn.BCEWithLogitsLoss

    def __init__(self, loss_cls=None, out_dim=None, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        if out_dim and out_dim > 1:
            task = "multilabel"
        else:
            task = "binary"
        self.out_dim = out_dim
        weights = kwargs.get("weights", None)
        if loss_cls is None:
            loss_cls = torch.nn.BCEWithLogitsLoss
        if weights is not None:
            self.loss = loss_cls(pos_weight=weights)
        else:
            self.loss = loss_cls()

        self.optimizer_kwargs = kwargs.get("optimizer_kwargs", dict())
        self.thres = kwargs.get("threshold", 0.5)
        self.metrics = ["F1Score", "Precision", "Recall", "AUROC"]
        self.metric_aggs = ["micro"]
        for metric in self.metrics:
            for agg in self.metric_aggs:
                setattr(
                    self,
                    metric + agg,
                    getattr(torchmetrics, metric)(
                        threshold=self.thres,
                        average=agg,
                        task=task,
                        num_labels=self.out_dim,
                    ),
                )

    def _get_prediction_and_labels(self, data, labels, output):
        return output, labels

    def _get_data_and_labels(self, batch, batch_idx):
        return dict(features=batch.x, labels=batch.y.float())

    def _execute(self, batch, batch_idx):
        data = self._get_data_and_labels(batch, batch_idx)
        labels = data["labels"]
        model_output = self(data, **data.get("model_kwargs", dict()))
        return data, labels, model_output

    def _get_data_for_loss(self, model_output, labels):
        return dict(input=model_output, target=labels.float())

    def training_step(self, *args, **kwargs):
        return self.calculate_all_metrics("train_", *args, on_step=True, **kwargs)

    def validation_step(self, *args, **kwargs):
        with torch.no_grad():
            return self.calculate_all_metrics("val_", *args, **kwargs)

    def test_step(self, *args, **kwargs):
        with torch.no_grad():
            return self.calculate_all_metrics("test_", *args, **kwargs)

    def calculate_all_metrics(self, prefix, *args, on_step=False, **kwargs):
        data, labels, model_output = self._execute(*args, **kwargs)
        loss = self.loss(**self._get_data_for_loss(model_output, labels))
        with torch.no_grad():
            p, l = self._get_prediction_and_labels(data, labels, model_output)
            for name in self.metrics:
                for agg in self.metric_aggs:
                    metric = getattr(self, name + agg)
                    self.log(
                        prefix + name + "_" + agg,
                        metric(preds=p.detach().cpu(), target=l.detach().cpu()),
                        on_step=on_step,
                        on_epoch=True,
                        prog_bar=True,
                        logger=True,
                        batch_size=p.shape[0],
                    )
            self.log(
                prefix + "loss",
                loss.detach(),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                batch_size=p.shape[0],
            )
        return loss

    def forward(self, x):
        raise NotImplementedError

    def configure_optimizers(self, **kwargs):
        optimizer = torch.optim.Adamax(self.parameters(), **self.optimizer_kwargs)
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
        epochs,
        model_args: list = None,
        model_kwargs: dict = None,
        loss=torch.nn.BCELoss,
        weighted=False,
        version=None,
        **kwargs
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

        tb_logger = pl_loggers.TensorBoardLogger("logs/", name=name, version=version)
        if os.path.isdir(tb_logger.log_dir):
            raise IOError("Fixed logging directory does already exist:", tb_logger.log_dir)
        best_checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(tb_logger.log_dir, "best_checkpoints"),
            filename="{epoch}-{val_F1Score_micro:.4f}--{val_loss:.4f}",
            save_top_k=5,
            monitor="val_loss",
            mode="min",
        )
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(tb_logger.log_dir, "periodic_checkpoints"),
            filename="{epoch}-{val_F1Score_micro:.4f}--{val_loss:.4f}",
            every_n_epochs=5,
            save_top_k=-1,
            save_last=True,
            verbose=True,
        )

        # Calculate weights per class

        net = cls(*model_args, loss_cls=loss, **model_kwargs, **kwargs)

        # Early stopping seems to be bugged right now with ddp accelerator :(
        es = EarlyStopping(
            monitor="val_loss", patience=10, min_delta=0.00, verbose=False, mode="min"
        )

        trainer = pl.Trainer(
            logger=tb_logger,
            min_epochs=epochs,
            max_epochs=epochs,
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
