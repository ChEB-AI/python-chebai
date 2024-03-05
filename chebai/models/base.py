from typing import Optional
import logging
import typing

from lightning.pytorch.core.module import LightningModule
import torch

from chebai.preprocessing.structures import XYData

logging.getLogger("pysmiles").setLevel(logging.CRITICAL)

_MODEL_REGISTRY = dict()


class ChebaiBaseNet(LightningModule):
    NAME = None
    LOSS = torch.nn.BCEWithLogitsLoss

    def __init__(
        self,
        criterion: torch.nn.Module = None,
        out_dim: Optional[int] = None,
        train_metrics: Optional[torch.nn.Module] = None,
        val_metrics: Optional[torch.nn.Module] = None,
        test_metrics: Optional[torch.nn.Module] = None,
        pass_loss_kwargs=True,
        optimizer_kwargs: Optional[typing.Dict] = None,
        **kwargs,
    ):
        super().__init__()
        self.criterion = criterion
        self.save_hyperparameters(
            ignore=["criterion", "train_metrics", "val_metrics", "test_metrics"]
        )
        self.out_dim = out_dim
        if optimizer_kwargs:
            self.optimizer_kwargs = optimizer_kwargs
        else:
            self.optimizer_kwargs = dict()
        self.train_metrics = train_metrics
        self.validation_metrics = val_metrics
        self.test_metrics = test_metrics
        self.pass_loss_kwargs = pass_loss_kwargs

    def __init_subclass__(cls, **kwargs):
        if cls.NAME in _MODEL_REGISTRY:
            raise ValueError(f"Model {cls.NAME} does already exist")
        else:
            _MODEL_REGISTRY[cls.NAME] = cls

    def _get_prediction_and_labels(self, data, labels, output):
        return output, labels

    def _process_labels_in_batch(self, batch):
        return batch.y.float()

    def _process_batch(self, batch, batch_idx):
        return dict(
            features=batch.x,
            labels=self._process_labels_in_batch(batch),
            model_kwargs=batch.additional_fields["model_kwargs"],
            loss_kwargs=batch.additional_fields["loss_kwargs"],
            idents=batch.additional_fields["idents"],
        )

    def _process_for_loss(self, model_output, labels, loss_kwargs):
        return model_output, labels, loss_kwargs

    def training_step(self, batch, batch_idx):
        return self._execute(
            batch, batch_idx, self.train_metrics, prefix="train_", sync_dist=True
        )

    def validation_step(self, batch, batch_idx):
        return self._execute(
            batch, batch_idx, self.validation_metrics, prefix="val_", sync_dist=True
        )

    def test_step(self, batch, batch_idx):
        return self._execute(
            batch, batch_idx, self.test_metrics, prefix="test_", sync_dist=True
        )

    def predict_step(self, batch, batch_idx, **kwargs):
        return self._execute(batch, batch_idx, self.test_metrics, prefix="", log=False)

    def _execute(self, batch, batch_idx, metrics, prefix="", log=True, sync_dist=False):
        assert isinstance(batch, XYData)
        batch = batch.to(self.device)
        data = self._process_batch(batch, batch_idx)
        labels = data["labels"]
        model_output = self(data, **data.get("model_kwargs", dict()))
        pr, tar = self._get_prediction_and_labels(data, labels, model_output)
        d = dict(data=data, labels=labels, output=model_output, preds=pr)
        if log:
            if self.criterion is not None:
                loss_data, loss_labels, loss_kwargs_candidates = self._process_for_loss(
                    model_output, labels, data.get("loss_kwargs", dict())
                )
                loss_kwargs = dict()
                if self.pass_loss_kwargs:
                    loss_kwargs = loss_kwargs_candidates
                loss = self.criterion(loss_data, loss_labels, **loss_kwargs)
                d["loss"] = loss
                self.log(
                    f"{prefix}loss",
                    loss.item(),
                    batch_size=len(batch),
                    on_step=False,  # avoid crowded metrics file (before True)
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                    sync_dist=sync_dist,
                )
            if metrics and labels is not None:
                for metric_name, metric in metrics.items():
                    metric.update(pr, tar)
                self._log_metrics(prefix, metrics, len(batch))
        return d

    def _log_metrics(self, prefix, metrics, batch_size):
        # don't use sync_dist=True if the metric is a torchmetrics-metric
        # (see https://github.com/Lightning-AI/pytorch-lightning/discussions/6501#discussioncomment-569757)
        for metric_name, metric in metrics.items():
            m = None  # m = metric.compute()
            if isinstance(m, dict):
                # todo: is this case needed? it requires logging values directly which does not give accurate results
                # with the current metric-setup
                for k, m2 in m.items():
                    self.log(
                        f"{prefix}{metric_name}{k}",
                        m2,
                        batch_size=batch_size,
                        on_step=False,
                        on_epoch=True,
                        prog_bar=True,
                        logger=True,
                    )
            else:
                self.log(
                    f"{prefix}{metric_name}",
                    metric,
                    batch_size=batch_size,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                )

    def forward(self, x):
        raise NotImplementedError

    def configure_optimizers(self, **kwargs):
        return torch.optim.Adamax(self.parameters(), **self.optimizer_kwargs)
