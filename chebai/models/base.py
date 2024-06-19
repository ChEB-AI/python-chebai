import logging
from lightning.pytorch.core.module import LightningModule
import torch
from typing import Optional, Dict, Any
import pickle

logging.getLogger("pysmiles").setLevel(logging.CRITICAL)

_MODEL_REGISTRY = dict()


class ChebaiBaseNet(LightningModule):
    NAME = None
    LOSS = torch.nn.BCEWithLogitsLoss

    def __init__(
        self,
        criterion: torch.nn.Module = None,
        out_dim=None,
        metrics: Optional[Dict[str, torch.nn.Module]] = None,
        pass_loss_kwargs=True,
        **kwargs,
    ):
        super().__init__()
        self.criterion = criterion
        self.save_hyperparameters(ignore=["criterion"])
        self.out_dim = out_dim
        self.optimizer_kwargs = kwargs.get("optimizer_kwargs", dict())
        self.train_metrics = metrics["train"]
        self.validation_metrics = metrics["validation"]
        self.test_metrics = metrics["test"]
        self.pass_loss_kwargs = pass_loss_kwargs

    def __init_subclass__(cls, **kwargs):
        if cls.NAME in _MODEL_REGISTRY:
            raise ValueError(f"Model {cls.NAME} does already exist")
        else:
            _MODEL_REGISTRY[cls.NAME] = cls

    def _get_prediction_and_labels(self, data, labels, output):
        return output, labels

    def _process_batch(self, batch, batch_idx):
        return dict(
            features=batch.x,
            labels=batch.y.float(),
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
        data = self._process_batch(batch, batch_idx)
        labels = data["labels"]
        model_output = self(data, **data.get("model_kwargs", dict()))
        d = dict(data=data, labels=labels, output=model_output)
        if log:
            if self.criterion is not None:
                loss_data, loss_labels, loss_kwargs_candidates = self._process_for_loss(
                    model_output, labels, data.get("loss_kwargs", dict())
                )
                loss_kwargs = dict()
                if self.pass_loss_kwargs:
                    loss_kwargs = loss_kwargs_candidates

                with open('./weights_2.pkl', 'rb') as f:
                    weights_beta = pickle.load(f)
                with open('./weights.pkl', 'rb') as f:
                    weights_simple = pickle.load(f)   
                loss_kwargs["weights_beta"] = torch.tensor(weights_beta).to('cuda')
                loss_kwargs["weights_simple"] = torch.tensor(weights_simple).to('cuda')
                loss_kwargs["model"] = self 
                loss = self.criterion(loss_data, loss_labels, **loss_kwargs)
                d["loss"] = loss
                self.log(
                    f"{prefix}loss",
                    loss.item(),
                    batch_size=batch.x.shape[0],
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                    sync_dist=sync_dist,
                )
            if metrics and labels is not None:
                pr, tar = self._get_prediction_and_labels(data, labels, model_output)
                for metric_name, metric in metrics.items():
                    m = metric(pr, tar)
                    if isinstance(m, dict):
                        for k, m2 in m.items():
                            self.log(
                                f"{prefix}{metric_name}{k}",
                                m2,
                                batch_size=batch.x.shape[0],
                                on_step=False,
                                on_epoch=True,
                                prog_bar=True,
                                logger=True,
                                sync_dist=sync_dist,
                            )
                    else:
                        self.log(
                            f"{prefix}{metric_name}",
                            m,
                            batch_size=batch.x.shape[0],
                            on_step=False,
                            on_epoch=True,
                            prog_bar=True,
                            logger=True,
                            sync_dist=sync_dist,
                        )
        return d

    def forward(self, x):
        raise NotImplementedError

    def configure_optimizers(self, **kwargs):
        return torch.optim.Adamax(self.parameters(), **self.optimizer_kwargs)
