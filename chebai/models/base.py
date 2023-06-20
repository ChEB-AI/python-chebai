import logging
from lightning.pytorch.core.module import LightningModule
import torch

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

    def _process_batch(self, batch, batch_idx):
        return dict(features=batch.x, labels=batch.y.float(), model_kwargs=batch.additional_fields["model_kwargs"], loss_kwargs=batch.additional_fields["loss_kwargs"])

    def training_step(self, batch, batch_idx):
        result = self._execute(batch, batch_idx)
        self.log("train_loss", result["loss"].item(), batch_size=batch.x.shape[0], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return result

    def validation_step(self, batch, batch_idx):
        result = self._execute(batch, batch_idx)
        self.log("val_loss", result["loss"].item(), batch_size=batch.x.shape[0], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return result

    def _execute(self, batch, batch_idx):
        data = self._process_batch(batch, batch_idx)
        labels = data["labels"]
        model_output = self(data, **data.get("model_kwargs", dict()))
        loss = self.criterion(model_output, labels, **data["loss_kwargs"])
        return dict(data=data, labels=labels, output=model_output, loss=loss)

    def forward(self, x):
        raise NotImplementedError

    def configure_optimizers(self, **kwargs):
        return torch.optim.Adamax(self.parameters(), **self.optimizer_kwargs)
