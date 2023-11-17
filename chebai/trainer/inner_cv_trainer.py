import inspect
from typing import Optional, Union
import logging
from lightning import Trainer, LightningDataModule, LightningModule
import lightning as pl
from sklearn import model_selection

from chebai.preprocessing.datasets.base import XYBaseDataModule

log = logging.getLogger(__name__)

class InnerCVTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        self.init_args = args
        self.init_kwargs = kwargs
        super().__init__(*args, **kwargs)


    def cv_fit(self, datamodule: XYBaseDataModule, n_splits: int = -1, *args, **kwargs):
        if n_splits < 2:
            self.fit(datamodule=datamodule, *args, **kwargs)
        else:
            datamodule.prepare_data()
            datamodule.setup()

            kfold = model_selection.KFold(n_splits=n_splits)

            for fold, (train_ids, val_ids) in enumerate(kfold.split(datamodule.train_val_data)):
                train_dataloader = datamodule.train_dataloader(ids=train_ids)
                val_dataloader = datamodule.val_dataloader(ids=val_ids)

                new_trainer = Trainer(*self.init_args, **self.init_kwargs)
                new_trainer.fit(train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, *args, **kwargs)

