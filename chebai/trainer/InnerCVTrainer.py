import logging
import os
from typing import Optional

from lightning import Trainer, LightningModule
from lightning.fabric.utilities.types import _PATH
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from lightning.pytorch.callbacks.model_checkpoint import _is_dir, rank_zero_warn

from chebai.loggers.custom import CustomLogger
from chebai.preprocessing.datasets.base import XYBaseDataModule

log = logging.getLogger(__name__)


class InnerCVTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        self.init_args = args
        self.init_kwargs = kwargs
        super().__init__(*args, **kwargs)
        # instantiation custom logger connector
        self._logger_connector.on_trainer_init(self.logger, 1)

    def cv_fit(self, datamodule: XYBaseDataModule, n_splits: int = -1, *args, **kwargs):
        if n_splits < 2:
            self.fit(datamodule=datamodule, *args, **kwargs)
        else:
            datamodule.prepare_data()
            datamodule.setup()

            kfold = MultilabelStratifiedKFold(n_splits=n_splits)

            for fold, (train_ids, val_ids) in enumerate(
                    kfold.split(datamodule.train_val_data, [data['labels'] for data in datamodule.train_val_data])):
                train_dataloader = datamodule.train_dataloader(ids=train_ids)
                val_dataloader = datamodule.val_dataloader(ids=val_ids)
                init_kwargs = self.init_kwargs
                new_trainer = Trainer(*self.init_args, **init_kwargs)
                logger = new_trainer.logger
                if isinstance(logger, CustomLogger):
                    logger.set_fold(fold)
                    print(f'Logging this fold at {logger.experiment.dir}')
                else:
                    rank_zero_warn(f"Using k-fold cross-validation without an adapted logger class")
                new_trainer.fit(train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, *args, **kwargs)

    @property
    def log_dir(self) -> Optional[str]:
        if len(self.loggers) > 0:
            logger = self.loggers[0]
            if isinstance(logger, WandbLogger):
                dirpath = logger.experiment.dir
            else:
                dirpath = self.loggers[0].log_dir
        else:
            dirpath = self.default_root_dir

        dirpath = self.strategy.broadcast(dirpath)
        return dirpath
