import logging
import os
from typing import Optional, Union

from lightning import Trainer
from lightning.fabric.utilities.types import _PATH
from lightning.pytorch.loggers import CSVLogger
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
                init_kwargs = self.init_kwargs
                new_logger = CSVLoggerCVSupport(save_dir=self.logger.save_dir, name=self.logger.name,
                                                version=self.logger.version, fold=fold)
                init_kwargs['logger'] = new_logger
                new_trainer = Trainer(*self.init_args, **init_kwargs)
                print(f'Using logger.save_dir: {new_trainer.logger.save_dir}')
                print(f'Using logger.log_dir: {new_trainer.logger.log_dir}')
                new_trainer.fit(train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, *args, **kwargs)


# extend CSVLogger to include fold number in log path
class CSVLoggerCVSupport(CSVLogger):

    def __init__(self, save_dir: _PATH, name: str = "lightning_logs", version: Optional[Union[int, str]] = None,
                 prefix: str = "", flush_logs_every_n_steps: int = 100, fold: int = None):
        super().__init__(save_dir, name, version, prefix, flush_logs_every_n_steps)
        self.fold = fold

    @property
    def log_dir(self) -> str:
        """The log directory for this run.

        By default, it is named ``'version_${self.version}'`` but it can be overridden by passing a string value for the
        constructor's version parameter instead of ``None`` or an int.
        Additionally: Save data for each fold separately
        """
        # create a pseudo standard path
        version = self.version if isinstance(self.version, str) else f"version_{self.version}"
        if self.fold is None:
            return os.path.join(self.root_dir, version)
        return os.path.join(self.root_dir, version, f'fold{self.fold}')
