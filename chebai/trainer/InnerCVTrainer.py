import logging
import os
from typing import Optional, Union, Iterable

from lightning import Trainer, LightningModule
from lightning.fabric.utilities.types import _PATH
from lightning.pytorch.trainer.connectors.logger_connector import _LoggerConnector
from lightning.fabric.loggers.tensorboard import _TENSORBOARD_AVAILABLE, _TENSORBOARDX_AVAILABLE
from lightning.pytorch.loggers import CSVLogger, Logger, TensorBoardLogger, WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.fabric.plugins.environments import SLURMEnvironment
from lightning_utilities.core.rank_zero import WarningCache

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


class ModelCheckpointCVSupport(ModelCheckpoint):

    def setup(self, trainer: "Trainer", pl_module: "LightningModule", stage: str) -> None:
        """Same as in parent class, duplicated to be able to call self.__resolve_ckpt_dir"""
        if self.dirpath is not None:
            self.dirpath = None
        dirpath = self.__resolve_ckpt_dir(trainer)
        dirpath = trainer.strategy.broadcast(dirpath)
        self.dirpath = dirpath
        if trainer.is_global_zero and stage == "fit":
            self.__warn_if_dir_not_empty(self.dirpath)

    def __warn_if_dir_not_empty(self, dirpath: _PATH) -> None:
        """Same as in parent class, duplicated because method in parent class is not accessible"""
        if self.save_top_k != 0 and _is_dir(self._fs, dirpath, strict=True) and len(self._fs.ls(dirpath)) > 0:
            rank_zero_warn(f"Checkpoint directory {dirpath} exists and is not empty.")

    def __resolve_ckpt_dir(self, trainer: "Trainer") -> _PATH:
        """Determines model checkpoint save directory at runtime. Reference attributes from the trainer's logger to
        determine where to save checkpoints. The path for saving weights is set in this priority:

        1.  The ``ModelCheckpoint``'s ``dirpath`` if passed in
        2.  The ``Logger``'s ``log_dir`` if the trainer has loggers
        3.  The ``Trainer``'s ``default_root_dir`` if the trainer has no loggers

        The path gets extended with subdirectory "checkpoints".

        """
        print(f'Resolving checkpoint dir (with cross-validation)')
        if self.dirpath is not None:
            # short circuit if dirpath was passed to ModelCheckpoint
            return self.dirpath
        if len(trainer.loggers) > 0:
            if trainer.loggers[0].save_dir is not None:
                save_dir = trainer.loggers[0].save_dir
            else:
                save_dir = trainer.default_root_dir
            name = trainer.loggers[0].name
            version = trainer.loggers[0].version
            version = version if isinstance(version, str) else f"version_{version}"
            cv_logger = trainer.loggers[0]
            if isinstance(cv_logger, CustomLogger) and cv_logger.fold is not None:
                # log_dir includes fold
                ckpt_path = os.path.join(cv_logger.log_dir, "checkpoints")
            else:
                ckpt_path = os.path.join(save_dir, str(name), version, "checkpoints")
        else:
            # if no loggers, use default_root_dir
            ckpt_path = os.path.join(trainer.default_root_dir, "checkpoints")

        print(f'Now using checkpoint path {ckpt_path}')
        return ckpt_path
