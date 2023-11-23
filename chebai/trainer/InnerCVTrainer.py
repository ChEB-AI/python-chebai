import logging
import os
from typing import Optional, Union, Iterable

from lightning import Trainer, LightningModule
from lightning.fabric.utilities.types import _PATH
from lightning.pytorch.trainer.connectors.logger_connector import _LoggerConnector
from lightning.fabric.loggers.tensorboard import _TENSORBOARD_AVAILABLE, _TENSORBOARDX_AVAILABLE
from lightning.pytorch.loggers import CSVLogger, Logger, TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.fabric.plugins.environments import SLURMEnvironment
from lightning_utilities.core.rank_zero import WarningCache

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from lightning.pytorch.callbacks.model_checkpoint import _is_dir, rank_zero_warn

from chebai.preprocessing.datasets.base import XYBaseDataModule

log = logging.getLogger(__name__)


class InnerCVTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        self.init_args = args
        self.init_kwargs = kwargs
        super().__init__(*args, **kwargs)
        # instantiation custom logger connector
        self._logger_connector = _LoggerConnectorCVSupport(self)
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
                new_logger = CSVLoggerCVSupport(save_dir=self.logger.save_dir, name=self.logger.name,
                                                version=self.logger.version, fold=fold)
                init_kwargs['logger'] = new_logger
                new_trainer = Trainer(*self.init_args, **init_kwargs)
                self._logger_connector = _LoggerConnectorCVSupport(self)
                self._logger_connector.on_trainer_init(self.logger, 1)
                print(f'Logging this fold at {new_trainer.logger.log_dir}')
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
        return os.path.join(self.root_dir, version, f'fold_{self.fold}')


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
            if isinstance(cv_logger, CSVLoggerCVSupport) and cv_logger.fold is not None:
                # log_dir includes fold
                ckpt_path = os.path.join(cv_logger.log_dir, "checkpoints")
            else:
                ckpt_path = os.path.join(save_dir, str(name), version, "checkpoints")
        else:
            # if no loggers, use default_root_dir
            ckpt_path = os.path.join(trainer.default_root_dir, "checkpoints")

        print(f'Now using checkpoint path {ckpt_path}')
        return ckpt_path


warning_cache = WarningCache()


class _LoggerConnectorCVSupport(_LoggerConnector):
    def configure_logger(self, logger: Union[bool, Logger, Iterable[Logger]]) -> None:
        if not logger:
            # logger is None or logger is False
            self.trainer.loggers = []
        elif logger is True:
            # default logger
            if _TENSORBOARD_AVAILABLE or _TENSORBOARDX_AVAILABLE:
                logger_ = TensorBoardLogger(save_dir=self.trainer.default_root_dir, version=SLURMEnvironment.job_id())
            else:
                warning_cache.warn(
                    "Starting from v1.9.0, `tensorboardX` has been removed as a dependency of the `lightning.pytorch`"
                    " package, due to potential conflicts with other packages in the ML ecosystem. For this reason,"
                    " `logger=True` will use `CSVLogger` as the default logger, unless the `tensorboard`"
                    " or `tensorboardX` packages are found."
                    " Please `pip install lightning[extra]` or one of them to enable TensorBoard support by default"
                )
                logger_ = CSVLogger(save_dir=self.trainer.default_root_dir)  # type: ignore[assignment]
            self.trainer.loggers = [logger_]
        elif isinstance(logger, Iterable):
            self.trainer.loggers = list(logger)
        else:
            self.trainer.loggers = [logger]
