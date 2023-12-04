import logging
import os
from typing import Optional, Union

import pandas as pd
from lightning import Trainer, LightningModule
from lightning.fabric.utilities.types import _PATH
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.fabric.plugins.environments import SLURMEnvironment
from lightning_utilities.core.rank_zero import WarningCache
from lightning.pytorch.loggers import CSVLogger
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from lightning.pytorch.callbacks.model_checkpoint import _is_dir, rank_zero_warn

from chebai.preprocessing.datasets.base import XYBaseDataModule
from chebai.preprocessing.collate import RaggedCollater
from chebai.preprocessing.reader import CLS_TOKEN, ChemDataReader
from torch.nn.utils.rnn import pad_sequence
import torch
import pandas as pd

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
                new_logger = CSVLoggerCVSupport(save_dir=self.logger.save_dir, name=self.logger.name,
                                                version=self.logger.version, fold=fold)
                init_kwargs['logger'] = new_logger
                new_trainer = Trainer(*self.init_args, **init_kwargs)
                print(f'Logging this fold at {new_trainer.logger.log_dir}')
                new_trainer.fit(train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, *args, **kwargs)

    def predict_from_file(self, model: LightningModule, checkpoint_path: _PATH, input_path: _PATH,
                          save_to: _PATH='predictions.csv', classes_path: Optional[_PATH] = None):
        loaded_model= model.__class__.load_from_checkpoint(checkpoint_path)
        with open(input_path, 'r') as input:
            smiles_strings = [inp.strip() for inp in input.readlines()]
        predictions = self._predict_smiles(loaded_model, smiles_strings)
        predictions_df = pd.DataFrame(predictions.detach().numpy())
        if classes_path is not None:
            with open(classes_path, 'r') as f:
                predictions_df.columns = [cls.strip() for cls in f.readlines()]
        predictions_df.index = smiles_strings
        predictions_df.to_csv(save_to)


    def _predict_smiles(self, model: LightningModule, smiles: list[str]):
        reader = ChemDataReader()
        parsed_smiles = [reader._read_data(s) for s in smiles]
        x = pad_sequence([torch.tensor(a) for a in parsed_smiles], batch_first=True)
        cls_tokens = (torch.ones(x.shape[0], dtype=torch.int, device=model.device).unsqueeze(-1) * CLS_TOKEN)
        features = torch.cat((cls_tokens, x), dim=1)
        model_output = model({'features': features})
        preds = torch.sigmoid(model_output['logits'])

        print(preds.shape)
        return preds


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
