import logging
import os
from typing import Optional, Union, List

import pandas as pd
from lightning import Trainer, LightningModule
from lightning.fabric.utilities.types import _PATH
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from lightning.fabric.plugins.environments import SLURMEnvironment
from lightning_utilities.core.rank_zero import (
    WarningCache,
    rank_zero_warn,
    rank_zero_info,
)
from lightning.pytorch.loggers import CSVLogger
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from lightning.pytorch.callbacks.model_checkpoint import _is_dir

from chebai.loggers.custom import CustomLogger
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
                kfold.split(
                    datamodule.train_val_data,
                    [data["labels"] for data in datamodule.train_val_data],
                )
            ):
                train_dataloader = datamodule.train_dataloader(ids=train_ids)
                val_dataloader = datamodule.val_dataloader(ids=val_ids)
                init_kwargs = self.init_kwargs
                new_trainer = InnerCVTrainer(*self.init_args, **init_kwargs)
                logger = new_trainer.logger
                if isinstance(logger, CustomLogger):
                    logger.set_fold(fold)
                    rank_zero_info(f"Logging this fold at {logger.experiment.dir}")
                else:
                    rank_zero_warn(
                        f"Using k-fold cross-validation without an adapted logger class"
                    )
                new_trainer.fit(
                    train_dataloaders=train_dataloader,
                    val_dataloaders=val_dataloader,
                    *args,
                    **kwargs,
                )

    def predict_from_file(
        self,
        model: LightningModule,
        checkpoint_path: _PATH,
        input_path: _PATH,
        save_to: _PATH = "predictions.csv",
        classes_path: Optional[_PATH] = None,
    ):
        loaded_model = model.__class__.load_from_checkpoint(checkpoint_path)
        with open(input_path, "r") as input:
            smiles_strings = [inp.strip() for inp in input.readlines()]
        loaded_model.eval()
        predictions = self._predict_smiles(loaded_model, smiles_strings)
        predictions_df = pd.DataFrame(predictions.detach().numpy())
        if classes_path is not None:
            with open(classes_path, "r") as f:
                predictions_df.columns = [cls.strip() for cls in f.readlines()]
        predictions_df.index = smiles_strings
        predictions_df.to_csv(save_to)

    def _predict_smiles(self, model: LightningModule, smiles: List[str]):
        reader = ChemDataReader()
        parsed_smiles = [reader._read_data(s) for s in smiles]
        x = pad_sequence([torch.tensor(a) for a in parsed_smiles], batch_first=True)
        cls_tokens = (
            torch.ones(x.shape[0], dtype=torch.int, device=model.device).unsqueeze(-1)
            * CLS_TOKEN
        )
        features = torch.cat((cls_tokens, x), dim=1)
        model_output = model({"features": features})
        preds = torch.sigmoid(model_output["logits"])

        print(preds.shape)
        return preds

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
