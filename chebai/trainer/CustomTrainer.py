import logging
from typing import List, Optional

import pandas as pd
import torch
from lightning import LightningModule, Trainer
from lightning.fabric.utilities.types import _PATH
from lightning.pytorch.loggers import WandbLogger
from torch.nn.utils.rnn import pad_sequence

from chebai.loggers.custom import CustomLogger
from chebai.preprocessing.reader import CLS_TOKEN, ChemDataReader

log = logging.getLogger(__name__)


class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        self.init_args = args
        self.init_kwargs = kwargs
        super().__init__(*args, **kwargs)
        # instantiation custom logger connector
        self._logger_connector.on_trainer_init(self.logger, 1)
        # log additional hyperparameters to wandb
        if isinstance(self.logger, CustomLogger):
            custom_logger = self.logger
            assert isinstance(custom_logger, CustomLogger)
            if custom_logger.verbose_hyperparameters:
                log_kwargs = {}
                for key, value in self.init_kwargs.items():
                    log_key, log_value = self._resolve_logging_argument(key, value)
                    log_kwargs[log_key] = log_value
                self.logger.log_hyperparams(log_kwargs)

    def _resolve_logging_argument(self, key, value):
        if isinstance(value, list):
            key_value_pairs = [
                self._resolve_logging_argument(f"{key}_{i}", v)
                for i, v in enumerate(value)
            ]
            return key, {k: v for k, v in key_value_pairs}
        if not (
            isinstance(value, str)
            or isinstance(value, float)
            or isinstance(value, int)
            or value is None
        ):
            params = {"class": value.__class__}
            params.update(value.__dict__)
            return key, params
        else:
            return key, value

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
