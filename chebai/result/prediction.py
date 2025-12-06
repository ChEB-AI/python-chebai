import os
from typing import List, Optional

import pandas as pd
import torch
from jsonargparse import CLI
from lightning.fabric.utilities.types import _PATH
from lightning.pytorch.cli import instantiate_module
from torch.utils.data import DataLoader

from chebai.models.base import ChebaiBaseNet
from chebai.preprocessing.datasets.base import XYBaseDataModule


class Predictor:
    def __init__(self, checkpoint_path: _PATH, batch_size: Optional[int] = None):
        """Initializes the Predictor with a model loaded from the checkpoint.

        Args:
            checkpoint_path: Path to the model checkpoint.
            batch_size: Optional batch size for the DataLoader. If not provided,
                the default from the datamodule will be used.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt_file = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )

        self._dm_hparams = ckpt_file["datamodule_hyper_parameters"]
        # self._dm_hparams.pop("splits_file_path")
        self._dm: XYBaseDataModule = instantiate_module(
            XYBaseDataModule, self._dm_hparams
        )
        print(f"Loaded datamodule class: {self._dm.__class__.__name__}")
        if batch_size is not None and int(batch_size) > 0:
            self._dm.batch_size = int(batch_size)

        self._model_hparams = ckpt_file["hyper_parameters"]
        self._model: ChebaiBaseNet = instantiate_module(
            ChebaiBaseNet, self._model_hparams
        )
        self._model.eval()
        # TODO: Enable torch.compile when supported
        # model = torch.compile(model)
        print(f"Loaded model class: {self._model.__class__.__name__}")

    def predict_from_file(
        self,
        smiles_file_path: _PATH,
        save_to: _PATH = "predictions.csv",
        classes_path: Optional[_PATH] = None,
    ) -> None:
        """
        Loads a model from a checkpoint and makes predictions on input data from a file.

        Args:
            smiles_file_path: Path to the input file containing SMILES strings.
            save_to: Path to save the predictions CSV file.
            classes_path: Optional path to a file containing class names:
                if no class names are provided, code will try to get the class path
                from the datamodule, else the columns will be numbered.
        """
        with open(smiles_file_path, "r") as input:
            smiles_strings = [inp.strip() for inp in input.readlines()]

        preds: torch.Tensor = self.predict_smiles(smiles=smiles_strings)

        predictions_df = pd.DataFrame(torch.cat(preds).detach().cpu().numpy())

        def _add_class_columns(class_file_path: _PATH):
            with open(class_file_path, "r") as f:
                predictions_df.columns = [cls.strip() for cls in f.readlines()]

        if classes_path is not None:
            _add_class_columns(classes_path)
        elif os.path.exists(self._dm.classes_txt_file_path):
            _add_class_columns(self._dm.classes_txt_file_path)

        predictions_df.index = smiles_strings
        predictions_df.to_csv(save_to)

    @torch.inference_mode()
    def predict_smiles(
        self,
        smiles: List[str],
    ) -> torch.Tensor:
        """
        Predicts the output for a list of SMILES strings using the model.

        Args:
            smiles: A list of SMILES strings.

        Returns:
            A tensor containing the predictions.
        """
        # For certain data prediction piplines, we may need model hyperparameters
        pred_dl: DataLoader = self._dm.predict_dataloader(
            smiles_list=smiles, model_hparams=self._model_hparams
        )

        preds = []
        for batch_idx, batch in enumerate(pred_dl):
            # For certain model prediction pipelines, we may need data module hyperparameters
            preds.append(
                self._model.predict_step(batch, batch_idx, dm_hparams=self._dm_hparams)
            )

        return torch.cat(preds)


class MainPredictor:
    @staticmethod
    def predict_from_file(
        checkpoint_path: _PATH,
        smiles_file_path: _PATH,
        save_to: _PATH = "predictions.csv",
        classes_path: Optional[_PATH] = None,
        batch_size: Optional[int] = None,
    ) -> None:
        predictor = Predictor(checkpoint_path, batch_size)
        predictor.predict_from_file(
            smiles_file_path,
            save_to,
            classes_path,
        )

    @staticmethod
    def predict_smiles(
        checkpoint_path: _PATH,
        smiles: List[str],
        batch_size: Optional[int] = None,
    ) -> torch.Tensor:
        predictor = Predictor(checkpoint_path, batch_size)
        return predictor.predict_smiles(smiles)


if __name__ == "__main__":
    # python chebai/result/prediction.py  predict_from_file --help
    # python chebai/result/prediction.py  predict_smiles --help
    CLI(MainPredictor, as_positional=False)
