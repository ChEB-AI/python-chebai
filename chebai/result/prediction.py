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
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        super().__init__()

    def predict_from_file(
        self,
        checkpoint_path: _PATH,
        input_path: _PATH,
        save_to: _PATH = "predictions.csv",
        classes_path: Optional[_PATH] = None,
    ) -> None:
        """
        Loads a model from a checkpoint and makes predictions on input data from a file.

        Args:
            model: The model to use for predictions.
            checkpoint_path: Path to the model checkpoint.
            input_path: Path to the input file containing SMILES strings.
            save_to: Path to save the predictions CSV file.
            classes_path: Optional path to a file containing class names (if no class names are provided, columns will be numbered).
        """
        with open(input_path, "r") as input:
            smiles_strings = [inp.strip() for inp in input.readlines()]

        self.predict_smiles(
            checkpoint_path,
            smiles=smiles_strings,
            classes_path=classes_path,
            save_to=save_to,
        )

    @torch.inference_mode()
    def predict_smiles(
        self,
        checkpoint_path: _PATH,
        smiles: List[str],
        classes_path: Optional[_PATH] = None,
        save_to: Optional[_PATH] = None,
        **kwargs,
    ) -> torch.Tensor | None:
        """
        Predicts the output for a list of SMILES strings using the model.

        Args:
            model: The model to use for predictions.
            smiles: A list of SMILES strings.

        Returns:
            A tensor containing the predictions.
        """
        ckpt_file = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )

        dm_hparams = ckpt_file["datamodule_hyper_parameters"]
        dm_hparams.pop("splits_file_path")
        dm: XYBaseDataModule = instantiate_module(XYBaseDataModule, dm_hparams)
        print(f"Loaded datamodule class: {dm.__class__.__name__}")

        model_hparams = ckpt_file["hyper_parameters"]
        model: ChebaiBaseNet = instantiate_module(ChebaiBaseNet, model_hparams)
        model.eval()
        # model = torch.compile(model)
        print(f"Loaded model class: {model.__class__.__name__}")

        # For certain data prediction piplines, we may need model hyperparameters
        pred_dl: DataLoader = dm.predict_dataloader(
            smiles_list=smiles, model_hparams=model_hparams
        )

        preds = []
        for batch_idx, batch in enumerate(pred_dl):
            # For certain model prediction pipelines, we may need data module hyperparameters
            preds.append(model.predict_step(batch, batch_idx, dm_hparams=dm_hparams))

        if not save_to:
            # If no save path is provided, return the predictions tensor
            return torch.cat(preds)

        predictions_df = pd.DataFrame(torch.cat(preds).detach().cpu().numpy())

        def _add_class_columns(class_file_path: _PATH):
            with open(class_file_path, "r") as f:
                predictions_df.columns = [cls.strip() for cls in f.readlines()]

        if classes_path is not None:
            _add_class_columns(classes_path)
        elif os.path.exists(dm.classes_txt_file_path):
            _add_class_columns(dm.classes_txt_file_path)

        predictions_df.index = smiles
        predictions_df.to_csv(save_to)


if __name__ == "__main__":
    # python chebai/result/prediction.py  predict_from_file --help
    CLI(Predictor, as_positional=False)
