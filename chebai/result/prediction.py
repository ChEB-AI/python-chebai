from typing import List, Optional

import pandas as pd
import torch
from jsonargparse import CLI
from lightning.fabric.utilities.types import _PATH
from lightning.pytorch.cli import instantiate_module

from chebai.models.base import ChebaiBaseNet
from chebai.preprocessing.datasets.base import XYBaseDataModule


class Predictor:
    def __init__(
        self,
        checkpoint_path: _PATH,
        batch_size: Optional[int] = None,
        compile_model: bool = True,
    ):
        """Initializes the Predictor with a model loaded from the checkpoint.

        Args:
            checkpoint_path: Path to the model checkpoint.
            batch_size: Optional batch size for the DataLoader. If not provided,
                the default from the datamodule will be used.
            compile_model: Whether to compile the model using torch.compile. Default is True.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt_file = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )
        assert (
            "_class_path" in ckpt_file["datamodule_hyper_parameters"]
            and "_class_path" in ckpt_file["hyper_parameters"]
        ), (
            "Datamodule and Model hyperparameters must include a '_class_path' key.\n"
            "Hence, either the checkpoint is corrupted or "
            "it was not saved properly with latest lightning version"
        )

        print("-" * 50)
        print(f"Using device: {self.device}")
        print(f"For Loaded checkpoint from: {checkpoint_path}")
        print("Below are the modules loaded from the checkpoint:")

        self._dm_hparams = ckpt_file["datamodule_hyper_parameters"]
        self._dm_hparams.pop("splits_file_path")
        self._dm_hparams.pop("augment_smiles", None)
        self._dm_hparams.pop("aug_smiles_variations", None)
        self._dm_hparams.pop("_instantiator", None)
        self._dm: XYBaseDataModule = instantiate_module(
            XYBaseDataModule, self._dm_hparams
        )
        if batch_size is not None and int(batch_size) > 0:
            self._dm.batch_size = int(batch_size)
        print("*" * 10, f"Loaded datamodule class: {self._dm.__class__.__name__}")

        self._model_hparams = ckpt_file["hyper_parameters"]
        self._model_hparams.pop("_instantiator", None)
        self._model_hparams.pop("classes_txt_file_path", None)
        self._model: ChebaiBaseNet = instantiate_module(
            ChebaiBaseNet, self._model_hparams
        )
        self._model.to(self.device)
        print("*" * 10, f"Loaded model class: {self._model.__class__.__name__}")

        self._classification_labels: list = ckpt_file.get("classification_labels")
        print(f"Loaded {len(self._classification_labels)} classification labels.")
        assert len(self._classification_labels) > 0, (
            "Classification labels list is empty."
        )
        assert len(self._classification_labels) == self._model.out_dim, (
            f"Number of class labels ({len(self._classification_labels)}) does not match "
            f"the model output dimension ({self._model.out_dim})."
        )

        if compile_model:
            self._model = torch.compile(self._model)  # type: ignore
        self._model.eval()
        print("-" * 50)

    def predict_from_file(
        self,
        smiles_file_path: _PATH,
        save_to: _PATH = "predictions.csv",
    ) -> None:
        """
        Loads a model from a checkpoint and makes predictions on input data from a file.

        Args:
            smiles_file_path: Path to the input file containing SMILES strings.
            save_to: Path to save the predictions CSV file.
        """
        with open(smiles_file_path, "r") as input:
            smiles_strings = [inp.strip() for inp in input.readlines()]

        preds: list[torch.Tensor | None] = self.predict_smiles(smiles=smiles_strings)
        if all(pred is None for pred in preds):
            print("No valid predictions were made. (All predictions are None.)")
            return

        num_of_cols = len(self._classification_labels)
        rows = [
            pred.tolist() if pred is not None else [None] * num_of_cols
            for pred in preds
        ]
        predictions_df = pd.DataFrame(
            rows, columns=self._classification_labels, index=smiles_strings
        )

        predictions_df.to_csv(save_to)
        print(f"Predictions saved to: {save_to}")

    @torch.inference_mode()
    def predict_smiles(
        self,
        smiles: List[str],
    ) -> list[torch.Tensor | None]:
        """
        Predicts the output for a list of SMILES strings using the model.

        Args:
            smiles: A list of SMILES strings.

        Returns:
            A tensor containing the predictions.
        """
        # For certain data prediction piplines, we may need model hyperparameters
        pred_dl, valid_indices = self._dm.predict_dataloader(
            smiles_list=smiles, model_hparams=self._model_hparams
        )

        preds = []
        for batch_idx, batch in enumerate(pred_dl):
            # For certain model prediction pipelines, we may need data module hyperparameters
            result = self._model.predict_step(
                batch, batch_idx, dm_hparams=self._dm_hparams
            )
            preds.append(result["preds"])
        preds = torch.cat(preds)

        # Initialize output with None
        output: list[torch.Tensor | None] = [None] * len(smiles)

        # Scatter predictions back
        for pred, idx in zip(preds, valid_indices):
            output[idx] = pred

        return output


class MainPredictor:
    @staticmethod
    def predict_from_file(
        checkpoint_path: _PATH,
        smiles_file_path: _PATH,
        save_to: _PATH = "predictions.csv",
        batch_size: Optional[int] = None,
    ) -> None:
        predictor = Predictor(checkpoint_path, batch_size)
        predictor.predict_from_file(
            smiles_file_path,
            save_to,
        )

    @staticmethod
    def predict_smiles(
        checkpoint_path: _PATH,
        smiles: List[str],
        batch_size: Optional[int] = None,
    ) -> list[torch.Tensor | None]:
        predictor = Predictor(checkpoint_path, batch_size)
        return predictor.predict_smiles(smiles)


if __name__ == "__main__":
    # python chebai/result/prediction.py  predict_from_file --help
    # python chebai/result/prediction.py  predict_smiles --help
    CLI(MainPredictor, as_positional=False)
