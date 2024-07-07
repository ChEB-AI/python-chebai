import os
import pickle
from typing import Any, Literal, Sequence

import torch
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import BasePredictionWriter


class PredictionWriter(BasePredictionWriter):
    """
    Custom callback for writing predictions to a file at the end of each epoch.

    Args:
        output_dir (str): The directory where prediction files will be saved.
        write_interval (str): When to write predictions. Options are "batch" or "epoch".
    """

    def __init__(
        self,
        output_dir: str,
        write_interval: Literal["batch", "epoch", "batch_and_epoch"],
    ):
        super().__init__(write_interval)
        self.output_dir = output_dir
        self.prediction_file_name = "predictions.pkl"

    def write_on_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        predictions: Sequence[Any],
        batch_indices: Sequence[Any],
    ) -> None:
        """
        Writes the predictions to a file at the end of the epoch.

        Args:
            trainer (Trainer): The Trainer instance.
            pl_module (LightningModule): The LightningModule instance.
            predictions (Sequence[Any]): Any sequence of predictions for the epoch.
            batch_indices (Sequence[Any]): Any sequence of batch indices.
        """
        results = [
            dict(
                ident=row["data"]["idents"][0],
                predictions=torch.sigmoid(row["output"]["logits"]).numpy(),
                labels=row["labels"][0].numpy() if row["labels"] is not None else None,
            )
            for row in predictions
        ]
        with open(
            os.path.join(self.output_dir, self.prediction_file_name), "wb"
        ) as fout:
            pickle.dump(results, fout)
