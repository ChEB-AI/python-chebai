import os
import pickle

import torch
from lightning.pytorch.callbacks import BasePredictionWriter


class PredictionWriter(BasePredictionWriter):
    def __init__(self, output_dir, write_interval):
        super().__init__(write_interval)
        self.output_dir = output_dir
        self.prediction_file_name = "predictions.pkl"

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
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
