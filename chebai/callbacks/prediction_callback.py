from lightning.pytorch.callbacks import BasePredictionWriter
import torch
import os


class PredictionWriter(BasePredictionWriter):
    def __init__(self, output_dir, write_interval):
        super().__init__(write_interval)
        self.output_dir = output_dir
        self.prediction_file_name = "predictions.pt"

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        torch.save(predictions, os.path.join(self.output_dir, self.prediction_file_name))
