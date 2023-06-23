from lightning.pytorch.callbacks import BasePredictionWriter
import torch
import os

class ChebaiPredictionWriter(BasePredictionWriter):
    def __init__(self, output_dir, write_interval):
        super().__init__(write_interval)
        self.output_dir = output_dir

    def write_on_batch_end(
            self, trainer,
            pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx

    ):
        outpath = os.path.join(self.output_dir, dataloader_idx, f"{batch_idx}.pt")
        os.makedirs(outpath, exist_ok=True)
        torch.save(prediction, outpath)


    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        torch.save(predictions, os.path.join(self.output_dir, "predictions.pt"))
