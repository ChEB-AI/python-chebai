import json
import os

import torch
from lightning.pytorch.callbacks import BasePredictionWriter


class ChebaiPredictionWriter(BasePredictionWriter):
    def __init__(self, output_dir, write_interval, target_file="predictions.json"):
        super().__init__(write_interval)
        self.output_dir = output_dir
        self.target_file = target_file

    def write_on_batch_end(
        self,
        trainer,
        pl_module,
        prediction,
        batch_indices,
        batch,
        batch_idx,
        dataloader_idx,
    ):
        outpath = os.path.join(self.output_dir, dataloader_idx, f"{batch_idx}.pt")
        os.makedirs(outpath, exist_ok=True)
        torch.save(prediction, outpath)

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        pred_list = []
        for p in predictions:
            idents = p["data"]["idents"]
            labels = p["data"]["labels"]
            if labels is not None:
                labels = labels.tolist()
            else:
                labels = [None for _ in idents]
            output = torch.sigmoid(p["output"]["logits"]).tolist()
            for i, l, p in zip(idents, labels, output):
                pred_list.append(dict(ident=i, labels=l, predictions=p))
        with open(os.path.join(self.output_dir, self.target_file), "wt") as fout:
            json.dump(pred_list, fout)
