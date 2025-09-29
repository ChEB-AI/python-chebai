import json
import os
from typing import Any, Dict, List, Literal, Union

import torch
from lightning.pytorch.callbacks import BasePredictionWriter


class ChebaiPredictionWriter(BasePredictionWriter):
    """
    A custom prediction writer for saving batch and epoch predictions during model training.

    This class inherits from `BasePredictionWriter` and is designed to save predictions
    in a specified output directory at specified intervals.

    Args:
        output_dir (str): The directory where predictions will be saved.
        write_interval (str): The interval at which predictions will be written.
        target_file (str): The name of the file where epoch predictions will be saved (default: "predictions.json").
    """

    def __init__(
        self,
        output_dir: str,
        write_interval: Literal["batch", "epoch", "batch_and_epoch"],
        target_file: str = "predictions.json",
    ) -> None:
        super().__init__(write_interval)
        self.output_dir = output_dir
        self.target_file = target_file

    def write_on_batch_end(
        self,
        trainer: Any,
        pl_module: Any,
        prediction: Union[torch.Tensor, List[torch.Tensor]],
        batch_indices: List[int],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """
        Saves batch predictions at the end of each batch.

        Args:
            trainer (Any): The trainer instance.
            pl_module (Any): The LightningModule instance.
            prediction (Union[torch.Tensor, List[torch.Tensor]]): The prediction output from the model.
            batch_indices (List[int]): The indices of the batch.
            batch (Any): The current batch.
            batch_idx (int): The index of the batch.
            dataloader_idx (int): The index of the dataloader.
        """
        outpath = os.path.join(self.output_dir, str(dataloader_idx), f"{batch_idx}.pt")
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        torch.save(prediction, outpath)

    def write_on_epoch_end(
        self,
        trainer: Any,
        pl_module: Any,
        predictions: List[Dict[str, Any]],
        batch_indices: List[int],
    ) -> None:
        """
        Saves all predictions at the end of each epoch in a JSON file.

        Args:
            trainer (Any): The trainer instance.
            pl_module (Any): The LightningModule instance.
            predictions (List[Dict[str, Any]]): The list of prediction outputs from the model.
            batch_indices (List[int]): The indices of the batches.
        """
        pred_list = []
        for p in predictions:
            idents = p["data"]["idents"]
            labels = p["data"]["labels"]
            if labels is not None:
                labels = labels.tolist()
            else:
                labels = [None for _ in idents]
            output = torch.sigmoid(p["output"]["logits"]).tolist()
            for i, l, o in zip(idents, labels, output):
                pred_list.append(dict(ident=i, labels=l, predictions=o))
        with open(os.path.join(self.output_dir, self.target_file), "wt") as fout:
            json.dump(pred_list, fout)
