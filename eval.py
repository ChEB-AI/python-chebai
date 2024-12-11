import pandas as pd

from chebai.result.utils import (
    evaluate_model,
    load_results_from_buffer,
)
from chebai.result.classification import print_metrics
from chebai.models.electra import Electra
from chebai.preprocessing.datasets.chebi import ChEBIOver50, ChEBIOver100
import os
import tqdm
import torch
import pickle

DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(DEVICE)


# Specify paths and parameters
checkpoint_name = "best_epoch=09_val_loss=0.0217_val_macro-f1=0.7101_val_micro-f1=0.9091"
print("checkpoint_name : ",checkpoint_name)
checkpoint_path = os.path.join("logs/wandb/run-20241207_192102-nug9ndqi/files/checkpoints", f"{checkpoint_name}.ckpt")
print("checkpoint_path : ",checkpoint_path)
kind = "test"  # Change to "train" or "validation" as needed
buffer_dir = os.path.join("results_buffer", checkpoint_name, kind)
print("buffer_dir : ",buffer_dir)
batch_size = 10  # Set batch size

# Load data module
data_module = ChEBIOver100(chebi_version=231)

data_module.splits_file_path="data/chebi_v231/ChEBI100/processed/augmented_splits.csv"
model_class = Electra

# evaluates model, stores results in buffer_dir
model = model_class.load_from_checkpoint(checkpoint_path)
if buffer_dir is None:
    preds, labels = evaluate_model(
        model,
        data_module,
        buffer_dir=buffer_dir,
        # No need to provide this parameter for Chebi dataset, "kind" parameter should be provided
        # filename=data_module.processed_file_names_dict[kind],
        batch_size=10,
        kind=kind,
    )
else:
    evaluate_model(
        model,
        data_module,
        buffer_dir=buffer_dir,
        # No need to provide this parameter for Chebi dataset, "kind" parameter should be provided
        # filename=data_module.processed_file_names_dict[kind],
        batch_size=10,
        kind=kind,
    )
    # load data from buffer_dir
    preds, labels = load_results_from_buffer(buffer_dir, device=DEVICE)


# Load classes from the classes.txt
with open(os.path.join(data_module.processed_dir_main, "classes.txt"), "r") as f:
    classes = [line.strip() for line in f.readlines()]


# output relevant metrics
print_metrics(
    preds,
    labels.to(torch.int),
    DEVICE,
    classes=classes,
    markdown_output=False,
    top_k=10,
)
