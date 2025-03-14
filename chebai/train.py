import csv
import multiprocessing as mp
import os
import pickle
import random
from typing import List, Tuple

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from model import ChEBIRecNN
from molecule import Molecule
from pytorch_lightning import loggers as pl_loggers
from sklearn.metrics import f1_score
from torch.utils import data

BATCH_SIZE = 100
NUM_EPOCHS = 100
LEARNING_RATE = 0.01


def eval_model(
    model: nn.Module, dataset: data.DataLoader, test_labels: List[torch.Tensor]
) -> Tuple[List[torch.Tensor], List[List[float]], List[float], float]:
    """
    Evaluate the model on the provided dataset.

    Args:
    - model (nn.Module): The neural network model.
    - dataset (data.DataLoader): DataLoader containing the evaluation dataset.
    - test_labels (List[torch.Tensor]): List of ground truth labels for the evaluation dataset.

    Returns:
    - raw_values (List[torch.Tensor]): List of raw output values from the model.
    - predictions (List[List[float]]): List of binary predictions (0 or 1) from the model.
    - final_scores (List[float]): List of F1 scores computed for each prediction.
    - avg_f1 (float): Average F1 score over all predictions.
    """
    raw_values = []
    predictions = []
    final_scores = []

    with torch.no_grad():
        for batch in dataset:
            for molecule, label in batch:
                model_outputs = model(molecule)
                prediction = [1.0 if i > 0.5 else 0.0 for i in model_outputs]
                predictions.append(prediction)
                raw_values.append(model_outputs)
                final_scores.append(f1_score(prediction, label.tolist()))

        avg_f1 = sum(final_scores) / len(final_scores)
        return raw_values, predictions, final_scores, avg_f1


def crawl_info(
    DAG: List[Tuple[str, str]], sink_parents: List[str]
) -> Tuple[List[int], List[int], dict, int, List[str]]:
    """
    Crawl information from the Directed Acyclic Graph (DAG).

    Args:
    - DAG (List[Tuple[str, str]]): List of tuples representing edges in the DAG.
    - sink_parents (List[str]): List of parent nodes of the sink node.

    Returns:
    - topological_order (List[int]): Nodes in topological order.
    - sources (List[int]): List of source nodes in the DAG.
    - parents (dict): Dictionary mapping nodes to their parent nodes.
    - sink (int): Sink node in the DAG.
    - sink_parents (List[str]): Updated list of parent nodes of the sink node.
    """
    topological_order = [int(i[0]) for i in DAG]
    target_nodes = [int(i[1]) for i in DAG]
    sink = target_nodes[-1]
    sources = []
    parents = {}

    for i in range(len(topological_order)):
        for j in range(len(target_nodes)):
            if topological_order[i] == target_nodes[j]:
                if topological_order[i] not in parents.keys():
                    parents[topological_order[i]] = []
                parents[topological_order[i]].append(topological_order[j])

    for node in topological_order:
        if node not in parents.keys():
            sources.append(node)

    return topological_order, sources, parents, sink, sink_parents


def collate(
    batch: List[Tuple[Molecule, torch.Tensor]],
) -> Tuple[List[Molecule], torch.Tensor]:
    """
    Collate function for DataLoader.

    Args:
    - batch (List[Tuple[Molecule, torch.Tensor]]): List of tuples containing molecules and labels.

    Returns:
    - input (List[Molecule]): List of Molecule objects.
    - labels (torch.Tensor): Tensor of labels.
    """
    input, labels = zip(*batch)
    return input, torch.stack(labels)


def _execute(
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    data: data.DataLoader,
    device: str,
    with_grad: bool = True,
) -> Tuple[float, float]:
    """
    Execute a single training or evaluation step.

    Args:
    - model (nn.Module): The neural network model.
    - loss_fn (nn.Module): Loss function.
    - optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
    - data (data.DataLoader): DataLoader containing the batched data.
    - device (str): Device (CPU or GPU) on which to run the operations.
    - with_grad (bool): Whether to compute gradients. Default is True (for training).

    Returns:
    - train_running_loss (float): Average loss over the data.
    - f1 (float): Average F1 score over the data.
    """
    train_running_loss = 0.0
    data_size = 0
    f1 = 0
    model.train(with_grad)
    num_batches = len(data)
    num_batch = 0

    for molecules, labels in data:
        num_batch += 1
        optimizer.zero_grad()
        prediction = model(molecules)
        loss = loss_fn(prediction, labels)
        data_size += 1
        f1 += f1_score(prediction > 0.5, labels > 0.5, average="micro")
        train_running_loss += loss.item()

        if with_grad:
            print(f"Batch {num_batch}/{num_batches}")
            loss.backward()
            optimizer.step()

    return train_running_loss / data_size, f1 / data_size


def execute_network(
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_data: data.DataLoader,
    validation_data: data.DataLoader,
    epochs: int,
    device: str,
) -> None:
    """
    Execute the training and evaluation loop over multiple epochs.

    Args:
    - model (nn.Module): The neural network model.
    - loss_fn (nn.Module): Loss function.
    - optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
    - train_data (data.DataLoader): DataLoader containing the training dataset.
    - validation_data (data.DataLoader): DataLoader containing the validation dataset.
    - epochs (int): Number of epochs to train the model.
    - device (str): Device (CPU or GPU) on which to run the operations.

    Returns:
    - None
    """
    model.to(device)
    model.device = device

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

    columns_name = [
        "epoch",
        "train_running_loss",
        "train_running_f1",
        "eval_running_loss",
        "eval_running_f1",
    ]
    with open(r"../loss_f1_training_validation.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(columns_name)

    for epoch in range(epochs):
        train_running_loss, train_running_f1 = _execute(
            model, loss_fn, optimizer, train_data, device, with_grad=True
        )

        with torch.no_grad():
            eval_running_loss, eval_running_f1 = _execute(
                model, loss_fn, optimizer, validation_data, device, with_grad=False
            )

        print(
            f"Epoch {epoch}: loss={train_running_loss:.5f}, f1={train_running_f1:.5f}, val_loss={eval_running_loss:.5f}, val_f1={eval_running_f1:.5f}".format(
                epoch, train_running_f1
            )
        )
        fields = [
            epoch,
            train_running_loss,
            train_running_f1,
            eval_running_loss,
            eval_running_f1,
        ]
        with open(r"../loss_f1_training_validation.csv", "a") as f:
            writer = csv.writer(f)
            writer.writerow(fields)


def prepare_data(infile: pickle.Pickler) -> pd.DataFrame:
    """
    Prepare the data from a pickle file.

    Args:
    - infile (pickle.Pickler): Pickle file containing the raw data.

    Returns:
    - train_df (pd.DataFrame): Processed DataFrame containing the data.
    """
    data = pickle.load(infile)
    infile.close()

    data_frame = pd.DataFrame.from_dict(data)
    data_frame.reset_index(drop=True, inplace=True)

    data_classes = list(data_frame.columns)
    data_classes.remove("MOLECULEID")
    data_classes.remove("SMILES")

    for col in data_classes:
        data_frame[col] = data_frame[col].astype(int)

    train_data = []
    for index, row in data_frame.iterrows():
        train_data.append(
            [
                data_frame.iloc[index].values[1],
                data_frame.iloc[index].values[2:502].tolist(),
            ]
        )

    train_df = pd.DataFrame(train_data, columns=["SMILES", "LABELS"])
    return train_df


def batchify(x: List, y: List) -> List:
    """
    Batchify the input data and labels.

    Args:
    - x (List): List of input data.
    - y (List): List of labels.

    Returns:
    - List: Batched data.
    """
    data = list(zip(x, y))
    return [
        data[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
        for i in range(1 + len(data) // BATCH_SIZE)
    ]


def load_data() -> (
    Tuple[List[Molecule], List[torch.Tensor], List[Molecule], List[torch.Tensor]]
):
    """
    Load and preprocess the data.

    Returns:
    - train_dataset (List[Molecule]): List of Molecule objects for training.
    - train_actual_labels (List[torch.Tensor]): List of ground truth labels for training.
    - validation_dataset (List[Molecule]): List of Molecule objects for validation.
    - validation_actual_labels (List[torch.Tensor]): List of ground truth labels for validation.
    """
    fpath = "data/full.pickle"
    if os.path.isfile(fpath):
        with open(fpath, "rb") as f:
            (
                train_dataset,
                train_actual_labels,
                validation_dataset,
                validation_actual_labels,
            ) = pickle.load(f)
    else:
        print("reading data from files!")
        train_infile = open("../data/JCI_graph/raw/train.pkl", "rb")
        test_infile = open("../data/JCI_graph/raw/test.pkl", "rb")
        validation_infile = open("../data/JCI_graph/raw/validation.pkl", "rb")

        # test_data = prepare_data(test_infile)

        print("prepare train data!")
        train_dataset = []
        train_actual_labels = []

        for index, row in prepare_data(train_infile).iterrows():
            try:
                mol = Molecule(row["SMILES"], True)

                DAGs_meta_info = mol.dag_to_node
                train_dataset.append(mol)
                train_actual_labels.append(torch.tensor(row["LABELS"]).float())
            except:
                pass

        print("prepare validation data!")
        validation_dataset = []
        validation_actual_labels = []

        for index, row in prepare_data(validation_infile).iterrows():
            try:
                mol = Molecule(row["SMILES"], True)

                DAGs_meta_info = mol.dag_to_node

                validation_dataset.append(mol)
                validation_actual_labels.append(torch.tensor(row["LABELS"]).float())
            except:
                pass

        with open(fpath, "wb") as f:
            pickle.dump(
                (
                    train_dataset,
                    train_actual_labels,
                    validation_dataset,
                    validation_actual_labels,
                ),
                f,
            )

    return (
        train_dataset,
        train_actual_labels,
        validation_dataset,
        validation_actual_labels,
    )


def move_molecule(m: Molecule) -> Molecule:
    """
    Move a molecule and collect its atom features.

    Args:
    - m (Molecule): The Molecule object to process.

    Returns:
    - Molecule: Processed Molecule object.
    """
    m.collect_atom_features()
    return m


if __name__ == "__main__":
    if torch.cuda.is_available():
        accelerator = "ddp"
        trainer_kwargs = dict(gpus=-1)
    else:
        accelerator = None
        trainer_kwargs = dict()

    (
        train_dataset,
        train_actual_labels,
        validation_dataset,
        validation_actual_labels,
    ) = load_data()
    train_data = data.DataLoader(
        list(
            zip(
                map(move_molecule, train_dataset),
                [l.float() for l in train_actual_labels],
            )
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate,
    )
    validation_data = data.DataLoader(
        list(
            zip(
                map(move_molecule, validation_dataset),
                [l.float() for l in validation_actual_labels],
            )
        ),
        batch_size=BATCH_SIZE,
        collate_fn=collate,
    )

    # Initialize model
    model = ChEBIRecNN()

    # Configure logging
    tb_logger = pl_loggers.CSVLogger("../logs/")
    trainer = pl.Trainer(
        logger=tb_logger,
        accelerator=accelerator,
        max_epochs=NUM_EPOCHS,
        **trainer_kwargs,
    )

    # Train the model
    trainer.fit(model, train_data, val_dataloaders=validation_data)

# Uncomment below section for saving the model and executing network training directly
"""
model = ChEBIRecNN()
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

print('num of parameters of the model: ', sum(p.numel() for p in model.parameters() if p.requires_grad))

params = {
    'model':model,
    'loss_fn':loss_fn,
    'optimizer':optimizer,
    'train_data': train_data,
    'validation_data': validation_data,
    'epochs':NUM_EPOCHS,
    'device': device
}

print('start training!')
execute_network(**params)

torch.save(model.state_dict(), 'ChEBI_RvNN_{}_epochs'.format(NUM_EPOCHS))
print('model saved!')
"""
