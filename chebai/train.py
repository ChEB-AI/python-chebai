import csv
import multiprocessing as mp
import os
import pickle

from model import ChEBIRecNN
from molecule import Molecule
from pytorch_lightning import loggers as pl_loggers
from sklearn.metrics import f1_score
from torch.utils import data
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn

BATCH_SIZE = 100
NUM_EPOCHS = 100
LEARNING_RATE = 0.01


def eval_model(model, dataset, test_labels):
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


def crawl_info(DAG, sink_parents):
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


import random


def collate(batch):
    input, labels = zip(*batch)
    return input, torch.stack(labels)


def _execute(model, loss_fn, optimizer, data, device, with_grad=True):
    train_running_loss = 0
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
    model, loss_fn, optimizer, train_data, validation_data, epochs, device
):
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


def prepare_data(infile):
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


def batchify(x, y):
    data = list(zip(x, y))
    return [
        data[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
        for i in range(1 + len(data) // BATCH_SIZE)
    ]


def load_data():
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


def move_molecule(m):
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

    model = ChEBIRecNN()

    tb_logger = pl_loggers.CSVLogger("../logs/")
    trainer = pl.Trainer(
        logger=tb_logger,
        accelerator=accelerator,
        max_epochs=NUM_EPOCHS,
        **trainer_kwargs,
    )
    trainer.fit(model, train_data, val_dataloaders=validation_data)

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
