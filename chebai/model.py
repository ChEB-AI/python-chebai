from typing import Iterable

from molecule import Molecule
from pytorch_lightning.metrics import F1
import networkx as nx
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChEBIRecNN(pl.LightningModule):
    def __init__(self):
        super(ChEBIRecNN, self).__init__()

        self.atom_enc = 62
        self.length = 200
        self.output_of_sinks = 500
        self.num_of_classes = 500

        self.norm = torch.nn.LayerNorm(self.length)

        self._f1 = F1(500, threshold=0.5)
        self._loss_fun = F.binary_cross_entropy_with_logits

        self.metrics = {"loss": self._loss_fun, "f1": self._f1}

        self.c1 = nn.Linear(self.length, self.length)
        self.c2 = nn.Linear(self.length, self.length)
        self.c3 = nn.Linear(self.length, self.length)
        self.c4 = nn.Linear(self.length, self.length)
        self.c5 = nn.Linear(self.length, self.length)
        self.c = {1: self.c1, 2: self.c2, 3: self.c3, 4: self.c4, 5: self.c5}

        self.NN_single_node = nn.Sequential(
            nn.Linear(self.atom_enc, self.length),
            nn.ReLU(),
            nn.Linear(self.length, self.length),
        )
        self.merge = nn.Sequential(
            nn.Linear(2 * self.length, self.length),
            nn.ReLU(),
            nn.Linear(self.length, self.length),
        )
        self.register_parameter(
            "attention_weight",
            torch.nn.Parameter(torch.rand(self.length, 1, requires_grad=True)),
        )
        self.register_parameter(
            "dag_weight",
            torch.nn.Parameter(torch.rand(self.length, 1, requires_grad=True)),
        )
        self.final = nn.Sequential(
            nn.Linear(self.length, self.length),
            nn.ReLU(),
            nn.Linear(self.length, self.length),
            nn.ReLU(),
            nn.Linear(self.length, self.num_of_classes),
        )

    def forward(self, molecules: Iterable[Molecule]):
        return torch.stack([self._proc_single_mol(molecule) for molecule in molecules])

    def _proc_single_mol(self, molecule):
        final_outputs = None
        # for each DAG, generate a hidden representation at its sink node
        last = None
        for sink, dag in molecule.dag_to_node.items():
            inputs = {}
            num_inputs = {}
            for node in nx.topological_sort(dag):
                atom = self.process_atom(node, molecule)
                if not any(dag.predecessors(node)):
                    output = atom
                else:
                    inp_prev = self.attention(self.attention_weight, inputs[node])
                    inp = torch.cat((inp_prev, atom), dim=0)
                    output = F.relu(self.merge(inp)) + inp_prev
                for succ in dag.successors(node):
                    try:
                        inputs[succ] = torch.cat(
                            (
                                self.c[num_inputs[succ]](inputs[succ]),
                                output.unsqueeze(0),
                            )
                        )
                        num_inputs[succ] += 1
                    except KeyError:
                        inputs[succ] = output.unsqueeze(0)
                        num_inputs[succ] = 1
                last = output
            if final_outputs is not None:
                final_outputs = torch.cat((final_outputs, last.unsqueeze(0)))
            else:
                final_outputs = last.unsqueeze(0)
        # take the average of hidden representation at all sinks
        return self.final(self.attention(self.dag_weight, final_outputs))

    def _calculate_metrics(self, prediction, labels, prefix=""):
        return {m: mf(prediction, labels) for m, mf in self.metrics.items()}

    def training_step(self, batch, batch_idx):
        molecules, labels = batch
        prediction = self(molecules)
        return self._calculate_metrics(prediction, labels)

    def validation_step(self, batch, batch_idx):
        molecules, labels = batch
        prediction = self(molecules)
        return self._calculate_metrics(prediction, labels, prefix="val_")

    def process_atom(self, node, molecule):
        return F.dropout(
            F.relu(
                self.NN_single_node(molecule.get_atom_features(node).to(self.device))
            ),
            p=0.1,
        )

    def training_epoch_end(self, outputs) -> None:
        for metric in self.metrics:
            avg = torch.stack([o[metric] for o in outputs]).mean()
            self.log(metric, avg)

    def validation_epoch_end(self, outputs) -> None:
        if not self.trainer.running_sanity_check:
            for metric in self.metrics:
                avg = torch.stack([o[metric] for o in outputs]).mean()
                self.log("val_" + metric, avg)

    @staticmethod
    def attention(weights, x):
        return torch.sum(
            torch.mul(torch.softmax(torch.matmul(x, weights), dim=0), x), dim=0
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
