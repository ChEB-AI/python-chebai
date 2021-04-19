import os
import sys
import pickle

import networkx as nx
from torch import nn
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from data import JCIPureData
from treelib import Node, Tree
import pysmiles as ps

import logging

logging.getLogger('pysmiles').setLevel(logging.CRITICAL)


class GraphCYK(pl.LightningModule):

    def __init__(self, dim, out):
        super().__init__()
        self.softmax = nn.Softmax()
        self.merge = nn.Linear(2 * dim, dim)
        self.embedding = nn.Embedding(800, dim)
        self.attention_weights = nn.Linear(dim, dim)
        self.output = nn.Linear(dim, out)

    def _execute(self, batch, batch_idx):
        loss = 0
        f1 = 0
        for d, label_bool in batch:
            pred = self(d)
            loss += F.binary_cross_entropy_with_logits(pred, label_bool.float())
            f1 += f1_score(label_bool, torch.sigmoid(pred) > 0.5)
        return loss, f1

    def training_step(self, *args, **kwargs):
        loss, f1 = self._execute(*args, **kwargs)
        self.log('train_loss', loss.detach().item(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_f1', f1.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, *args, **kwargs):
        with torch.no_grad():
            loss, f1 = self._execute(*args, **kwargs)
            self.log('val_loss', loss.detach().item(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log('val_f1', f1.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
            return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer

    def forward(self, graph):
        return self.output(self.step(graph))

    def step(self, graph):
        clusters = list(self.get_all_subgraphs(graph))
        nx.subgraph(graph)
        return list(outputs[-1])[0][1]

    def weight_merge(self, x):
        return x * self.softmax(self.attention_weights(x))

    @staticmethod
    def get_all_subgraphs(graph: nx.Graph):
        clusters = dict()
        for n in graph.nodes:
            t = nx.Graph()
            t.add_node(n, x=graph.nodes[n]["x"])
            for x in _extend(t, graph, n, {n}, [t]):
                yield x


def _extend(tree: nx.Graph, graph: nx.Graph, max_source_index, opens:set, parents: list):
    hidden_parents = {}
    for to_extend in opens:
        for neigh in graph.neighbors(to_extend):
            if neigh not in tree.nodes:
                t2 = tree.copy()
                t2.add_node(neigh, x=graph.nodes[neigh]["x"])
                t2.add_edge(to_extend, neigh)
                new_opens = opens.copy()
                new_opens.remove(to_extend)
                new_opens.add(neigh)
                new_parents = parents + [t2]
                parent_map = (tree, nx.weisfeiler_lehman_graph_hash(t2, node_attr="x"))
                if ((neigh == max_source_index and graph.nodes[neigh]["x"] > graph.nodes[to_extend]["x"]) or neigh > max_source_index):
                    yield t2, parent_map
                    for x in _extend(t2, graph, max(max_source_index, to_extend), new_opens, new_parents):
                        yield x
                    else:
                        yield None, parent_map


if __name__ == "__main__":
    data = JCIPureData()
    data.prepare_data()
    if torch.cuda.is_available():
        trainer_kwargs = dict(gpus=-1, accelerator="ddp")
    else:
        trainer_kwargs = dict(gpus=0)
    net = GraphCYK(100, 500)
    tb_logger = pl_loggers.CSVLogger('logs/')
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(tb_logger.log_dir, "checkpoints"),
        filename="{epoch}-{step}-{val_loss:.7f}",
        save_top_k=5,
        save_last=True,
        verbose=True,
        monitor='val_loss',
        mode='min'
    )

    trainer = pl.Trainer(logger=tb_logger, callbacks=[checkpoint_callback], replace_sampler_ddp=False, **trainer_kwargs)
    trainer.fit(net, data.train_dataloader(), val_dataloaders=data.val_dataloader())
