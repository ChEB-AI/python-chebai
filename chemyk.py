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
from torch_geometric.nn import GraphConv
from torch_geometric.utils import from_networkx
from torch_geometric.data.dataloader import Collater
from itertools import chain
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
        self.subgraph_net = GraphConv(dim, out)


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
        final_result = 0
        c = Collater(follow_batch=["x", "edge_index", "label"])
        g = nx.DiGraph()
        data = {}
        for execution_tree in self.get_all_subgraphs(graph):
            for (tree, parent_hash,  hsh) in execution_tree:
                if tree:
                    g.add_node(hsh, tree=tree)
                if parent_hash:
                    g.add_edge(parent_hash, hsh)
                else:
                    assert root is None, f"Multiple roots found: {root}, {hsh}"
                    root = hsh

            inv_levels = nx.single_source_shortest_path_length(g, root)
            levels = [[] for i in set(inv_levels.values())]
            for x, i in inv_levels.items():
                levels[i].append(x)

            for tree_ids in levels:
                y = []
                for tid in tree_ids:
                    sg = nx.subgraph(graph, g.nodes[tid]["tree"].nodes)
                    y.append(sg)
                y = c(map(from_networkx, y))
                x = torch.sum(self.subgraph_net(x=y.x_batch, edge_index=y.edge_index_batch))
                for pred in g.predecessors(tree_id):
                    x += data[pred]
                data[tree_id] = x
            final_result += data[root]
        return final_result

    def weight_merge(self, x):
        return x * self.softmax(self.attention_weights(x))

    @staticmethod
    def get_all_subgraphs(graph: nx.Graph):
        for n in graph.nodes:
            t = nx.Graph()
            t.add_node(n, x=graph.nodes[n]["x"])
            yield chain(((t, None, nx.weisfeiler_lehman_graph_hash(t, node_attr="x")),), _extend(t, graph, n, {n}))


def _extend(tree: nx.Graph, graph: nx.Graph, max_source_index, opens:set):
    for to_extend in opens:
        for neigh in graph.neighbors(to_extend):
            if neigh not in tree.nodes:
                t2 = tree.copy()
                t2.add_node(neigh, x=graph.nodes[neigh]["x"])
                t2.add_edge(to_extend, neigh)
                new_opens = opens.copy()
                #new_opens.remove(to_extend)
                new_opens.add(neigh)
                parent_map = (nx.weisfeiler_lehman_graph_hash(tree, node_attr="x"), nx.weisfeiler_lehman_graph_hash(t2, node_attr="x"))
                if ((neigh == max_source_index and graph.nodes[neigh]["x"] > graph.nodes[to_extend]["x"]) or neigh > max_source_index):
                    yield t2, *parent_map
                    for x in _extend(t2, graph, max(max_source_index, to_extend), new_opens):
                        yield x
                else:
                    yield None, *parent_map


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
