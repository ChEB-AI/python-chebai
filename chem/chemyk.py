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
from torch_geometric.nn import GraphConv
from torch_geometric.utils import from_networkx
from torch_geometric.data.dataloader import Collater
from itertools import chain
import logging

logging.getLogger('pysmiles').setLevel(logging.CRITICAL)


class NormalCYK(pl.LightningModule):

    def __init__(self, d_input, d_output, d_internal=None):
        super().__init__()
        if d_internal is None:
            d_internal = d_output
        self.embedding = nn.Embedding(800, d_internal)
        self.s = nn.Linear(d_internal, 1)
        self.a_l = nn.Linear(d_internal,1)
        self.a_r = nn.Linear(d_internal, 1)
        self.w_l = nn.Linear(d_internal, d_internal)
        self.w_r = nn.Linear(d_internal, d_internal)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer

    def _execute(self, batch, batch_idx):
        d, label_bool = batch
        pred = self.step(d)
        loss = F.binary_cross_entropy_with_logits(pred, label_bool.float())
        f1 = f1_score(label_bool, torch.sigmoid(pred) > 0.5)
        return loss, f1

    def step(self, smiles, *args, **kwargs):
        h = [[self.embedding(s) for s in smiles]]
        for width in range(2, len(smiles)):
            # i + k = j <-> k = j - i
            print(width)
            l = [self.attention(self.merge(torch.stack([torch.stack((h[merge_point-1][left], h[width-merge_point-1][left+merge_point]))
                for merge_point in range(1,width)])))
                    for left in range(len(smiles) - width)]
            h.append(l)
        return h[-1][0]

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

    def merge(self, lr):
        x = torch.stack([self.a_l(lr[:,0]), self.a_r(lr[:,1])])
        beta = torch.softmax(x, 0)
        return torch.sum(beta*torch.stack([self.w_l(lr[:,0]), self.w_r(lr[:,1])]), dim=0)

    def attention(self, parts):
        at = torch.softmax(self.s(parts), 0)
        return torch.sum(at*parts, dim=0)


class GraphCYK(pl.LightningModule):

    def __init__(self, dim, out):
        super().__init__()
        self.softmax = nn.Softmax()
        self.merge = nn.Linear(2 * dim, dim)
        self.embedding = nn.Embedding(800, dim)
        self.attention_weights = nn.Linear(dim, dim)
        self.output = nn.Linear(dim, out)
        self.subgraph_net = GraphConv(dim, dim)


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

        return final_result

    def weight_merge(self, x):
        return x * self.softmax(self.attention_weights(x))

    @staticmethod
    def get_all_subgraphs(graph: nx.Graph):
        for n in graph.nodes:
            yield chain((({n}, None, nx.weisfeiler_lehman_graph_hash(nx.subgraph(graph, {n}), node_attr="x")),), _extend({n}, graph, n, set(graph.neighbors(n))))


def _extend(tree: set, graph: nx.Graph, max_source_index, opens:set):
    for to_extend in opens:
        t2 = tree.copy()
        t2.add(to_extend)
        msi = max(max_source_index, to_extend)
        new_opens = {x for x in opens.union(set(graph.neighbors(to_extend)).difference(t2)) if (x > msi or (x == msi and graph.nodes[x]["x"] > graph.nodes[to_extend]["x"])) and x != to_extend}
        parent_map = (nx.weisfeiler_lehman_graph_hash(nx.subgraph(graph, tree), node_attr="x"),
                      nx.weisfeiler_lehman_graph_hash(nx.subgraph(graph, t2), node_attr="x"))
        yield t2, *parent_map
        for x in _extend(t2, graph, msi, new_opens):
            yield x


class ChemLSTM(pl.LightningModule):

    def __init__(self, in_d, out_d):
        super().__init__()
        self.lstm = nn.LSTM(in_d, out_d, batch_first=True)
        self.embedding = nn.Embedding(800, in_d)
        self.output = nn.Sequential(nn.Linear(out_d, out_d), nn.ReLU(), nn.Linear(out_d, out_d))

    def _execute(self, batch, batch_idx):
        loss = 0
        f1 = 0
        x, y = batch
        pred = self(x)
        loss += F.binary_cross_entropy_with_logits(pred, y.float())
        f1 += f1_score(y, torch.sigmoid(pred) > 0.5, average="micro")
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

    def forward(self, x):
        x = self.embedding(x)
        x = self.lstm(x)[1][0]
        x = self.output(x)
        return x.squeeze(0)

def run_graphyk():
    data = JCIPureData()
    data.prepare_data()
    if torch.cuda.is_available():
        trainer_kwargs = dict(gpus=-1, accelerator="ddp")
    else:
        trainer_kwargs = dict(gpus=0)
    net = GraphCYK(100, 500)
    tb_logger = pl_loggers.CSVLogger('../logs/')
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


def run_cyk():
    data = JCIExtendedData(batch_size=100)
    data.prepare_data()
    data.setup()
    train_data = data.train_dataloader()
    val_data = data.val_dataloader()
    if torch.cuda.is_available():
        trainer_kwargs = dict(gpus=-1, accelerator="ddp")
    else:
        trainer_kwargs = dict(gpus=0)
    net = ChemLSTM(100, 500)
    tb_logger = pl_loggers.CSVLogger('../logs/')
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
    trainer.fit(net, train_data, val_dataloaders=val_data)

if __name__ == "__main__":
    run_cyk()
