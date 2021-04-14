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
        clusters = self.get_all_subgraphs(graph)
        outputs = [{(c, self.embedding(torch.tensor(graph.nodes[n]["x"]))) for c in next(clusters) for n in c}]
        for layer in clusters:
            outputs.append(self.merge_clusters(layer, outputs))
        return list(outputs[-1])[0][1]

    def merge_clusters(self, layer, output_list):
        return {(cluster, sum(self.merge(torch.cat((o1, o2))) for i in range(len(cluster) // 2)
                              for c1, o1 in output_list[i]
                              for c2, o2 in output_list[-i - 1]
                              if c1.union(c2) == cluster)) for cluster in layer}

    def weight_merge(self, x):
        return x * self.softmax(self.attention_weights(x))

    @staticmethod
    def get_all_subgraphs(graph: nx.Graph):
        clusters = dict()
        for n in graph.nodes:
            g2 = graph.subgraph((n,))
            hsh = nx.weisfeiler_lehman_graph_hash(g2, node_attr="x")
            clusters[hsh] = clusters.get(hsh, []) + [(g2,[])]
        new_clusters = clusters
        while new_clusters:
            new_clusters = {}
            for lhash, lgraphs in clusters.items():
                for lhash, rgraphs in clusters.items():
                    for l, _ in lgraphs:
                        for r, _ in rgraphs:
                            if l != r:
                                g2 = graph.subgraph(set(l.nodes).union(set(r.nodes)))
                                if nx.is_connected(g2):
                                    hsh = nx.weisfeiler_lehman_graph_hash(g2, node_attr="x")
                                    new_clusters[hsh] = new_clusters.get(hsh, []) + [(g2, [l, r])]
            clusters.update(new_clusters)
        return clusters




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
