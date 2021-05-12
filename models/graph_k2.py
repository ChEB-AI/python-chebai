import os
from torch import nn
import torch
from sklearn.metrics import f1_score
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.metrics import F1
import torch.nn.functional as F
from torch_geometric import nn as tgnn
from torch_scatter import scatter_mean
from torch_geometric.data import DataLoader
from k_gnn import avg_pool
from data import JCIExtendedGraphData, JCIExtendedGraphTwoData

import logging
import sys

from base import JCIBaseNet

logging.getLogger('pysmiles').setLevel(logging.CRITICAL)


class JCIGraphK2Net(JCIBaseNet):

    def __init__(self, in_length, hidden_length, num_classes, weights=None):
        super().__init__(num_classes, weights)
        self.embedding = torch.nn.Embedding(800, in_length)

        self.conv1_1 = tgnn.GraphConv(in_length, in_length)
        self.conv1_2 = tgnn.GraphConv(in_length, in_length)
        self.conv1_3 = tgnn.GraphConv(in_length, hidden_length)

        self.conv2_1 = tgnn.GraphConv(in_length, in_length)
        self.conv2_2 = tgnn.GraphConv(in_length, in_length)
        self.conv2_3 = tgnn.GraphConv(in_length, hidden_length)

        self.output_net = nn.Sequential(nn.Linear(hidden_length*2, hidden_length), nn.ELU(),
                                        nn.Linear(hidden_length, hidden_length), nn.ELU(),
                                        nn.Linear(hidden_length, num_classes))

    def forward(self, x):
        a = self.embedding(x.x)
        a = self.dropout(a)

        a = F.elu(self.conv1_1(a, x.edge_index.long()))
        a = F.elu(self.conv1_2(a, x.edge_index.long()))
        a = F.elu(self.conv1_3(a, x.edge_index.long()))
        a_1 = scatter_mean(a, x.batch, dim=0)

        a = avg_pool(a, x.assignment_index_2)
        a = F.elu(self.conv2_1(a, x.edge_index.long()))
        a = F.elu(self.conv2_2(a, x.edge_index.long()))
        a = F.elu(self.conv2_3(a, x.edge_index.long()))
        a_2 = scatter_mean(a, x.batch_2, dim=0)

        a = torch.cat([a_1, a_2], dim=1)

        a = self.dropout(a)
        return self.output_net(a)


if __name__ == "__main__":
    batch_size = int(sys.argv[1])
    data = JCIExtendedGraphTwoData(batch_size=batch_size)
    JCIGraphK2Net.run(data, "graph_k2", model_args=[100, 100, 500])
