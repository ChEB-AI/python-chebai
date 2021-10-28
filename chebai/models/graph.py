import logging
import sys

from torch import nn
from torch_geometric import nn as tgnn
from torch_scatter import scatter_add, scatter_max, scatter_mean
import torch
import torch.nn.functional as F

from chebai.models.base import JCIBaseNet

logging.getLogger("pysmiles").setLevel(logging.CRITICAL)


class JCIGraphNet(JCIBaseNet):
    NAME = "GNN"

    def __init__(self, in_length, hidden_length, num_classes, **kwargs):
        super().__init__(num_classes, **kwargs)
        self.embedding = torch.nn.Embedding(800, in_length)

        self.conv1 = tgnn.GraphConv(in_length, in_length)
        self.conv2 = tgnn.GraphConv(in_length, in_length)
        self.conv3 = tgnn.GraphConv(in_length, hidden_length)

        self.output_net = nn.Sequential(
            nn.Linear(hidden_length, hidden_length),
            nn.ELU(),
            nn.Linear(hidden_length, hidden_length),
            nn.ELU(),
            nn.Linear(hidden_length, num_classes),
        )

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        a = self.embedding(x.x)
        a = self.dropout(a)
        a = F.elu(self.conv1(a, x.edge_index.long()))
        a = F.elu(self.conv2(a, x.edge_index.long()))
        a = F.elu(self.conv3(a, x.edge_index.long()))
        a = self.dropout(a)
        a = scatter_add(a, x.batch, dim=0)
        return self.output_net(a)


class JCIGraphAttentionNet(JCIBaseNet):
    NAME = "AGNN"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        in_length = kwargs.get("in_length")
        hidden_length = kwargs.get("hidden_length")
        self.embedding = torch.nn.Embedding(800, in_length)
        self.edge_embedding = torch.nn.Embedding(4, in_length)
        in_length = in_length + 10
        self.conv1 = tgnn.GATConv(
            in_length, in_length, 5, concat=False, dropout=0.1, add_self_loops=True
        )
        self.conv2 = tgnn.GATConv(
            in_length, in_length, 5, concat=False, add_self_loops=True
        )
        self.conv3 = tgnn.GATConv(
            in_length, in_length, 5, concat=False, add_self_loops=True
        )
        self.conv4 = tgnn.GATConv(
            in_length, in_length, 5, concat=False, add_self_loops=True
        )
        self.conv5 = tgnn.GATConv(
            in_length, in_length, 5, concat=False, add_self_loops=True
        )
        self.output_net = nn.Sequential(
            nn.Linear(in_length, hidden_length),
            nn.LeakyReLU(),
            nn.Linear(hidden_length, hidden_length),
            nn.LeakyReLU(),
            nn.Linear(hidden_length, 500),
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, batch):
        a = self.embedding(batch.x)
        a = self.dropout(a)
        a = torch.cat([a, torch.rand((*a.shape[:-1], 10)).to(self.device)], dim=1)
        a = F.leaky_relu(self.conv1(a, batch.edge_index.long()))
        a = F.leaky_relu(self.conv2(a, batch.edge_index.long()))
        a = F.leaky_relu(self.conv3(a, batch.edge_index.long()))
        a = F.leaky_relu(self.conv4(a, batch.edge_index.long()))
        a = F.leaky_relu(self.conv5(a, batch.edge_index.long()))
        a = self.dropout(a)
        a = scatter_mean(a, batch.batch, dim=0)
        a = self.output_net(a)
        return a
