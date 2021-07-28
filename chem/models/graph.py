from torch import nn
import torch
import torch.nn.functional as F
from torch_geometric import nn as tgnn
from torch_scatter import scatter_mean
from chem.data import JCIExtendedGraphData, JCIGraphData
import logging
import sys

from chem.models.base import JCIBaseNet

logging.getLogger('pysmiles').setLevel(logging.CRITICAL)


class JCIGraphNet(JCIBaseNet):
    NAME = "GNN"

    def __init__(self, in_length, hidden_length, num_classes, **kwargs):
        super().__init__(num_classes, **kwargs)
        self.embedding = torch.nn.Embedding(800, in_length)

        self.conv1 = tgnn.GraphConv(in_length, in_length)
        self.conv2 = tgnn.GraphConv(in_length, in_length)
        self.conv3 = tgnn.GraphConv(in_length, hidden_length)

        self.output_net = nn.Sequential(nn.Linear(hidden_length,hidden_length), nn.ELU(), nn.Linear(hidden_length,hidden_length), nn.ELU(), nn.Linear(hidden_length, num_classes))

        self.dropout = nn.Dropout(0.1)


    def forward(self, x):
        a = self.embedding(x.x)
        a = self.dropout(a)
        a = F.elu(self.conv1(a, x.edge_index.long()))
        a = F.elu(self.conv2(a, x.edge_index.long()))
        a = F.elu(self.conv3(a, x.edge_index.long()))
        a = self.dropout(a)
        a = scatter_mean(a, x.batch, dim=0)
        return self.output_net(a)

class JCIGraphAttentionNet(JCIBaseNet):
    NAME = "AGNN"

    def __init__(self, in_length, hidden_length, num_classes, **kwargs):
        super().__init__(num_classes, **kwargs)
        self.conv1 = tgnn.GATConv(in_length, in_length, 5, concat=False)
        self.conv2 = tgnn.GATConv(in_length, in_length, 5, concat=False)
        self.conv3 = tgnn.GATConv(in_length, hidden_length, 5, concat=False)
        self.output_net = nn.Sequential(nn.Linear(hidden_length, hidden_length),
                                        nn.ELU(),
                                        nn.Linear(hidden_length, hidden_length),
                                        nn.ELU(),
                                        nn.Linear(hidden_length, num_classes))
        self.embedding = torch.nn.Embedding(800, in_length)
        self.dropout = nn.Dropout(0.1)
        self.edge_embedding = torch.nn.Embedding(4, in_length)

    def forward(self, batch):
        a = self.embedding(batch.x)
        a = self.dropout(a)
        a = F.elu(self.conv1(a, batch.edge_index.long()))
        a = F.elu(self.conv2(a, batch.edge_index.long()))
        a = F.elu(self.conv3(a, batch.edge_index.long()))
        a = self.dropout(a)
        a = scatter_mean(a, batch.batch, dim=0)
        return self.output_net(a)


