from torch import nn
import torch
import torch.nn.functional as F
from torch_geometric import nn as tgnn
from torch_scatter import scatter_mean
from chem.data import JCIExtendedGraphData
import logging
import sys

from base import JCIBaseNet

logging.getLogger('pysmiles').setLevel(logging.CRITICAL)


class JCIGraphNet(JCIBaseNet):

    def __init__(self, in_length, hidden_length, num_classes, weights):
        super().__init__(num_classes, weights)
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


if __name__ == "__main__":
    data = JCIExtendedGraphData(int(sys.argv[1]))
    JCIGraphNet.run(data, "graph", model_args=[100, 100, 500])

