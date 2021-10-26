from itertools import combinations
import logging
import sys

from torch import exp, nn, tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import networkx as nx
import torch

from chebai.models.base import JCIBaseNet

logging.getLogger("pysmiles").setLevel(logging.CRITICAL)


class ChemYK(JCIBaseNet):
    NAME = "ChemYK"

    def __init__(self, in_d, out_d, num_classes, **kwargs):
        super().__init__(num_classes, **kwargs)
        self.embedding = nn.Embedding(800, 100)
        self.left = nn.Linear(in_d, in_d)
        self.right = nn.Linear(in_d, in_d)
        self.w_l = nn.Linear(in_d, in_d)
        self.w_r = nn.Linear(in_d, in_d)
        self.ff_rep = nn.Linear(in_d, in_d)
        self.softmax = nn.Softmax()
        self.attention_weight = nn.Linear(in_d, in_d)
        self.top_level_attention_weight = nn.Linear(in_d, in_d)
        self.output = nn.Sequential(
            nn.Linear(in_d, in_d),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_d, num_classes),
        )

    def forward(self, batch, max_width=5):
        result = []
        for data in batch.x:
            # Calculate embeddings
            clusters = [
                (
                    frozenset({x, y}),
                    self.merge(
                        [
                            (
                                self.embedding(data.nodes[x]["x"]),
                                self.embedding(data.nodes[y]["x"]),
                            )
                        ]
                    ),
                )
                for x, y in data.edges
            ]
            while len(clusters[0][0]) < max_width:
                new_clusters = dict()
                for (cluster_l, value_l), (cluster_r, value_r) in combinations(
                    clusters, 2
                ):
                    if len(cluster_l.union(cluster_r)) == len(cluster_l) + 1:
                        u = cluster_l.union(cluster_r)
                        new_clusters[u] = new_clusters.get(u, []) + [(value_l, value_r)]
                clusters = [(c, self.merge(pairs)) for c, pairs in new_clusters.items()]
            # Collect graph embeddings
            ge = self.top_level_merge(clusters)
            result.append(ge)
        return self.output(torch.stack(result))

    def merge(self, pairs):
        return sum(self.fold(self._pair_merge(x, y)) for x, y in pairs)

    def _pair_merge(self, x, y):
        beta = self.softmax(torch.stack([self.left(x), self.right(y)]))
        h2 = beta[0] * self.w_l(x) + beta[1] * self.w_r(y)
        return self.ff_rep(h2) + h2

    def fold(self, h):
        return exp(self.attention_weight(h)) * h

    def top_level_merge(self, clusters):
        t = torch.stack([c for (_, c) in clusters])
        sm = self.softmax(self.top_level_attention_weight(t))
        return torch.sum(t * sm, dim=0)


def graphyk(graph: nx.Graph):
    graph.nodes()
