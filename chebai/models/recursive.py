import logging

import networkx as nx
import torch
import torch.nn.functional as F
from torch import exp, nn, tensor

from chebai.models.base import ChebaiBaseNet

logging.getLogger("pysmiles").setLevel(logging.CRITICAL)


class Recursive(ChebaiBaseNet):
    NAME = "REC"

    def __init__(self, in_d, out_d, num_classes, **kwargs):
        super().__init__(num_classes, **kwargs)
        mem_len = in_d
        self.internal_dimension = in_d
        self.embedding = nn.Embedding(800, 100)

        self.input_post = nn.Linear(in_d, in_d)

        self.input_attention = nn.MultiheadAttention(in_d, 5)
        self.hidden_attention = nn.MultiheadAttention(in_d, 5)
        self.merge_attention = nn.MultiheadAttention(in_d, 5)

        self.hidden_post = nn.Linear(in_d, in_d)

        self.merge_post = nn.Linear(in_d, in_d)

        self.post = nn.Linear(in_d, in_d)

        self.children_attention = nn.MultiheadAttention(in_d, 5)

        self.input_norm_1 = nn.LayerNorm(in_d)
        self.input_norm_2 = nn.LayerNorm(in_d)
        self.hidden_norm_1 = nn.LayerNorm(in_d)
        self.merge_norm_1 = nn.LayerNorm(in_d)
        self.merge_norm_2 = nn.LayerNorm(in_d)

        self.base = torch.nn.parameter.Parameter(torch.empty((in_d,)))
        self.base_memory = torch.nn.parameter.Parameter(torch.empty((mem_len,)))
        self.output = nn.Sequential(
            nn.Linear(in_d, in_d),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_d, num_classes),
        )

    def forward(self, batch):
        result = []
        for row in batch:
            graph = row[0]
            c = nx.center(graph)[0]
            d = nx.single_source_shortest_path(graph, c)
            if graph.edges:
                digraph = nx.DiGraph(
                    (a, b) if d[a] > d[b] else (b, a) for (a, b) in graph.edges
                )
            else:
                digraph = nx.DiGraph(graph)
            child_results = {}
            x = None
            for node in nx.topological_sort(digraph):
                child_values = child_results.pop(node, [])
                inp = self.embedding(graph.nodes[node]["x"])
                if not child_values:
                    hidden_state = self.base_memory
                else:
                    hidden_state = self.merge_childen(child_values, inp)
                x = self.input(inp, hidden_state)
                for s in digraph.successors(node):
                    child_results[s] = child_results.get(s, []) + [x]
            result.append(self.output(x))
        return torch.stack(result)

    def merge_childen(self, child_values, x):
        stack = torch.stack(child_values).unsqueeze(0).transpose(1, 0)
        att = self.children_attention(
            x.expand(1, stack.shape[1], -1).transpose(1, 0), stack, stack
        )[0]
        return torch.sum(att.squeeze(0), dim=0)

    def input(self, x0, hidden):
        x = x0.unsqueeze(0).unsqueeze(0)
        a = self.input_norm_1(x + self.input_attention(x, x, x)[0])
        a = self.input_norm_2(a + F.relu(self.input_post(a)))

        h0 = hidden.unsqueeze(0).unsqueeze(0)
        b = self.hidden_norm_1(h0 + self.input_attention(h0, h0, h0)[0])
        # b = self.norm(b + self.hidden_post(b))

        c = self.merge_norm_1(b + self.merge_attention(a, b, b)[0])
        c = self.merge_norm_2(c + F.relu(self.merge_post(c)))

        return self.post(c).squeeze(0).squeeze(0)
