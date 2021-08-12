import networkx as nx
import torch
from torch import nn, exp, tensor
import torch.nn.functional as F
import logging
from chem.models.base import JCIBaseNet


logging.getLogger('pysmiles').setLevel(logging.CRITICAL)


class Recursive(JCIBaseNet):
    NAME = "REC"

    def __init__(self, in_d, out_d, num_classes, **kwargs):
        super().__init__(num_classes, **kwargs)
        mem_len = in_d
        self.internal_dimension = in_d
        self.embedding = nn.Embedding(800, 100)

        self.input_weight = nn.Linear(in_d, in_d)
        self.input_hidden_weight = nn.Linear(mem_len, in_d)

        self.state_reset = nn.Linear(mem_len, in_d)

        self.cell_weight = nn.Linear(in_d, in_d)
        self.cell_hidden_weight = nn.Linear(mem_len, in_d)

        self.output_weight = nn.Linear(in_d, in_d)
        self.output_hidden_weight = nn.Linear(mem_len, in_d)

        self.base = torch.nn.parameter.Parameter(torch.empty((in_d,)))
        self.base_memory = torch.nn.parameter.Parameter(torch.empty((mem_len,)))
        self.output = nn.Sequential(nn.Linear(in_d, in_d), nn.ReLU(), nn.Dropout(0.2), nn.Linear(in_d, num_classes))

    def forward(self, batch):
        result = []
        for row in batch:
            graph = row[0]
            c = nx.center(graph)[0]
            d = nx.single_source_shortest_path(graph, c)
            if graph.edges:
                digraph = nx.DiGraph((a,b) if d[a] < d[b] else (b,a) for (a,b) in graph.edges)
            else:
                digraph = nx.DiGraph(graph.nodes)
            child_results = {}
            x = None
            for node in nx.topological_sort(digraph):
                child_values = child_results.pop(node, [])
                if not child_values:
                    hidden_state = self.base_memory
                else:
                    hidden_state = self.merge_childen(child_values)
                x = self.input(self.embedding(graph.nodes[node]["x"]), hidden_state)
                for s in digraph.successors(node):
                    child_results[s] = child_results.get(s, []) + [x]
            result.append(self.output(x))
        return torch.stack(result)

    @staticmethod
    def merge_childen(child_values):
        s = torch.stack(child_values)
        return torch.sum(F.softmax(s) * s, dim=0)

    def input(self,x, hidden):
        r = F.softmax(self.input_weight(x) + self.input_hidden_weight(hidden))
        z = F.softmax(self.cell_weight(x) + self.cell_hidden_weight(hidden))
        n = F.leaky_relu(self.output_weight(x) + self.output_hidden_weight(hidden) + r * self.state_reset(hidden))
        return (1-z) * n + z * hidden


