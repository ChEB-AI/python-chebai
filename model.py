import torch
import torch.nn as nn

from typing import Iterable
import networkx as nx

class ChEBIRecNN(nn.Module):

    def __init__(self):
        super(ChEBIRecNN, self).__init__()

        self.length = 104
        self.output_of_sinks = 500
        self.num_of_classes = 500

        self.activation = nn.ReLU()
        self.sigmoid_activation = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.2)

        self.NN_single_node = nn.Linear(self.length, self.length)
        self.merge = nn.Linear(2*self.length, self.length)
        self.attention_weight = torch.transpose(torch.autograd.Variable(torch.rand(self.length)), 1, 0)
        self.dag_weight = torch.transpose(torch.autograd.Variable(torch.rand(self.length)), 1, 0)
        self.final = nn.Linear(self.length, self.num_of_classes)

    def forward(self, DAGs_of_a_mol: Iterable[nx.DiGraph]):
        sinks_output = []
        # for each DAG, generate a hidden representation at its sink node
        final_outputs = torch.empty()
        for dag in DAGs_of_a_mol:
            inputs = {node: torch.empty(self.length) for node in dag.nodes}
            last = None
            for node in nx.topological_sort(dag):
                if not dag.has_predecessor(node):
                    output = self.activation(self.NN_single_node(nx.get_node_attributes(dag, "atom_features")))
                    for succ in dag.successors(node):
                        inputs[succ] = inputs[succ].stack(output)
                else:
                      inp = torch.sum(torch.softmax(self.attention_weight*inputs[node], dim=0)*inputs[node])

                      inp = torch.cat((inp, nx.get_node_attributes(dag, "atom_features")))
                      output = self.activation(self.merge(inp))
                      for succ in dag.successors(node):
                          inputs[succ] = inputs[succ].stack(output)
                last = output
            final_outputs = final_outputs.stack(last)
        # take the average of hidden representation at all sinks
        return self.final(torch.sum(torch.softmax(self.dag_weight*final_outputs, dim=0)*final_outputs))
