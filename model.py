import torch
import torch.nn as nn

from typing import Iterable
import networkx as nx
from molecule import Molecule

class ChEBIRecNN(nn.Module):

    def __init__(self):
        super(ChEBIRecNN, self).__init__()

        self.length = 62
        self.output_of_sinks = 500
        self.num_of_classes = 500

        self.activation = nn.ReLU()
        self.sigmoid_activation = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.2)

        self.NN_single_node = nn.Linear(self.length, self.length).double()
        self.merge = nn.Linear(2*self.length, self.length).double()
        self.attention_weight = torch.autograd.Variable(torch.rand(self.length,1).double())
        self.dag_weight = torch.autograd.Variable(torch.rand(self.length,1).double())
        self.final = nn.Linear(self.length, self.num_of_classes).double()

    def forward(self, molecule: Molecule):
        final_outputs = None
        # for each DAG, generate a hidden representation at its sink node
        for sink, dag in molecule.dag_to_node.items():
            inputs = {}
            last = None
            for node in nx.topological_sort(dag):
                atom = molecule.get_atom_features(node)
                if not any(dag.predecessors(node)):
                    output = self.activation(self.NN_single_node(atom))
                    for succ in dag.successors(node):
                        try:
                            inputs[succ] = torch.cat((inputs[succ], output.unsqueeze(0)))
                        except KeyError:
                            inputs[succ] = output.unsqueeze(0)
                else:
                      inp = self.attention(self.attention_weight, inputs[node])
                      inp = torch.cat((inp, atom), dim=0)
                      output = self.activation(self.merge(inp))
                      for succ in dag.successors(node):
                          try:
                              inputs[succ] = torch.cat((inputs[succ], output.unsqueeze(0)))
                          except KeyError:
                              inputs[succ] = output.unsqueeze(0)
                last = output
            if final_outputs is not None:
                final_outputs = torch.cat((final_outputs, last.unsqueeze(0)))
            else:
                final_outputs = last.unsqueeze(0)
        # take the average of hidden representation at all sinks
        return torch.sigmoid(self.final(self.attention(self.dag_weight, final_outputs)))

    def attention(self, weights, x):
        return torch.sum(torch.mul(torch.softmax(torch.matmul(x, weights), dim=0),x), dim=0)