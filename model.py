import torch
import torch.nn as nn

from typing import Iterable
import networkx as nx
from molecule import Molecule

class ChEBIRecNN(nn.Module):

    def __init__(self):
        super(ChEBIRecNN, self).__init__()

        self.atom_enc = 62
        self.length = 200
        self.output_of_sinks = 500
        self.num_of_classes = 500

        self.activation = nn.ReLU()
        self.sigmoid_activation = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.1)

        self.norm = torch.nn.LayerNorm(self.length)

        self.NN_single_node = nn.Sequential(nn.Linear(self.atom_enc, self.length), nn.ReLU(), nn.Linear(self.length, self.length))
        self.merge = nn.Sequential(nn.Linear(2*self.length, self.length), nn.ReLU(), nn.Linear(self.length, self.length))
        self.register_parameter("attention_weight", torch.nn.Parameter(torch.rand(self.length,1, requires_grad=True)))
        self.register_parameter("dag_weight", torch.nn.Parameter(torch.rand(self.length,1, requires_grad=True)))
        self.final = nn.Sequential(nn.Linear(self.length, self.length), nn.ReLU(), nn.Linear(self.length, self.length), nn.ReLU(), nn.Linear(self.length, self.num_of_classes))

    def forward(self, molecule: Molecule):
        final_outputs = None
        # for each DAG, generate a hidden representation at its sink node
        last = None
        for sink, dag in molecule.dag_to_node.items():
            inputs = {}
            for node in nx.topological_sort(dag):
                atom = self.process_atom(node, molecule)
                if not any(dag.predecessors(node)):
                    output = atom
                else:
                      inp_prev = self.attention(self.attention_weight, inputs[node])
                      inp = torch.cat((inp_prev, atom), dim=0)
                      output = self.activation(self.merge(inp)) + inp_prev
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
        result = self.final(self.attention(self.dag_weight, final_outputs))
        return result

    def process_atom(self, node, molecule):
        return self.dropout(self.activation(self.NN_single_node(molecule.get_atom_features(node))))

    def attention(self, weights, x):
        return torch.sum(torch.mul(torch.softmax(torch.matmul(x, weights), dim=0),x), dim=0)
