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
        self.NN_internal_with_1_parent = nn.Linear(2 * self.length, self.length)
        self.NN_internal_with_2_parents = nn.Linear(3 * self.length, self.length)
        self.NN_internal_with_3_parents = nn.Linear(4 * self.length, self.length)
        self.NN_internal_with_4_parents = nn.Linear(5 * self.length, self.length)
        self.NN_internal_with_5_parents = nn.Linear(6 * self.length, self.length)
        self.NN_internal_with_6_parents = nn.Linear(7 * self.length, self.length)
        self.NN_internal_with_7_parents = nn.Linear(8 * self.length, self.length)
        self.NN_sink_with_1_parent = nn.Linear(1 * self.length, self.output_of_sinks)
        self.NN_sink_with_2_parents = nn.Linear(2 * self.length, self.output_of_sinks)
        self.NN_sink_with_3_parents = nn.Linear(3 * self.length, self.output_of_sinks)
        self.NN_sink_with_4_parents = nn.Linear(4 * self.length, self.output_of_sinks)
        self.NN_sink_with_5_parents = nn.Linear(5 * self.length, self.output_of_sinks)
        self.NN_sink_with_6_parents = nn.Linear(6 * self.length, self.output_of_sinks)
        self.NN_sink_with_7_parents = nn.Linear(7 * self.length, self.output_of_sinks)
        self.NN_classification_logits = nn.Linear(self.output_of_sinks, self.num_of_classes)

        self.NN_for_hidden_representation_map = {
            1: self.NN_internal_with_1_parent,
            2: self.NN_internal_with_2_parents,
            3: self.NN_internal_with_3_parents,
            4: self.NN_internal_with_4_parents,
            5: self.NN_internal_with_5_parents,
            6: self.NN_internal_with_6_parents,
            7: self.NN_internal_with_7_parents
        }
        self.NN_for_sinks_map = {
            1: self.NN_sink_with_1_parent,
            2: self.NN_sink_with_2_parents,
            3: self.NN_sink_with_3_parents,
            4: self.NN_sink_with_4_parents,
            5: self.NN_sink_with_5_parents,
            6: self.NN_sink_with_6_parents,
            7: self.NN_sink_with_7_parents
        }

    def forward(self, DAGs_of_a_mol :Iterable[nx.DiGraph]):
      sinks_output = []

      # for each DAG, generate a hidden representation at its sink node
      for dag in DAGs_of_a_mol:
          outputs = {}
          for node in nx.topological_sort(dag):
              if node in sources:
                network = self.NN_single_node

                if node not in outputs.keys():
                  outputs[node] = []

                logits = network(torch.tensor(context_matrix[node]))
                outputs[node] = self.activation(logits)

              else:
                all_inputs = []
                all_inputs.append(torch.tensor(context_matrix[node]))

                for parent in parents[node]:
                  all_inputs.append(torch.tensor(context_matrix[parent]))

                network = self.NN_for_hidden_representation_map[len(parents[node])]

                if node not in outputs.keys():
                  outputs[node] = []
                  logits = network(torch.cat(all_inputs, dim=0))
                  outputs[node] = self.activation(logits)

          all_sink_inputs = []
          for sink_parent in sink_parents:
              all_sink_inputs.append(outputs[sink_parent])

          sink_network = self.NN_for_sinks_map[len(sink_parents)]
          result_at_sink = sink_network(torch.cat(all_sink_inputs, dim=0))

          sinks_output.append(result_at_sink)

      # take the average of hidden representation at all sinks
      avg_sinks_output = [float(sum(col)/len(col)) for col in zip(*sinks_output)]
      network = self.NN_classification_logits
      final_logits = network(torch.tensor(avg_sinks_output).double())
      return self.sigmoid_activation(final_logits)
