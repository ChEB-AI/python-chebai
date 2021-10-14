import networkx as nx
import torch
from pytorch_lightning.utilities.apply_func import TransferableDataType
from torch.utils.data.dataset import T_co
from torch_geometric.data import Data


class PrePairData(Data):
    def __init__(self, l=None, r=None, label=None):
        super(PrePairData, self).__init__()
        self.l = l
        self.r = r
        self.label = label


class PairData(Data):
    def __init__(self, ppd: PrePairData, graph):
        super(PairData, self).__init__()

        s = graph.nodes[ppd.l]["enc"]
        self.edge_index_s = s.edge_index
        self.x_s = s.x

        t = graph.nodes[ppd.r]["enc"]
        self.edge_index_t = t.edge_index
        self.x_t = t.x

        self.label = ppd.label

    def __inc__(self, key, value):
        if key == "edge_index_s":
            return self.x_s.size(0)
        if key == "edge_index_t":
            return self.x_t.size(0)
        else:
            return super().__inc__(key, value)


class XYData(torch.utils.data.Dataset, TransferableDataType):
    def __getitem__(self, index) -> T_co:
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)

    def __init__(self, x, y, additional_fields=None, **kwargs):
        super().__init__(**kwargs)
        if additional_fields:
            for key, value in additional_fields.items():
                setattr(self, key, value)
        self.x = x
        self.y = y

        self.additional_fields = (
            list(additional_fields.keys()) if additional_fields else []
        )

    def to_x(self, device):
        return self.x.to(device)

    def to_y(self, device):
        return self.y.to(device)

    def to(self, device):
        x = self.to_x(device)
        y = self.to_y(device)
        return XYData(
            x,
            y,
            additional_fields={k: getattr(self, k) for k in self.additional_fields},
        )


class XYMolData(XYData):
    def to_x(self, device):
        l = []
        for g in self.x:
            graph = g.copy()
            nx.set_node_attributes(
                graph,
                {k: v.to(device) for k, v in nx.get_node_attributes(g, "x").items()},
                "x",
            )
            l.append(graph)
        return tuple(l)
