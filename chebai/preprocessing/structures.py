from torch.utils.data.dataset import T_co
import networkx as nx
import torch


class XYData(torch.utils.data.Dataset):
    def __getitem__(self, index) -> T_co:
        return self.x[index], self.y[index]

    def __len__(self):
        # return batch size
        return len(self.x)

    def __init__(self, x, y, **kwargs):
        super().__init__()
        self.additional_fields = kwargs
        self.x = x
        self.y = y

    def to_x(self, device):
        return self.x.to(device)

    def to_y(self, device):
        return self.y.to(device)

    def _to_if_tensor(self, obj, device):
        if isinstance(obj, torch.Tensor):
            return obj.to(device)
        elif isinstance(obj, dict):
            return {k: self._to_if_tensor(v, device) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._to_if_tensor(v, device) for v in obj]
        else:
            return obj

    def to(self, device):
        x = self.to_x(device)
        if self.y is not None:
            y = self.to_y(device)
        else:
            y = None
        return XYData(
            x,
            y,
            **{
                k: self._to_if_tensor(v, device)
                for k, v in self.additional_fields.items()
            },
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
