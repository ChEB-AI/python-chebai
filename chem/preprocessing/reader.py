import os
import random

from collections import Counter

from pysmiles.read_smiles import _tokenize
import pysmiles as ps
import torch

class DataReader:
    def _get_raw_data(self, row):
        return row[0]

    def _get_raw_label(self, row):
        return row[1]

    @classmethod
    def name(cls):
        raise NotImplementedError

    def _read_data(self, raw_data):
        return raw_data

    def _read_label(self, raw_label):
        return raw_label

    def to_data(self, row):
        return self._read_data(self._get_raw_data(row)), self._read_label(
            self._get_raw_label(row)
        )


class DefaultLabeler:
    def __call__(self, data):
        return data[1]


class ReplacementLabeler:
    def __call__(self, data):
        x, _ = data
        alphabet = set(x)
        i = random.randint(0, len(x) - 1)
        o = x.copy()
        r = random.choice(list(alphabet.difference([o[i]])))
        o[i] = r
        return o, [1 if j == i else 0 for j in range(len(x))]


class ChemDataReader(DataReader):
    @classmethod
    def name(cls):
        return "smiles_token"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache = []

    def _get_raw_label(self, row):
        return None

    def _get_raw_data(self, row):
        return row

    def _read_data(self, raw_data):
        l = []
        for v in _tokenize(raw_data):
            try:
                l.append(self.cache.index(v))
            except ValueError:
                l.append(len(self.cache) + 1)
                self.cache.append(v)
        return l


class OrdReader(DataReader):
    @classmethod
    def name(cls):
        return "ord"

    def _read_data(self, raw_data):
        return [ord(s) for s in raw_data]


class MolDatareader(DataReader):
    @classmethod
    def name(cls):
        return "mol"

    def __init__(self, batch_size, **kwargs):
        super().__init__(batch_size, **kwargs)
        self.cache = []

    def to_data(self, row):
        return self.get_encoded_mol(
            row[self.SMILES_INDEX], self.cache
        ), self._get_label(row)

    def get_encoded_mol(self, smiles, cache):
        try:
            mol = ps.read_smiles(smiles)
        except ValueError:
            return None
        d = {}
        for node in mol.nodes:
            try:
                m = mol.nodes[node]["element"]
            except KeyError:
                m = "*"
            try:
                x = cache.index(m)
            except ValueError:
                x = len(cache)
                cache.append(m)
            d[node] = torch.tensor(x)
            for attr in list(mol.nodes[node].keys()):
                del mol.nodes[node][attr]
        nx.set_node_attributes(mol, d, "x")
        return mol


class GraphDataset(DataReader):
    @classmethod
    def name(cls):
        return "graph"

    def __init__(self, batch_size, **kwargs):
        super().__init__(batch_size, **kwargs)
        self.collater = Collater(
            follow_batch=["x", "edge_attr", "edge_index", "label"], exclude_keys=[]
        )
        self.cache = []

    def process_smiles(self, smiles):
        def cache(m):
            try:
                x = self.cache.index(m)
            except ValueError:
                x = len(self.cache)
                self.cache.append(m)
            return x

        try:
            mol = ps.read_smiles(smiles)
        except ValueError:
            return None
        d = {}
        de = {}
        for node in mol.nodes:
            try:
                m = mol.nodes[node]["element"]
            except KeyError:
                m = "*"
            d[node] = cache(m)
            for attr in list(mol.nodes[node].keys()):
                del mol.nodes[node][attr]
        for edge in mol.edges:
            de[edge] = mol.edges[edge]["order"]
            for attr in list(mol.edges[edge].keys()):
                del mol.edges[edge][attr]
        nx.set_node_attributes(mol, d, "x")
        nx.set_edge_attributes(mol, de, "edge_attr")
        return from_networkx(mol)

    def collate(self, list_of_tuples):
        return self.collater(list_of_tuples)

    def to_data(self, df):
        for row in df.values[:DATA_LIMIT]:
            d = self.process_smiles(row[self.SMILES_INDEX])
            if d is not None and d.num_nodes > 1:
                d.y = torch.tensor(row[self.LABEL_INDEX :].astype(bool)).unsqueeze(0)
                yield d


try:
    from k_gnn import TwoMalkin
except ModuleNotFoundError:
    pass
else:
    from k_gnn.dataloader import collate

    class GraphTwoDataset(GraphDataset):
        @classmethod
        def name(cls):
            return "graph_k2"

        def to_data(self, df: pd.DataFrame):
            for data in super().to_data(df)[:DATA_LIMIT]:
                if data.num_nodes >= 6:
                    x = data.x
                    data.x = data.x.unsqueeze(0)
                    data = TwoMalkin()(data)
                    data.x = x
                    yield data

        def collate(self, list_of_tuples):
            return collate(list_of_tuples)

    class JCIExtendedGraphTwoData(JCIExtendedBase, GraphTwoDataset):
        pass

    class JCIGraphTwoData(JCIBase, GraphTwoDataset):
        pass
