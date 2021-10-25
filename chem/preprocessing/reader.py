from collections import Counter
import os
import random

from pysmiles.read_smiles import _tokenize
import pandas as pd
import pysmiles as ps
import torch

from chem.preprocessing.collate import DefaultCollater, RaggedCollater


class DataReader:
    COLLATER = DefaultCollater

    def __init__(self, collator_kwargs=None, **kwargs):
        if collator_kwargs is None:
            collator_kwargs = dict()
        self.collater = self.COLLATER(**collator_kwargs)

    def _get_raw_data(self, row):
        return row[0]

    def _get_raw_label(self, row):
        return row[1]

    def name(cls):
        raise NotImplementedError

    def _read_data(self, raw_data):
        return raw_data

    def _read_label(self, raw_label):
        return raw_label

    def _read_components(self, row):
        return self._get_raw_data(row), self._get_raw_label(row)

    def to_data(self, row):
        x, y = self._read_components(row)
        return self._read_data(x), self._read_label(y)


class ChemDataReader(DataReader):
    COLLATER = RaggedCollater

    @classmethod
    def name(cls):
        return "smiles_token"

    def __init__(self, *args, p=0.2, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache = []
        self._p = 0.2

    def _read_components(self, row):
        data = []
        labels = []
        stream = self._get_raw_data(row)
        counter = Counter(stream).elements()
        for t in stream:
            l = 0
            if not all(x == t for x in stream) and random.random() < self._p:
                l = 1
                t0 = t
                while t0 == t:
                    t0 = random.choice(stream)
                t = t0
            data.append(t)
            labels.append(l)
        return data, labels

    def _get_raw_data(self, row):
        l = []
        for v in _tokenize(row.split("\t")[-1]):
            try:
                l.append(self.cache.index(v) + 1)
            except ValueError:
                l.append(len(self.cache) + 1)
                self.cache.append(v)
        return l


class OrdReader(DataReader):
    COLLATER = RaggedCollater

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
