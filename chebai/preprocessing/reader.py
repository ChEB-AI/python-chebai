from abc import ABC
from collections import Counter
from tempfile import TemporaryDirectory
import os
import pickle
import random

from pysmiles.read_smiles import _tokenize
from tokenizers.implementations import ByteLevelBPETokenizer
from torch_geometric.utils import from_networkx
from transformers import RobertaTokenizerFast
import networkx as nx
import pandas as pd
import pysmiles as ps
import selfies as sf
import torch

from chebai.preprocessing.collate import (
    DefaultCollater,
    GraphCollater,
    RaggedCollater,
)

EMBEDDING_OFFSET = 10
PADDING_TOKEN_INDEX = 0
MASK_TOKEN_INDEX = 1

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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dirname = os.path.dirname(__file__)
        with open(os.path.join(dirname, "bin", "tokens.pkl"), "rb") as pk:
            self.cache = pickle.load(pk)

    def _read_data(self, raw_data):
        return [self.cache.index(v) + EMBEDDING_OFFSET for v in _tokenize(raw_data)]


class ChemDataUnlabeledReader(ChemDataReader):
    COLLATER = RaggedCollater

    @classmethod
    def name(cls):
        return "smiles_token_unlabeled"

    def _read_components(self, row):
        return row[0], None


class ChemBPEReader(DataReader):
    COLLATER = RaggedCollater

    @classmethod
    def name(cls):
        return "smiles_bpe"

    def __init__(self, *args, data_path=None, max_len=1800, vsize=4000, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = RobertaTokenizerFast.from_pretrained(
            data_path, max_len=max_len
        )

    def _get_raw_data(self, row):
        return self.tokenizer(row[0])["input_ids"]


class SelfiesReader(DataReader):
    COLLATER = RaggedCollater

    def __init__(self, *args, data_path=None, max_len=1800, vsize=4000, **kwargs):
        super().__init__(*args, **kwargs)
        with open("chebai/preprocessing/bin/selfies.txt", "rt") as pk:
            self.cache = [l.strip() for l in pk]

    @classmethod
    def name(cls):
        return "selfies"

    def _get_raw_data(self, row):
        try:
            splits = sf.split_selfies(sf.encoder(row[0].strip(), strict=True))
        except Exception as e:
            print(e)
            return
        else:
            return [self.cache.index(x) + EMBEDDING_OFFSET for x in splits]


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


class GraphReader(DataReader):
    COLLATER = GraphCollater

    @classmethod
    def name(cls):
        return "graph"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        with open("chebai/preprocessing/bin/tokens.pkl", "rb") as pk:
            self.cache = pickle.load(pk)

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

    def to_data(self, row):
        d = self.process_smiles(row[1])
        if d is not None and d.num_nodes > 1:
            d.y = torch.tensor(row[2:].astype(bool)).unsqueeze(0)
            return d


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
