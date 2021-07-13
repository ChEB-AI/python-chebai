__all__ = [
    "JCIData",
    "JCIExtendedData",
    "JCIGraphData",
    "JCIExtendedGraphData",
]

from abc import ABC
from typing import Iterator, Any, Union, List, Optional

import fastobo
import networkx as nx
import pickle
import os
from sklearn.model_selection import train_test_split
import torch
import requests
import pysmiles as ps
import random
from itertools import chain
import glob
import pytorch_lightning as pl
from pytorch_lightning.utilities.apply_func import TransferableDataType
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import T_co
from torch.nn.utils.rnn import pad_sequence
from torch import Tensor
from torch_geometric.data import InMemoryDataset
from k_gnn import TwoMalkin
from k_gnn.dataloader import collate
import pandas as pd

from torch_geometric.utils.convert import from_networkx
from torch_geometric.data import Dataset as TGDataset, Data
from torch_geometric.data.dataloader import Collater
import multiprocessing as mp

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
        if key == 'edge_index_s':
            return self.x_s.size(0)
        if key == 'edge_index_t':
            return self.x_t.size(0)
        else:
            return super().__inc__(key, value)


def extract_largest_index(path, kind):
    return max(int(n[len(path+kind)+2:-len(".pt")]) for n in glob.glob(os.path.join(path, f'{kind}.*.pt')))+1


class XYBaseDataModule(pl.LightningDataModule):

    ROOT = None
    PATH = []
    RAW_PATH = []
    SMILES_INDEX = None
    LABEL_INDEX = None

    def __init__(self, batch_size=1, **kwargs):
        root = os.path.join("data", self.ROOT)
        self.processed_dir = os.path.join(root,"processed", *self.PATH)
        self.raw_dir = os.path.join(root, "raw", *self.RAW_PATH)
        self.train_split=0.85
        self.batch_size = batch_size
        super().__init__(root, **kwargs)

    @property
    def num_classes(self):
        return 500

    def dataloader(self, kind, **kwargs):
        return DataLoader(torch.load(os.path.join(self.processed_dir, f"{kind}.pt")), collate_fn=self.collate, batch_size=self.batch_size, **kwargs)

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return self.dataloader("train", shuffle=True, **kwargs)

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return self.dataloader("validation", shuffle=False, **kwargs)

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return self.dataloader("test", shuffle=False, **kwargs)

    def setup(self, **kwargs):
        if any(not os.path.isfile(os.path.join(self.processed_dir, f)) for f in self.processed_file_names):
            self.setup_processed()

    def setup_processed(self):
        raise NotImplementedError

    @property
    def processed_file_names(self):
        raise NotImplementedError

    def to_data(self, df: pd.DataFrame):
        raise NotImplementedError


class JCIBase(XYBaseDataModule):
    ROOT = "JCI"
    RAW_PATH = []
    LABEL_INDEX = 2
    SMILES_INDEX = 1


    @property
    def processed_file_names(self):
        return ["test.pt", "train.pt", "validation.pt"]

    def download(self):
        pass

    @property
    def raw_file_names(self):
        return ["test.pkl", "train.pkl", "validation.pkl"]

    def prepare_data(self, *args, **kwargs):
        print("Check for raw data in", self.raw_dir)
        if any(not os.path.isfile(os.path.join(self.raw_dir, f)) for f in self.raw_file_names):
            raise ValueError("Raw data is missing")

    def setup_processed(self):
        print("Transform splits")
        os.makedirs(self.processed_dir, exist_ok=True)
        for k in ["test", "train", "validation"]:
            print("transform", k)
            torch.save(list(self.to_data(pickle.load(open(os.path.join(self.raw_dir, f"{k}.pkl"), "rb")))), os.path.join(self.processed_dir, f"{k}.pt"))


class OrdDataset(XYBaseDataModule):
    def collate(self, list_of_tuples):
        x, y = zip(*list_of_tuples)
        return XYData(pad_sequence([torch.tensor(a) for a in x],
                                    batch_first=True),
                      pad_sequence([torch.tensor(a) for a in y],
                                   batch_first=True),
                      additional_fields=dict(lens=list(map(len, x))))

    def to_data(self, df: pd.DataFrame):
        for row in df.values:
            yield [ord(s) for s in row[self.SMILES_INDEX]], row[
                                                            self.LABEL_INDEX:].astype(
                bool)


class MolDataset(XYBaseDataModule):
    def __init__(self, batch_size, **kwargs):
        super().__init__(batch_size, **kwargs)
        self.cache = []

    def collate(self, list_of_tuples):
        x, y = zip(*list_of_tuples)
        return XYData(x, y)

    def setup_processed(self):
        super().setup_processed()
        torch.save(self.cache,
                   os.path.join(self.processed_dir, f"embeddings.pt"))

    def to_data(self, df: pd.DataFrame):
        for row in df.values[:10]:
            yield get_encoded_mol(row[self.SMILES_INDEX], self.cache), row[
                                                            self.LABEL_INDEX:].astype(
                bool)


class JCIData(JCIBase, OrdDataset):
    PATH = ["smiles_ord"]


class JCIMolData(JCIBase, MolDataset):
    PATH = ["mol"]

    def to(self, device):
        return XYData(self.x.to(device), self.y.to(device), additional_fields={k: getattr(self, k) for k in self.additional_fields})


class JCIExtendedBase(XYBaseDataModule):
    ROOT = "JCI_extended"
    RAW_PATH = []
    LABEL_INDEX = 3
    SMILES_INDEX = 2

    def setup_processed(self):
        print("Transform splits")
        os.makedirs(self.processed_dir, exist_ok=True)
        for k in ["test", "train", "validation"]:
            print("transform", k)
            torch.save(list(self.to_data(pickle.load(open(os.path.join(self.raw_dir, f"{k}.pkl"), "rb")))),
                       os.path.join(self.processed_dir, f"{k}.pt"))

    def extract_class_hierarchy(self):
        elements = [term_callback(clause) for clause in
                    fastobo.load(os.path.join(self.raw_dir, "chebi.obo")) if
                    clause and ":" in str(clause.id)]
        g = nx.DiGraph()
        for n in elements:
            g.add_node(n["id"], **n)
        g.add_edges_from([(p, q["id"]) for q in elements for p in q["parents"]])
        print("Compute transitive closure")
        return nx.transitive_closure_dag(g)

    def get_splits(self, g):
        fixed_nodes = list(g.nodes)
        print("Split datasets")
        random.shuffle(fixed_nodes)
        train_split, test_split = train_test_split(fixed_nodes,
                                                   train_size=self.train_split,
                                                   shuffle=True)
        test_split, validation_split = train_test_split(test_split,
                                                        train_size=self.train_split,
                                                        shuffle=True)
        return train_split[:10], test_split[:2], validation_split[:2]

    def save(self, g, train_split, test_split, validation_split):
        smiles = nx.get_node_attributes(g, "smiles")
        names = nx.get_node_attributes(g, "name")
        print("build labels")
        for k, nodes in dict(train=train_split, test=test_split, validation=validation_split).items():
            print("Process", k)
            data = pd.DataFrame(dict(id=nodes))
            data["name"] = data["id"].apply(lambda node: names.get(node))
            data["SMILES"] = data["id"].apply(lambda node: smiles.get(node))
            data = data[~data["SMILES"].isnull()]
            for n in JCI_500_COLUMNS_INT:
                data[n] = data["id"].apply(lambda node: ((n in g.predecessors(node)) or (n == node)))
            data = data[data.iloc[:,3:].any(1)]
            pickle.dump(data, open(os.path.join(self.raw_dir, f"{k}.pkl"), "wb"))

    @property
    def processed_file_names(self):
        return ["test.pt", "train.pt", "validation.pt"]

    @property
    def raw_file_names(self):
        return ["test.pkl", "train.pkl", "validation.pkl"]

    def prepare_data(self, *args, **kwargs):
        print("Check for raw data in", self.raw_dir)
        if any(not os.path.isfile(os.path.join(self.raw_dir, f)) for f in self.raw_file_names):
            os.makedirs(self.raw_dir, exist_ok=True)
            print("Missing raw data. Go fetch...")
            if not os.path.isfile(os.path.join(self.raw_dir, "chebi.obo")):
                print("Load ChEBI ontology")
                url = 'http://purl.obolibrary.org/obo/chebi.obo'
                r = requests.get(url, allow_redirects=True)
                open(os.path.join(self.raw_dir, "chebi.obo"), 'wb').write(
                    r.content)
            g = self.extract_class_hierarchy()
            self.save(g, *self.get_splits(g))


class JCIExtendedData(JCIExtendedBase, OrdDataset):
    PATH = ["smiles_ord"]


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

        self.additional_fields = list(additional_fields.keys()) if additional_fields else []

    def to(self, device):
        x = self.x if not isinstance(self.x, Tensor) else self.x.to(device)
        y = self.y if not isinstance(self.y, Tensor) else self.y.to(device)
        return XYData(x, y, additional_fields={k: getattr(self, k) for k in self.additional_fields})


class GraphDataset(XYBaseDataModule):

    def __init__(self, batch_size, **kwargs):
        super().__init__(batch_size, **kwargs)
        self.collater = Collater(follow_batch=["x", "edge_index", "label"])
        self.cache = []

    def setup_processed(self):
        super().setup_processed()
        torch.save(self.cache, os.path.join(self.processed_dir, f"embeddings.pt"))

    def process_smiles(self, smiles):
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
                x = self.cache.index(m)
            except ValueError:
                x = len(self.cache)
                self.cache.append(m.copy())
            d[node] = x
            for attr in list(mol.nodes[node].keys()):
                del mol.nodes[node][attr]
        nx.set_node_attributes(mol, d, "x")
        nx.set_edge_attributes(mol, {e:e for e in mol.edges}, "original" )
        return from_networkx(mol)

    def collate(self, list_of_tuples):
        return self.collater(list_of_tuples)

    def to_data(self, df: pd.DataFrame):
        for row in df.values:
            d = self.process_smiles(row[self.SMILES_INDEX])
            if d is not None and d.num_nodes > 1:
                d.y = torch.tensor(row[self.LABEL_INDEX:].astype(bool)).unsqueeze(0)
                yield d


class JCIGraphData(JCIBase, GraphDataset):

    PATH = ["graph"]


class JCIExtendedGraphData(JCIExtendedBase, GraphDataset):

    PATH = ["graph"]


class JCIExtendedGraphTwoData(JCIExtendedGraphData):
    PATH = ["graph_k2"]

    def to_data(self, df: pd.DataFrame):
        for data in super().to_data(df):
            if data.num_nodes >=6:
                x = data.x
                data.x = data.x.unsqueeze(0)
                data = TwoMalkin()(data)
                data.x = x
                yield data

    def collate(self, list_of_tuples):
        return collate(list_of_tuples)


class PartOfData(TGDataset):

    def len(self):
        return extract_largest_index(self.processed_dir, self.kind)

    def get(self, idx):
        return pickle.load(open(os.path.join(self.processed_dir, f"{self.kind}.{idx}.pt"), "rb"))

    def __init__(self, root, kind="train", batch_size=100, train_split=0.95, part_split=0.1, pre_transform=None, **kwargs):
        self.cache_file = ".part_data.pkl"
        self._ignore = set()
        self.train_split = train_split
        self.part_split = part_split
        self.kind = kind
        self.batch_size = batch_size
        super().__init__(root, pre_transform=pre_transform, transform=self.transform, **kwargs)
        self.graph = pickle.load(open(os.path.join(self.processed_dir, self.processed_cache_names[0]), "rb"))


    def transform(self, ppds):
        return [PairData(ppd, self.graph) for ppd in ppds]

    def download(self):
        url = 'http://purl.obolibrary.org/obo/chebi.obo'
        r = requests.get(url, allow_redirects=True)
        open(self.raw_paths[0], 'wb').write(r.content)

    def process(self):
        doc = fastobo.load(self.raw_paths[0])
        elements = list()
        for clause in doc:
            callback = CALLBACKS.get(type(clause))
            if callback is not None:
                elements.append(callback(clause))

        g = nx.DiGraph()
        for n in elements:
            g.add_node(n["id"], **n)
        g.add_edges_from([(p, q["id"]) for q in elements for p in q["parents"]])

        print("pass parts")
        self.pass_parts(g, 23367, set())
        print("Load data")
        children = frozenset(list(nx.single_source_shortest_path(g, 23367).keys())[:100])
        parts = frozenset({p for c in children for p in g.nodes[c]["has_part"]})

        print("Create molecules")
        nx.set_node_attributes(g, dict(map(get_mol_enc,((i,g.nodes[i]["smiles"]) for i in (children.union(parts))))), "enc")

        print("Filter invalid structures")
        children = [p for p in children if g.nodes[p]["enc"]]
        random.shuffle(children)
        children, children_test_only = train_test_split(children[:100], test_size=self.part_split)

        parts = [p for p in parts if g.nodes[p]["enc"]]
        random.shuffle(parts)
        parts, parts_test_only = train_test_split(parts, test_size=self.part_split)

        has_parts = {n: g.nodes[n]["has_part"] for n in g.nodes}
        pickle.dump(g, open(os.path.join(self.processed_dir, self.processed_cache_names[0]), "wb"))
        del g

        print("Transform into torch structure")

        kinds = ("train", "test", "validation")
        batches = {k:list() for k in kinds}
        batch_counts = {k:0 for k in kinds}
        for l in children:
            pts = has_parts[l]
            for r in parts:
                # If there are no positive labels, move the datapoint to test set (where it has positive labels)
                if pts.intersection(parts):
                    if random.random() < self.train_split:
                        k = "train"
                    elif random.random() < self.train_split or batch_counts["validation"]:
                        k = "test"
                    else:
                        k = "validation"
                else:
                    k = "test"
                batches[k].append(PrePairData(l, r, float(r in pts)))
                if len(batches[k]) >= self.batch_size:
                    pickle.dump(batches[k], open(os.path.join(self.processed_dir, f"{k}.{batch_counts[k]}.pt"), "wb"))
                    batch_counts[k] += 1
                    batches[k] = []

        k = k0 = "train"
        b = batches[k]
        if b:
            if not batch_counts["validation"]:
                k = "validation"
            pickle.dump(b, open(os.path.join(self.processed_dir, f"{k}.{batch_counts[k]}.pt"), "wb"))
            del batches[k0]
            del b

        test_batch = batches["test"]
        batch_count = batch_counts["test"]
        for l,r in chain(((l,r) for l in children for r in parts_test_only), ((l,r) for l in children_test_only for r in parts_test_only), ((l,r) for l in children_test_only for r in parts)):
            test_batch.append(PrePairData(l, r, float(r in has_parts[l])))
            if len(test_batch) >= self.batch_size:
                pickle.dump(test_batch, open(os.path.join(self.processed_dir, f"test.{batch_count}.pt"), "wb"))
                batch_count += 1
                test_batch = []
        if test_batch:
            pickle.dump(test_batch, open(os.path.join(self.processed_dir, f"test.{batch_count}.pt"), "wb"))

    @property
    def raw_file_names(self):
        return ["chebi.obo"]

    @property
    def processed_file_names(self):
        return ["train.0.pt", "test.0.pt", "validation.0.pt"]

    @property
    def processed_cache_names(self):
        return ["cache.pt"]

    def pass_parts(self, d: nx.DiGraph, root, parts=None):
        if parts is None:
            parts = set()
        parts = set(parts.union(d.nodes[root]["has_part"]))
        nx.set_node_attributes(d, {root: parts}, "has_part")
        for child in d.successors(root):
            self.pass_parts(d, child, set(parts))

    def extract_children(self, d: nx.DiGraph, root, part_cache):
        smiles = d.nodes[root]["smiles"]
        if smiles:
            yield root
        for child in d.successors(root):
            for r in self.extract_children(d, child, part_cache):
                yield r

def chebi_to_int(s):
    return int(s[s.index(":")+1:])

def term_callback(doc):
    parts = set()
    parents = []
    name = None
    smiles = None
    for clause in doc:
        if isinstance(clause, fastobo.term.PropertyValueClause):
            t = clause.property_value
            if str(t.relation) == "http://purl.obolibrary.org/obo/chebi/smiles":
                assert smiles is None
                smiles = t.value
        elif isinstance(clause, fastobo.term.RelationshipClause):
            if str(clause.typedef) == "has_part":
                parts.add(chebi_to_int(str(clause.term)))
        elif isinstance(clause, fastobo.term.IsAClause):
            parents.append(chebi_to_int(str(clause.term)))
        elif isinstance(clause, fastobo.term.NameClause):
            name = str(clause.name)
    return {
        "id": chebi_to_int(str(doc.id)),
        "parents": parents,
        "has_part": parts,
        "name": name,
        "smiles": smiles,
    }

def get_mol_enc(x):
    i, s= x
    return i, mol_to_data(s) if s else None


def mol_to_data(smiles):
    try:
        mol = ps.read_smiles(smiles)
    except:
        return None
    d = {}
    for node in mol.nodes:
        el = mol.nodes[node].get("element")
        if el is not None:
            v = atom_index.index(el)
            base = [float(i == v) for i in range(118)]
            wildcard = [0.0]
        else:
            base = [0.0 for i in range(118)]
            wildcard = [1.0]
        d[node] = base + [mol.nodes[node].get("charge",0.0), mol.nodes[node].get("hcount",0.0)] + wildcard

        for attr in list(mol.nodes[node].keys()):
            del mol.nodes[node][attr]
    nx.set_node_attributes(mol, d, "x")
    return from_networkx(mol)


def get_encoded_mol(smiles, cache):
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
        d[node] = x
        for attr in list(mol.nodes[node].keys()):
            del mol.nodes[node][attr]
    nx.set_node_attributes(mol, d, "x")
    return mol


atom_index =(
            "\*",
            "H",
            "He",
            "Li",
            "Be",
            "B",
            "C",
            "N",
            "O",
            "F",
            "Ne",
            "Na",
            "Mg",
            "Al",
            "Si",
            "P",
            "S",
            "Cl",
            "Ar",
            "K",
            "Ca",
            "Sc",
            "Ti",
            "V",
            "Cr",
            "Mn",
            "Fe",
            "Co",
            "Ni",
            "Cu",
            "Zn",
            "Ga",
            "Ge",
            "As",
            "Se",
            "Br",
            "Kr",
            "Rb",
            "Sr",
            "Y",
            "Zr",
            "Nb",
            "Mo",
            "Tc",
            "Ru",
            "Rh",
            "Pd",
            "Ag",
            "Cd",
            "In",
            "Sn",
            "Sb",
            "Te",
            "I",
            "Xe",
            "Cs",
            "Ba",
            "La",
            "Hf",
            "Ta",
            "W",
            "Re",
            "Os",
            "Ir",
            "Pt",
            "Au",
            "Hg",
            "Tl",
            "Pb",
            "Bi",
            "Po",
            "At",
            "Rn",
            "Fr",
            "Ra",
            "Ac",
            "Rf",
            "Db",
            "Sg",
            "Bh",
            "Hs",
            "Mt",
            "Ds",
            "Rg",
            "Cn",
            "Nh",
            "Fl",
            "Mc",
            "Lv",
            "Ts",
            "Og",
            "Ce",
            "Pr",
            "Nd",
            "Pm",
            "Sm",
            "Eu",
            "Gd",
            "Tb",
            "Dy",
            "Ho",
            "Er",
            "Tm",
            "Yb",
            "Lu",
            "Th",
            "Pa",
            "U",
            "Np",
            "Pu",
            "Am",
            "Cm",
            "Bk",
            "Cf",
            "Es",
            "Fm",
            "Md",
            "No",
            "Lr",
            "c",
            "n",
            "s",
            "o",
            "se",
            "p",
)

JCI_500_COLUMNS = ['CHEBI:25716',
 'CHEBI:72010',
 'CHEBI:60926',
 'CHEBI:39206',
 'CHEBI:24315',
 'CHEBI:22693',
 'CHEBI:23981',
 'CHEBI:23066',
 'CHEBI:35343',
 'CHEBI:18303',
 'CHEBI:60971',
 'CHEBI:35753',
 'CHEBI:24836',
 'CHEBI:59268',
 'CHEBI:35992',
 'CHEBI:51718',
 'CHEBI:27093',
 'CHEBI:38311',
 'CHEBI:46940',
 'CHEBI:26399',
 'CHEBI:27325',
 'CHEBI:33637',
 'CHEBI:37010',
 'CHEBI:36786',
 'CHEBI:59777',
 'CHEBI:36871',
 'CHEBI:26799',
 'CHEBI:50525',
 'CHEBI:26848',
 'CHEBI:52782',
 'CHEBI:75885',
 'CHEBI:37533',
 'CHEBI:47018',
 'CHEBI:27150',
 'CHEBI:26707',
 'CHEBI:131871',
 'CHEBI:134179',
 'CHEBI:24727',
 'CHEBI:59238',
 'CHEBI:26373',
 'CHEBI:46774',
 'CHEBI:33642',
 'CHEBI:38686',
 'CHEBI:74222',
 'CHEBI:23666',
 'CHEBI:46770',
 'CHEBI:16460',
 'CHEBI:37485',
 'CHEBI:21644',
 'CHEBI:52565',
 'CHEBI:33576',
 'CHEBI:76170',
 'CHEBI:46640',
 'CHEBI:61902',
 'CHEBI:22750',
 'CHEBI:35348',
 'CHEBI:48030',
 'CHEBI:2571',
 'CHEBI:38131',
 'CHEBI:83575',
 'CHEBI:136889',
 'CHEBI:26250',
 'CHEBI:36244',
 'CHEBI:23906',
 'CHEBI:38261',
 'CHEBI:22916',
 'CHEBI:35924',
 'CHEBI:24689',
 'CHEBI:32877',
 'CHEBI:50511',
 'CHEBI:26588',
 'CHEBI:24385',
 'CHEBI:5653',
 'CHEBI:48591',
 'CHEBI:38295',
 'CHEBI:58944',
 'CHEBI:134396',
 'CHEBI:49172',
 'CHEBI:26558',
 'CHEBI:64708',
 'CHEBI:35923',
 'CHEBI:25961',
 'CHEBI:47779',
 'CHEBI:46812',
 'CHEBI:37863',
 'CHEBI:22718',
 'CHEBI:36562',
 'CHEBI:38771',
 'CHEBI:36078',
 'CHEBI:26935',
 'CHEBI:33555',
 'CHEBI:23044',
 'CHEBI:15693',
 'CHEBI:33892',
 'CHEBI:33909',
 'CHEBI:35766',
 'CHEBI:51149',
 'CHEBI:35972',
 'CHEBI:38304',
 'CHEBI:46942',
 'CHEBI:24026',
 'CHEBI:33721',
 'CHEBI:38093',
 'CHEBI:38830',
 'CHEBI:26875',
 'CHEBI:37963',
 'CHEBI:61910',
 'CHEBI:47891',
 'CHEBI:74818',
 'CHEBI:50401',
 'CHEBI:24834',
 'CHEBI:33299',
 'CHEBI:63424',
 'CHEBI:63427',
 'CHEBI:15841',
 'CHEBI:33666',
 'CHEBI:26214',
 'CHEBI:22484',
 'CHEBI:27024',
 'CHEBI:46845',
 'CHEBI:64365',
 'CHEBI:63566',
 'CHEBI:38757',
 'CHEBI:83264',
 'CHEBI:24867',
 'CHEBI:37841',
 'CHEBI:33720',
 'CHEBI:36885',
 'CHEBI:59412',
 'CHEBI:64612',
 'CHEBI:36500',
 'CHEBI:37015',
 'CHEBI:84135',
 'CHEBI:51751',
 'CHEBI:18133',
 'CHEBI:57613',
 'CHEBI:38976',
 'CHEBI:25810',
 'CHEBI:24873',
 'CHEBI:35571',
 'CHEBI:83812',
 'CHEBI:37909',
 'CHEBI:51750',
 'CHEBI:15889',
 'CHEBI:48470',
 'CHEBI:24676',
 'CHEBI:22480',
 'CHEBI:139051',
 'CHEBI:23252',
 'CHEBI:51454',
 'CHEBI:88061',
 'CHEBI:46874',
 'CHEBI:38338',
 'CHEBI:62618',
 'CHEBI:59266',
 'CHEBI:84403',
 'CHEBI:27116',
 'CHEBI:77632',
 'CHEBI:38418',
 'CHEBI:35213',
 'CHEBI:35496',
 'CHEBI:78799',
 'CHEBI:38314',
 'CHEBI:35568',
 'CHEBI:35573',
 'CHEBI:33847',
 'CHEBI:16038',
 'CHEBI:33741',
 'CHEBI:33654',
 'CHEBI:17387',
 'CHEBI:33572',
 'CHEBI:36233',
 'CHEBI:22297',
 'CHEBI:23990',
 'CHEBI:38102',
 'CHEBI:24436',
 'CHEBI:35189',
 'CHEBI:79202',
 'CHEBI:68489',
 'CHEBI:18254',
 'CHEBI:78189',
 'CHEBI:47019',
 'CHEBI:61655',
 'CHEBI:24373',
 'CHEBI:26347',
 'CHEBI:36709',
 'CHEBI:73539',
 'CHEBI:35507',
 'CHEBI:35293',
 'CHEBI:140326',
 'CHEBI:46668',
 'CHEBI:17188',
 'CHEBI:61109',
 'CHEBI:35819',
 'CHEBI:33744',
 'CHEBI:73474',
 'CHEBI:134361',
 'CHEBI:33238',
 'CHEBI:26766',
 'CHEBI:17517',
 'CHEBI:25508',
 'CHEBI:22580',
 'CHEBI:26394',
 'CHEBI:35356',
 'CHEBI:50918',
 'CHEBI:24860',
 'CHEBI:2468',
 'CHEBI:33581',
 'CHEBI:26519',
 'CHEBI:37948',
 'CHEBI:33823',
 'CHEBI:59554',
 'CHEBI:46848',
 'CHEBI:24897',
 'CHEBI:26893',
 'CHEBI:63394',
 'CHEBI:29348',
 'CHEBI:35790',
 'CHEBI:25241',
 'CHEBI:58958',
 'CHEBI:24397',
 'CHEBI:25413',
 'CHEBI:24302',
 'CHEBI:46850',
 'CHEBI:51867',
 'CHEBI:35314',
 'CHEBI:50893',
 'CHEBI:36130',
 'CHEBI:33558',
 'CHEBI:24782',
 'CHEBI:36087',
 'CHEBI:26649',
 'CHEBI:47923',
 'CHEBI:33184',
 'CHEBI:23643',
 'CHEBI:25985',
 'CHEBI:33257',
 'CHEBI:61355',
 'CHEBI:24697',
 'CHEBI:36838',
 'CHEBI:23451',
 'CHEBI:33242',
 'CHEBI:26872',
 'CHEBI:50523',
 'CHEBI:16701',
 'CHEBI:36699',
 'CHEBI:35505',
 'CHEBI:24360',
 'CHEBI:59737',
 'CHEBI:26455',
 'CHEBI:51285',
 'CHEBI:35504',
 'CHEBI:36309',
 'CHEBI:33554',
 'CHEBI:47909',
 'CHEBI:50858',
 'CHEBI:53339',
 'CHEBI:25609',
 'CHEBI:23665',
 'CHEBI:35902',
 'CHEBI:35552',
 'CHEBI:139592',
 'CHEBI:35724',
 'CHEBI:38337',
 'CHEBI:35241',
 'CHEBI:29075',
 'CHEBI:62941',
 'CHEBI:140345',
 'CHEBI:59769',
 'CHEBI:28863',
 'CHEBI:47882',
 'CHEBI:35903',
 'CHEBI:33641',
 'CHEBI:47784',
 'CHEBI:23079',
 'CHEBI:25036',
 'CHEBI:50018',
 'CHEBI:28874',
 'CHEBI:35276',
 'CHEBI:26764',
 'CHEBI:65323',
 'CHEBI:51276',
 'CHEBI:37022',
 'CHEBI:22478',
 'CHEBI:23449',
 'CHEBI:72823',
 'CHEBI:63567',
 'CHEBI:50753',
 'CHEBI:38785',
 'CHEBI:46952',
 'CHEBI:36914',
 'CHEBI:33653',
 'CHEBI:62937',
 'CHEBI:36315',
 'CHEBI:37667',
 'CHEBI:38835',
 'CHEBI:35315',
 'CHEBI:33551',
 'CHEBI:18154',
 'CHEBI:79346',
 'CHEBI:26932',
 'CHEBI:39203',
 'CHEBI:25235',
 'CHEBI:23003',
 'CHEBI:64583',
 'CHEBI:46955',
 'CHEBI:33658',
 'CHEBI:59202',
 'CHEBI:28892',
 'CHEBI:33599',
 'CHEBI:33259',
 'CHEBI:64611',
 'CHEBI:37947',
 'CHEBI:65321',
 'CHEBI:63571',
 'CHEBI:25830',
 'CHEBI:50492',
 'CHEBI:26961',
 'CHEBI:33482',
 'CHEBI:63436',
 'CHEBI:47017',
 'CHEBI:51681',
 'CHEBI:48901',
 'CHEBI:52575',
 'CHEBI:35683',
 'CHEBI:24353',
 'CHEBI:61778',
 'CHEBI:13248',
 'CHEBI:35990',
 'CHEBI:33485',
 'CHEBI:35871',
 'CHEBI:27933',
 'CHEBI:27136',
 'CHEBI:26407',
 'CHEBI:33566',
 'CHEBI:47880',
 'CHEBI:24921',
 'CHEBI:38077',
 'CHEBI:48975',
 'CHEBI:59835',
 'CHEBI:83273',
 'CHEBI:22562',
 'CHEBI:33838',
 'CHEBI:35627',
 'CHEBI:51614',
 'CHEBI:36836',
 'CHEBI:63423',
 'CHEBI:22331',
 'CHEBI:25529',
 'CHEBI:36314',
 'CHEBI:83822',
 'CHEBI:38164',
 'CHEBI:51006',
 'CHEBI:28965',
 'CHEBI:38716',
 'CHEBI:76567',
 'CHEBI:35381',
 'CHEBI:51269',
 'CHEBI:37141',
 'CHEBI:25872',
 'CHEBI:36526',
 'CHEBI:51702',
 'CHEBI:25106',
 'CHEBI:51737',
 'CHEBI:38672',
 'CHEBI:36132',
 'CHEBI:38700',
 'CHEBI:25558',
 'CHEBI:17855',
 'CHEBI:18946',
 'CHEBI:83565',
 'CHEBI:15705',
 'CHEBI:35186',
 'CHEBI:33694',
 'CHEBI:36711',
 'CHEBI:23403',
 'CHEBI:35238',
 'CHEBI:36807',
 'CHEBI:47788',
 'CHEBI:24531',
 'CHEBI:33663',
 'CHEBI:22715',
 'CHEBI:57560',
 'CHEBI:38163',
 'CHEBI:23899',
 'CHEBI:50994',
 'CHEBI:26776',
 'CHEBI:51569',
 'CHEBI:35259',
 'CHEBI:77636',
 'CHEBI:35727',
 'CHEBI:35786',
 'CHEBI:24780',
 'CHEBI:26714',
 'CHEBI:26712',
 'CHEBI:26819',
 'CHEBI:63944',
 'CHEBI:36520',
 'CHEBI:25409',
 'CHEBI:22928',
 'CHEBI:23824',
 'CHEBI:79020',
 'CHEBI:26605',
 'CHEBI:139588',
 'CHEBI:52396',
 'CHEBI:37668',
 'CHEBI:50995',
 'CHEBI:52395',
 'CHEBI:61777',
 'CHEBI:38445',
 'CHEBI:24698',
 'CHEBI:63551',
 'CHEBI:35693',
 'CHEBI:83403',
 'CHEBI:36094',
 'CHEBI:35479',
 'CHEBI:25704',
 'CHEBI:25754',
 'CHEBI:38958',
 'CHEBI:21731',
 'CHEBI:23697',
 'CHEBI:38260',
 'CHEBI:33861',
 'CHEBI:22485',
 'CHEBI:2580',
 'CHEBI:18379',
 'CHEBI:23424',
 'CHEBI:33296',
 'CHEBI:37554',
 'CHEBI:33839',
 'CHEBI:36054',
 'CHEBI:23232',
 'CHEBI:18035',
 'CHEBI:63353',
 'CHEBI:23114',
 'CHEBI:76578',
 'CHEBI:26208',
 'CHEBI:32955',
 'CHEBI:24922',
 'CHEBI:36141',
 'CHEBI:24043',
 'CHEBI:35692',
 'CHEBI:46867',
 'CHEBI:38530',
 'CHEBI:24654',
 'CHEBI:38032',
 'CHEBI:26820',
 'CHEBI:35789',
 'CHEBI:62732',
 'CHEBI:26912',
 'CHEBI:22160',
 'CHEBI:26410',
 'CHEBI:36059',
 'CHEBI:51069',
 'CHEBI:33570',
 'CHEBI:24129',
 'CHEBI:37826',
 'CHEBI:16385',
 'CHEBI:26822',
 'CHEBI:46761',
 'CHEBI:83925',
 'CHEBI:25248',
 'CHEBI:37581',
 'CHEBI:35748',
 'CHEBI:26195',
 'CHEBI:33958',
 'CHEBI:58342',
 'CHEBI:17478',
 'CHEBI:36834',
 'CHEBI:25513',
 'CHEBI:57643',
 'CHEBI:38298',
 'CHEBI:64482',
 'CHEBI:33240',
 'CHEBI:47622',
 'CHEBI:33704',
 'CHEBI:83820',
 'CHEBI:33676',
 'CHEBI:32952',
 'CHEBI:131927',
 'CHEBI:26188',
 'CHEBI:35716',
 'CHEBI:28963',
 'CHEBI:22798',
 'CHEBI:60980',
 'CHEBI:17984',
 'CHEBI:37240',
 'CHEBI:28868',
 'CHEBI:27208',
 'CHEBI:15904',
 'CHEBI:35715',
 'CHEBI:22251',
 'CHEBI:61078',
 'CHEBI:61079',
 'CHEBI:58946',
 'CHEBI:37123',
 'CHEBI:33497',
 'CHEBI:50699',
 'CHEBI:22475',
 'CHEBI:35436']

JCI_500_COLUMNS_INT = sorted([int(n.split(":")[-1]) for n in JCI_500_COLUMNS])
