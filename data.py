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
from torch.utils.data import DataLoader
from torch_geometric.data import InMemoryDataset

from torch_geometric.utils.convert import from_networkx
from torch_geometric.data import Dataset as TGDataset, Data

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





class ClassificationData(TGDataset):

    def len(self):
        return self.set_lengths[self.split]

    def get(self, idx):
        return torch.load(os.path.join(self.processed_dir,f"{self.split}.{idx}.pt"))

    def __init__(self, root, split="train", **kwargs):
        self.split = split
        self.train_split = 0.8
        self.set_lengths = dict(test=196, train=976, validation=49)
        super().__init__(root, **kwargs)

    def download(self):
        url = 'http://purl.obolibrary.org/obo/chebi.obo'
        r = requests.get(url, allow_redirects=True)
        open(self.raw_paths[0], 'wb').write(r.content)

    @property
    def processed_file_names(self):
        return [f"{k}.{i}.pt" for k in ("train", "test", "validation") for i in range(self.set_lengths[k])]

    @property
    def raw_file_names(self):
        return ["chebi.obo"]

    def callback(self, term):
        return term_callback(term)

    def process(self):
        self.cache = []
        elements = [self.callback(clause) for clause in fastobo.load(self.raw_paths[0])]

        g = nx.DiGraph()
        for n in elements:
            g.add_node(n["id"], **n)
        g.add_edges_from([(p, q["id"]) for q in elements for p in q["parents"]])
        g = nx.transitive_closure_dag(g)
        fixed_nodes = list(g.nodes)
        random.shuffle(fixed_nodes)
        train_split, test_split = train_test_split(fixed_nodes, train_size=self.train_split, shuffle=True)
        test_split, validation_split = train_test_split(test_split, train_size=self.train_split, shuffle=True)
        smiles = nx.get_node_attributes(g, "smiles")
        print("Create graphs")
        for k, nodes in dict(train=train_split, test=test_split, validation=validation_split).items():
            counter = 0
            l = []
            for node in nodes:
                if smiles[node] is not None:
                    superclasses = set(g.predecessors(node))
                    labels = tuple(n in superclasses for n in fixed_nodes)
                    d = self.process_row(node, smiles[node], labels)
                    if d is not None and d.edge_index.shape[1] > 0:
                        l.append(d)
                    if len(l) > 100:
                        print(f"Save {k}.{counter}.pt")
                        torch.save(l, os.path.join(self.processed_dir, f"{k}.{counter}.pt"))
                        counter += 1
                        l = []
            if l:
                torch.save(l, os.path.join(self.processed_dir, f"{k}.{counter}.pt"))
        torch.save(self.cache, os.path.join(self.processed_dir, "embeddings.pt"))

    def process_row(self, iden, smiles, labels):
        d = self.mol_to_data(smiles)
        if d is not None:
            d["label"] = torch.tensor(labels).unsqueeze(0)
        else:
            print(f"Could not process {iden}: {smiles}")
        return d

    def mol_to_data(self, smiles):
        try:
            mol = ps.read_smiles(smiles)
        except:
            return None
        d = {}
        for node in mol.nodes:
            m = mol.nodes[node]
            try:
                x = self.cache.index(m)
            except ValueError:
                x = len(self.cache)
                self.cache.append(m.copy())
            d[node] = x
            for attr in list(mol.nodes[node].keys()):
                del mol.nodes[node][attr]
        nx.set_node_attributes(mol, d, "x")
        return from_networkx(mol)


class JCIClassificationData(InMemoryDataset):

    def __init__(self, root, split="train", **kwargs):
        self.split = split
        super().__init__(root, **kwargs)
        self.data, self.slices = torch.load(os.path.join(self.processed_dir,f"{split}.pkl"))

    @property
    def processed_file_names(self):
        return ["test.pkl", "train.pkl", "validation.pkl"]

    def download(self):
        pass

    @property
    def raw_file_names(self):
        return ["test.pkl", "train.pkl", "validation.pkl"]

    def process(self):
        self.cache = []
        for f in self.processed_file_names:
            structure = pickle.load(open(os.path.join(self.raw_dir,f), "rb"))
            s = structure.apply(self.process_row, axis=1)
            data, slices = self.collate([x for x in s if x is not None and x["edge_index"].size()[1]!=0])
            print("Could not process the following molecules", [x["Name"] for x in s if x is not None and x["edge_index"].size()[1]==0])
            torch.save((data, slices), os.path.join(self.processed_dir,f))
        torch.save(self.cache, os.path.join(self.processed_dir, "embeddings.pt"))

    def process_row(self, row):
        d = self.mol_to_data(row[1])
        if d is not None:
            d["label"] = torch.tensor(row[2:]).unsqueeze(0)
        else:
            print(f"Could not process {row[0]}: {row[1]}")
        return d


class JCIPureData(pl.LightningDataModule):

    def prepare_data(self, *args, **kwargs):
        if not all(os.path.isfile(os.path.join(self.processed_dir, f)) for f in self.processed_file_names):
            self.process()

    def setup(self, stage: Optional[str] = None):
        pass

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(torch.load(os.path.join(self.processed_dir, "train.pkl")), collate_fn=lambda x:x)

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(torch.load(os.path.join(self.processed_dir, "validation.pkl")), collate_fn=lambda x:x)

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(torch.load(os.path.join(self.processed_dir, "test.pkl")), collate_fn=lambda x:x)

    def __init__(self,**kwargs):
        root = os.path.join("data", "JCI_pure")
        self.processed_dir = os.path.join(root,"processed")
        self.raw_dir = os.path.join(root, "raw")
        self.cache = list()
        super().__init__(root, **kwargs)

    @property
    def processed_file_names(self):
        return ["test.pkl", "train.pkl", "validation.pkl"]

    def download(self):
        pass

    @property
    def raw_file_names(self):
        return ["test.pkl", "train.pkl", "validation.pkl"]

    def process(self):
        self.cache = []
        for f in self.processed_file_names:
            structure = pickle.load(open(os.path.join(self.raw_dir,f), "rb"))
            l = structure.apply(self.process_row, axis=1)
            torch.save(list(l), os.path.join(self.processed_dir,f))
        torch.save(self.cache, os.path.join(self.processed_dir, "embeddings.pt"))

    def process_row(self, row):
        return self.mol_to_data(row[1]), torch.tensor(row[2:])

    def mol_to_data(self, smiles):
        try:
            mol = ps.read_smiles(smiles)
        except:
            return None
        d = {}
        for node in mol.nodes:
            m = mol.nodes[node]
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
        return mol

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