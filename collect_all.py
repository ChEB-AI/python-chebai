import fastobo
import networkx as nx
import pickle
import os
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from sklearn.metrics import f1_score
from model import Molecule
from typing import Iterable
from rdkit import Chem
from torch_geometric.nn import GATConv
from torch_geometric.utils.convert import from_networkx

def load_parts():
    cache_file = ".part_data.pkl"
    if not os.path.isfile(cache_file):
        doc = fastobo.load("chebi.obo")
        elements = list()
        for clause in doc:
            callback = CALLBACKS.get(type(clause))
            if callback is not None:
                elements.append(callback(clause))

        g = nx.DiGraph()
        for n in elements:
            g.add_node(n["id"], **n)
        g.add_edges_from([(p, q["id"]) for q in elements for p in q["parents"]])

        pass_parts(g, "CHEBI:23367", set())
        part_cache = dict()
        data = [
            (l, r)
            for l, r in extract_children(g, "CHEBI:23367", part_cache=part_cache)
            if r
        ]
        parts = list({p for l, r in data for p in r})
        train_parts, test_parts = train_test_split(parts, test_size=0.1, shuffle=True)
        tr, te = train_test_split(data, test_size=0.3, shuffle=True)
        te, val = train_test_split(te, test_size=0.3, shuffle=True)
        d = ((
                    unfold_dataset(tr, train_parts),
                    unfold_dataset(val, train_parts),
                    unfold_dataset(te, train_parts),
                    unfold_dataset(te, test_parts),
                ), part_cache)
        pickle.dump(d, open(cache_file, "wb"))
        return d
    else:
        return pickle.load(open(cache_file, "rb"))


def unfold_dataset(data, parts):
    return data #[(l, p, p in r) for l, r in data for p in parts]


def pass_parts(d: nx.DiGraph, root, parts=None):
    if parts is None:
        parts = set()
    parts = set(parts.union(d.nodes[root]["has_part"]))
    nx.set_node_attributes(d, {root: parts}, "has_part")
    for child in d.successors(root):
        pass_parts(d, child, set(parts))


def extract_children(d: nx.DiGraph, root, part_cache):
    smiles = d.nodes[root]["smiles"]
    if smiles:
        try:
            ident = add_to_cache(d, root, part_cache)
        except ValueError as e:
            print(e)
        else:
            ps = set()
            for part in d.nodes[root]["has_part"]:
                psmiles = d.nodes[part]["smiles"]
                if psmiles:
                    try:
                        ps.add(add_to_cache(d, part, part_cache))
                    except ValueError as e:
                        print(e)
            yield ident, ps
    for child in d.successors(root):
        for r in extract_children(d, child, part_cache):
            yield r


def add_to_cache(d, node, cache):
    n = d.nodes[node]
    s = n["id"]
    ident = int(s[s.index(":") + 1 :])
    if ident not in cache:
        mol = Chem.MolFromSmiles(n["smiles"])
        if mol is None:
            raise ValueError("Could not process molecule: %s"%n)
        cache[ident] = (s, n["smiles"], n["name"], mol_to_data(mol))
    return ident


def term_callback(doc):
    parts = set()
    parents = []
    name = None
    smiles = None
    for clause in doc:
        if isinstance(clause, fastobo.term.PropertyValueClause):
            t = clause.property_value
            if str(t.relation) == "http://purl.obolibrary.org/obo/chebi/smiles":
                smiles = t.value
        elif isinstance(clause, fastobo.term.RelationshipClause):
            if str(clause.typedef) == "has_part":
                parts.add(str(clause.term))
        elif isinstance(clause, fastobo.term.IsAClause):
            parents.append(str(clause.term))
        elif isinstance(clause, fastobo.term.NameClause):
            name = str(clause.name)
    return {
        "id": str(doc.id),
        "parents": parents,
        "has_part": parts,
        "name": name,
        "smiles": smiles,
    }


CALLBACKS = {
    fastobo.term.TermFrame: term_callback,
}


class PartOfNet(nn.Module):

    def __init__(self, in_length, loops=10):
        super().__init__()
        self.loops=loops
        self.left_graph_net = GATConv(in_length, in_length)
        self.right_graph_net = GATConv(in_length, in_length)
        self.output_net = nn.Sequential(nn.Linear(2*in_length,in_length*in_length), nn.Linear(in_length*in_length,in_length), nn.Linear(in_length,1))

    def forward(self, l, r):
        a = self.left_graph_net(l.x, l.edge_index.long())
        b = self.right_graph_net(r.x, r.edge_index.long())
        return self.output_net(torch.cat([torch.sum(a,dim=0),torch.sum(b,dim=0)], dim=0))

def mol_to_data(mol):
    graph = nx.Graph()
    for i in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(i)
        graph.add_node(i, x=[float(i == atom.GetAtomicNum()) for i in range(1,120)])
        for neighbour in atom.GetNeighbors():
            neighbour_idx = neighbour.GetIdx()
            bond = mol.GetBondBetweenAtoms(i, neighbour_idx)
            graph.add_edge(i, neighbour_idx, edge_x=int(bond.GetBondType()))
    return from_networkx(graph)

def train(data, cache, all_parts):
    floss = torch.nn.BCEWithLogitsLoss()
    net = PartOfNet(119)
    if torch.cuda.is_available():
        net.to("cuda:0")
    optimizer = torch.optim.Adam(net.parameters())
    for cidx, parts in data:
        c = cache[cidx][3]
        loss = 0
        for pidx in all_parts:
            optimizer.zero_grad()
            p = cache[pidx][3]
            label = torch.tensor([float(pidx in parts)])
            pred = net(c, p)
            loss += floss(pred, label)
        print(loss.item())
        loss.backward()
        optimizer.step()

if __name__ == "__main__":
    (train_data, validation_data, _, _), cache = load_parts()

    all_parts = {p for _,ps in train_data for p in ps}
    train(train_data, cache, all_parts)