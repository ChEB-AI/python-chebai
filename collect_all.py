import fastobo
import networkx as nx
import pickle
import os
from sklearn.model_selection import train_test_split
import torch
from torch import nn
import requests

try:
    from rdkit import Chem
except:
    pass
import multiprocessing as mp
from torch_geometric import nn as tgnn
from torch_geometric.utils.convert import from_networkx
from torch_geometric.data import InMemoryDataset, Data, DataLoader


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


class PartOfData(InMemoryDataset):

    def transform(self, ppd: PrePairData):
        return PairData(ppd, self.graph)

    def __init__(self, root, transform=None, pre_transform=None):
        self.cache_file = ".part_data.pkl"
        self._ignore = set()
        super().__init__(root, self.transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.graph = torch.load(os.path.join(self.processed_dir, self.processed_cache_names[0]))

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
        self.pass_parts(g, "CHEBI:23367", set())
        print("Load data")
        children = list(nx.single_source_shortest_path(g, "CHEBI:23367").keys())[:100]
        parts = list({p for c in children for p in g.nodes[c]["has_part"]})
        print("Create molecules")
        with mp.Pool(1) as p:
            nx.set_node_attributes(g, dict(p.map(get_mol_enc,((g,i) for i in (children + parts)))), "enc")

        # Filter invalid structures
        children = [p for p in children if g.nodes[p]["enc"]]
        parts = [p for p in parts if g.nodes[p]["enc"]]

        data_list = [PrePairData(l, r, float(r in g.nodes[l]["has_part"])) for l in children for r in parts]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        torch.save(g, os.path.join(self.processed_dir, self.processed_cache_names[0]))

    @property
    def raw_file_names(self):
        return ["chebi.obo"]

    @property
    def processed_file_names(self):
        return ["data.pt"]

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

    def add_to_cache(self, d, node, cache):
        n = d.nodes[node]
        s = n["id"]
        ident = int(s[s.index(":") + 1:])
        if ident not in self._ignore:
            if ident not in cache:
                mol = Chem.MolFromSmiles(n["smiles"])
                if mol is None:
                    self._ignore.add(ident)
                    return None
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
        self.left_graph_net = tgnn.GATConv(in_length, in_length)
        self.right_graph_net = tgnn.GATConv(in_length, in_length)
        self.output_net = nn.Sequential(nn.Linear(2*in_length,in_length*in_length), nn.Linear(in_length*in_length,in_length), nn.Linear(in_length,1))

    def forward(self, x):
        a = self.left_graph_net(x.x_s, x.edge_index_s.long())
        b = self.right_graph_net(x.x_t, x.edge_index_t.long())
        return self.output_net(torch.cat([torch.sum(a,dim=0),torch.sum(b,dim=0)], dim=0))


def get_mol_enc(x):
    g, i = x
    s = g.nodes[i]["smiles"]
    return i, mol_to_data(s) if s else None

def mol_to_data(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    else:
        graph = nx.Graph()
        for i in range(mol.GetNumAtoms()):
            atom = mol.GetAtomWithIdx(i)
            graph.add_node(i, x=[float(i == atom.GetAtomicNum()) for i in range(1,120)])
            for neighbour in atom.GetNeighbors():
                neighbour_idx = neighbour.GetIdx()
                bond = mol.GetBondBetweenAtoms(i, neighbour_idx)
                graph.add_edge(i, neighbour_idx, edge_x=int(bond.GetBondType()))
        return from_networkx(graph)


def train(dataset):
    floss = torch.nn.BCEWithLogitsLoss()
    net = PartOfNet(119)
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters())
    for epoch in range(10):
        running_loss = 0
        batches = 0
        for data in dataset:
            data.to(device)
            optimizer.zero_grad()
            pred = net(data)
            loss = floss(pred, data.label)
            running_loss += loss.item()
            batches += 1
            loss.backward()
            optimizer.step()
        print("Epoch", epoch, "loss =", running_loss/batches)
    torch.save(net,"net.pt")


if __name__ == "__main__":
    data = PartOfData(".")
    loader = DataLoader(data)#, follow_batch=["x_s", "x_t", "edge_index_s", "edge_index_t"])
    train(loader)