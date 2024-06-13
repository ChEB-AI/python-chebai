import abc
from os import makedirs
from tempfile import NamedTemporaryFile

import networkx as nx
import numpy as np
import pandas as pd
import torch
from matplotlib import cm, colors
from matplotlib import pyplot as plt
from matplotlib import rc
from matplotlib.image import AxesImage, imread
from networkx.algorithms.isomorphism import GraphMatcher
from pysmiles.read_smiles import *
from pysmiles.read_smiles import _tokenize
from rdkit import Chem
from rdkit.Chem.Draw import MolToMPL, rdMolDraw2D

from chebai.preprocessing.datasets import JCI_500_COLUMNS, JCI_500_COLUMNS_INT
from chebai.result.base import ResultProcessor


class AttentionMolPlot:
    def draw_attention_molecule(self, smiles, attention):
        pmol = self.read_smiles_with_index(smiles)
        rdmol = Chem.MolFromSmiles(smiles)
        if not rdmol:
            raise NoRDMolException
        rdmolx = self.mol_to_nx(rdmol)
        gm = GraphMatcher(pmol, rdmolx)
        iso = next(gm.isomorphisms_iter())
        token_to_node_map = {
            pmol.nodes[node]["token_index"]: iso[node] for node in pmol.nodes
        }
        d = rdMolDraw2D.MolDraw2DCairo(500, 500)
        cmap = cm.ScalarMappable(cmap=cm.Greens)

        aggr_attention_colors = cmap.to_rgba(
            np.max(attention[2:, :], axis=0), norm=False
        )
        cols = {
            token_to_node_map[token_index]: tuple(
                aggr_attention_colors[token_index].tolist()
            )
            for node, token_index in nx.get_node_attributes(pmol, "token_index").items()
        }
        highlight_atoms = [
            token_to_node_map[token_index]
            for node, token_index in nx.get_node_attributes(pmol, "token_index").items()
        ]
        rdMolDraw2D.PrepareAndDrawMolecule(
            d, rdmol, highlightAtoms=highlight_atoms, highlightAtomColors=cols
        )

        d.FinishDrawing()
        return d

    def plot_attentions(self, smiles, attention, threshold, labels):
        d = self.draw_attention_molecule(smiles, attention)
        cmap = cm.ScalarMappable(cmap=cm.Greens)
        attention_colors = cmap.to_rgba(attention, norm=False)
        num_tokens = sum(1 for _ in _tokenize(smiles))

        fig = plt.figure(figsize=(15, 15), facecolor="w")

        rc("font", **{"family": "monospace", "monospace": "DejaVu Sans Mono"})
        fig.tight_layout()

        ax2, ax = fig.subplots(2, 1, gridspec_kw={"height_ratios": [10, 1]})

        with NamedTemporaryFile(mode="wt", suffix=".png") as svg1:
            d.WriteDrawingText(svg1.name)
            ax2.imshow(imread(svg1.name))
        ax2.axis("off")
        ax2.spines["left"].set_position("center")
        ax2.spines["bottom"].set_position("zero")
        ax2.autoscale(tight=True)

        table = plt.table(
            cellText=[
                (["[CLS]"] + [t for _, t in _tokenize(smiles)])
                for _ in range(attention.shape[0])
            ],
            cellColours=attention_colors,
            cellLoc="center",
        )
        table.auto_set_column_width(list(range(num_tokens)))
        table.scale(1, 4)
        table.set_fontsize(26)

        ax.add_table(table)
        ax.axis("off")
        ax.spines["top"].set_position("zero")
        ax.autoscale(tight=True)

        self.counter += 1
        for w, label, predicted in labels:
            if predicted:
                cat = "p"
            else:
                cat = "n"
            if predicted == label:
                cat = "t" + cat
            else:
                cat = "f" + cat
            fig.savefig(
                f"/tmp/plots/{w}/{cat}_{self.counter}.png",
                transparent=False,
                bbox_inches="tight",
                pad_inches=0,
            )
        plt.close()

    @staticmethod
    def mol_to_nx(mol):
        G = nx.Graph()

        for atom in mol.GetAtoms():
            G.add_node(
                atom.GetIdx(),
                atomic_num=atom.GetAtomicNum(),
                formal_charge=atom.GetFormalCharge(),
                chiral_tag=atom.GetChiralTag(),
                hybridization=atom.GetHybridization(),
                num_explicit_hs=atom.GetNumExplicitHs(),
                is_aromatic=atom.GetIsAromatic(),
            )
        for bond in mol.GetBonds():
            G.add_edge(
                bond.GetBeginAtomIdx(),
                bond.GetEndAtomIdx(),
                bond_type=bond.GetBondType(),
            )
        return G

    @staticmethod
    def read_smiles_with_index(
        smiles,
        explicit_hydrogen=False,
        zero_order_bonds=True,
        reinterpret_aromatic=True,
    ):
        """
        This is just a re-implementation of pysmiles.read_smiles, that stores token indices
        """
        bond_to_order = {"-": 1, "=": 2, "#": 3, "$": 4, ":": 1.5, ".": 0}
        mol = nx.Graph()
        anchor = None
        idx = 0
        default_bond = 1
        next_bond = None
        branches = []
        ring_nums = {}
        for token_index, (tokentype, token) in enumerate(_tokenize(smiles)):
            if tokentype == TokenType.ATOM:
                mol.add_node(idx, token_index=token_index, **parse_atom(token))
                if anchor is not None:
                    if next_bond is None:
                        next_bond = default_bond
                    if next_bond or zero_order_bonds:
                        mol.add_edge(anchor, idx, order=next_bond)
                    next_bond = None
                anchor = idx
                idx += 1
            elif tokentype == TokenType.BRANCH_START:
                branches.append(anchor)
            elif tokentype == TokenType.BRANCH_END:
                anchor = branches.pop()
            elif tokentype == TokenType.BOND_TYPE:
                if next_bond is not None:
                    raise ValueError(
                        "Previous bond (order {}) not used. "
                        'Overwritten by "{}"'.format(next_bond, token)
                    )
                next_bond = bond_to_order[token]
            elif tokentype == TokenType.RING_NUM:
                if token in ring_nums:
                    jdx, order = ring_nums[token]
                    if next_bond is None and order is None:
                        next_bond = default_bond
                    elif order is None:  # Note that the check is needed,
                        next_bond = next_bond  # But this could be pass.
                    elif next_bond is None:
                        next_bond = order
                    elif next_bond != order:  # Both are not None
                        raise ValueError(
                            "Conflicting bond orders for ring "
                            "between indices {}".format(token)
                        )
                    # idx is the index of the *next* atom we're adding. So: -1.
                    if mol.has_edge(idx - 1, jdx):
                        raise ValueError(
                            "Edge specified by marker {} already "
                            "exists".format(token)
                        )
                    if idx - 1 == jdx:
                        raise ValueError(
                            "Marker {} specifies a bond between an "
                            "atom and itself".format(token)
                        )
                    if next_bond or zero_order_bonds:
                        mol.add_edge(idx - 1, jdx, order=next_bond)
                    next_bond = None
                    del ring_nums[token]
                else:
                    if idx == 0:
                        raise ValueError(
                            "Can't have a marker ({}) before an atom" "".format(token)
                        )
                    # idx is the index of the *next* atom we're adding. So: -1.
                    ring_nums[token] = (idx - 1, next_bond)
                    next_bond = None
            elif tokentype == TokenType.EZSTEREO:
                LOGGER.warning(
                    'E/Z stereochemical information, which is specified by "%s", will be discarded',
                    token,
                )
        if ring_nums:
            raise KeyError("Unmatched ring indices {}".format(list(ring_nums.keys())))

        # Time to deal with aromaticity. This is a mess, because it's not super
        # clear what aromaticity information has been provided, and what should be
        # inferred. In addition, to what extend do we want to provide a "sane"
        # molecule, even if this overrides what the SMILES string specifies?
        cycles = nx.cycle_basis(mol)
        ring_idxs = set()
        for cycle in cycles:
            ring_idxs.update(cycle)
        non_ring_idxs = set(mol.nodes) - ring_idxs
        for n_idx in non_ring_idxs:
            if mol.nodes[n_idx].get("aromatic", False):
                raise ValueError(
                    "You specified an aromatic atom outside of a"
                    " ring. This is impossible"
                )

        mark_aromatic_edges(mol)
        fill_valence(mol)
        if reinterpret_aromatic:
            mark_aromatic_atoms(mol)
            mark_aromatic_edges(mol)
            for idx, jdx in mol.edges:
                if (
                    not mol.nodes[idx].get("aromatic", False)
                    or not mol.nodes[jdx].get("aromatic", False)
                ) and mol.edges[idx, jdx].get("order", 1) == 1.5:
                    mol.edges[idx, jdx]["order"] = 1

        if explicit_hydrogen:
            add_explicit_hydrogens(mol)
        else:
            remove_explicit_hydrogens(mol)
        return mol


class AttentionOnMoleculesProcessor(AttentionMolPlot, ResultProcessor):
    def __init__(self, *args, headers=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.headers = headers

    def start(self):
        self.counter = 0

    @classmethod
    def _identifier(cls):
        return "platt"

    def filter(self, l):
        return

    def process_prediction(
        self, proc_id, preds, raw_features, model_output, labels, **kwargs
    ):
        atts = torch.stack(model_output["attentions"]).squeeze(1).detach().numpy()
        predictions = preds.detach().numpy().squeeze(0) > 0.5
        if self.headers is None:
            headers = list(range(len(labels)))
        else:
            headers = self.headers

        for w in headers:
            makedirs(f"/tmp/plots/{w}", exist_ok=True)

        try:
            self.plot_attentions(
                raw_features,
                np.max(np.max(atts, axis=2), axis=1),
                0.4,
                [
                    (ident, label, predicted)
                    for label, ident, predicted in zip(labels, headers, predictions)
                    if (label or predicted)
                ],
            )
        except StopIteration:
            print("Could not match", raw_features)
        except NoRDMolException:
            pass


class LastLayerAttentionProcessor(AttentionMolPlot, ResultProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def start(self):
        self.counter = 0
        for w in JCI_500_COLUMNS_INT:
            makedirs(f"/tmp/plots/{w}", exist_ok=True)

    @classmethod
    def _identifier(cls):
        return "platt_last"

    def filter(self, l):
        return

    def process_prediction(self, raw_features, raw_labels, features, labels, pred):
        atts = torch.stack(pred["attentions"]).squeeze(1).detach().numpy()
        last_layer = np.max(atts, axis=2)[-1, :]
        if np.any(last_layer > 0.4):
            try:
                self.plot_attentions(
                    raw_features,
                    np.max(np.max(atts, axis=2), axis=1),
                    0.4,
                    [
                        ident
                        for present, ident in zip(labels, JCI_500_COLUMNS_INT)
                        if present
                    ],
                )
            except StopIteration:
                print("Could not match", raw_features)
            except NoRDMolException:
                pass


class SingletonAttentionProcessor(AttentionMolPlot, ResultProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def start(self):
        self.counter = 0
        for w in JCI_500_COLUMNS_INT:
            makedirs(f"/tmp/plots/{w}", exist_ok=True)

    @classmethod
    def _identifier(cls):
        return "platt_singles"

    def filter(self, l):
        return

    def process_prediction(self, raw_features, raw_labels, features, labels, pred):
        atts = torch.stack(pred["attentions"]).squeeze(1).detach().numpy()
        if sum(labels) == 1:
            try:
                predictions = (
                    torch.sigmoid(pred["logits"]).detach().numpy().squeeze(0) > 0.5
                )
                self.plot_attentions(
                    raw_features,
                    np.max(np.average(atts, axis=2), axis=1),
                    0.4,
                    [
                        (ident, label, predicted)
                        for label, ident, predicted in zip(
                            labels, JCI_500_COLUMNS_INT, predictions
                        )
                        if (label or predicted)
                    ],
                )
            except StopIteration:
                print("Could not match", raw_features)
            except NoRDMolException:
                pass


class AttentionNetwork(ResultProcessor):
    def __init__(self, *args, headers=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.headers = headers
        self.i = 0

    @classmethod
    def _identifier(cls):
        return "platt_table"

    def start(self):
        self.counter = 0

    def process_prediction(
        self,
        proc_id,
        preds,
        raw_features,
        model_output,
        labels,
        ident=None,
        threshold=0.5,
        **kwargs,
    ):
        if self.headers is None:
            headers = list(range(len(labels)))
        else:
            headers = self.headers

        for w in headers:
            makedirs(f"plots/{w}", exist_ok=True)

        atts = torch.stack(model_output["attentions"]).squeeze(1).detach().numpy()
        predictions = preds.detach().numpy().squeeze(0) > 0.5
        plt.rcParams.update({"font.size": 8})
        try:
            attentions = atts
            tokens = ["[CLS]"] + [s for _, s in _tokenize(raw_features)]
            cmap = cm.ScalarMappable(cmap=cm.Greens)
            assert len(tokens) == attentions.shape[2]

            rows = int((attentions.shape[1] + 2))
            width = len(tokens)
            height = 12
            rdmol = Chem.MolFromSmiles(raw_features)
            if rdmol is not None:
                fig0 = MolToMPL(rdmol, fitImage=True)
                fig0.text(
                    0.1,
                    0,
                    "annotated:"
                    + ", ".join(
                        str(l) for (l, is_member) in zip(headers, labels) if is_member
                    )
                    + "\n"
                    + "predicted:"
                    + ", ".join(
                        str(l)
                        for (l, is_member) in zip(headers, predictions)
                        if is_member
                    ),
                    fontdict=dict(fontsize=10),
                )
                fig0.savefig(
                    f"plots/mol_{ident}.png",
                    bbox_inches="tight",
                    pad_inches=0,
                )
                plt.close(fig0)
                fig = plt.figure(figsize=(10 * 12, width // 3))
                l_tokens = {i: str(t) for i, t in enumerate(tokens)}
                r_tokens = {(len(tokens) + i): str(t) for i, t in enumerate(tokens)}
                labels = dict(list(l_tokens.items()) + list(r_tokens.items()))
                edges = [(l, r) for r in r_tokens.keys() for l in l_tokens.keys()]
                g = nx.Graph()
                g.add_nodes_from(l_tokens, bipartite=0)
                g.add_nodes_from(r_tokens, bipartite=1)
                g.add_edges_from(edges)
                pos = np.array(
                    [(0, -i) for i in range(len(l_tokens))]
                    + [(1, -i) for i in range(len(l_tokens))]
                )

                offset = np.array(
                    [(1, 0) for i in range(len(l_tokens))]
                    + [(1, 0) for i in range(len(l_tokens))]
                )
                # axes = fig.subplots(1, 6 * 8 + 5, subplot_kw=dict(frameon=False))

                ax = fig.add_subplot(111)
                ax.axis("off")
                for layer in range(attentions.shape[0]):
                    for head in range(attentions.shape[1]):
                        index = 8 * (layer) + head + layer + 1

                        at = np.concatenate([a for a in attentions[layer, head]])
                        col = cmap.cmap(at)
                        col[:, 3] = at
                        nx.draw_networkx(
                            g,
                            pos=pos + (index * offset),
                            edge_color=col,
                            ax=ax,
                            labels=labels,
                            node_color="none",
                            node_size=8,
                        )
                        # sns.heatmap(attentions[i,j], linewidth=0.5, ax=ax, cmap=cm.Greens, square=True, vmin=0, vmax=1, xticklabels=tokens, yticklabels=tokens)
                fig.subplots_adjust()
                fig.savefig(
                    f"plots/att_{ident}.png",
                    # transparent=True,
                    bbox_inches="tight",
                    pad_inches=0,
                    dpi=100,
                )

            plt.close()
        except StopIteration:
            print("Could not match", raw_features)
        except NoRDMolException:
            pass
        finally:
            plt.close()


class NoRDMolException(Exception):
    pass
