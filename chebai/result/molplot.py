from pysmiles.read_smiles import *
from rdkit import Chem
import networkx as nx
from networkx.algorithms.isomorphism import GraphMatcher
from rdkit.Chem.Draw import rdMolDraw2D
import torch
from pysmiles.read_smiles import _tokenize
from chebai.result.base import ResultProcessor

from chebai.preprocessing.datasets import JCI_500_COLUMNS_INT, JCI_500_COLUMNS
import svgutils.compose as sc
from matplotlib import pyplot as plt, cm, colors
from matplotlib.image import AxesImage, imread
from tempfile import NamedTemporaryFile
import numpy as np
from matplotlib import rc
from os import makedirs

class AttentionOnMoleculesProcessor(ResultProcessor):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.wanted_classes = (37141, 24873, 33555, 28874, 22693)

    def start(self):
        self.counter = 0
        for w in self.wanted_classes:
            makedirs(f"/tmp/plots/{w}", exist_ok=True)

    @classmethod
    def _identifier(cls):
        return "platt"

    def process_prediction(self, raw_features, raw_labels, features, labels, pred):
        if any(True for (ident, match) in zip(JCI_500_COLUMNS_INT, labels) if
               match and ident in self.wanted_classes):
            atts = torch.stack(pred["attentions"]).squeeze(1).detach().numpy()
            try:
                self.plot_attentions(raw_features, np.max(np.max(atts, axis=2), axis=1), 0.4, [ident for present, ident in zip(labels, JCI_500_COLUMNS_INT) if present and ident in self.wanted_classes])
            except StopIteration:
                print("Could not match", raw_features)
            except NoRDMolException:
                pass

    def plot_attentions(self, smiles, attention, threshold, wanted_names):
        pmol = self.read_smiles_with_index(smiles)
        rdmol = Chem.MolFromSmiles(smiles)
        if not rdmol:
            raise NoRDMolException
        rdmolx = self.mol_to_nx(rdmol)
        gm = GraphMatcher(pmol, rdmolx)
        iso = next(gm.isomorphisms_iter())
        token_to_node_map = {pmol.nodes[node]["token_index"]: iso[node] for node in pmol.nodes}
        d = rdMolDraw2D.MolDraw2DCairo(500, 500)
        cmap = cm.ScalarMappable(cmap=cm.Greens)
        attention_colors = cmap.to_rgba(attention, norm=False)
        aggr_attention_colors = cmap.to_rgba(np.max(attention[2:,:], axis=0), norm=False)
        cols = {
            token_to_node_map[token_index]: tuple(aggr_attention_colors[token_index].tolist()) for
            node, token_index in
            nx.get_node_attributes(pmol, "token_index").items()}
        highlight_atoms = [token_to_node_map[token_index] for node, token_index in
            nx.get_node_attributes(pmol, "token_index").items()]
        rdMolDraw2D.PrepareAndDrawMolecule(d, rdmol,
                                           highlightAtoms=highlight_atoms,
                                           highlightAtomColors=cols)

        d.FinishDrawing()

        num_tokens = sum(1 for _ in _tokenize(smiles))

        fig = plt.figure(figsize=(15,15),facecolor='w')

        rc('font',**{'family':'monospace', 'monospace': 'DejaVu Sans Mono'})
        fig.tight_layout()

        ax2, ax = fig.subplots(2, 1, gridspec_kw={'height_ratios': [10, 1]})

        with NamedTemporaryFile(mode="wt",suffix=".png") as svg1:
            d.WriteDrawingText(svg1.name)
            ax2.imshow(imread(svg1.name))
        ax2.axis('off')
        ax2.spines['left'].set_position('center')
        ax2.spines['bottom'].set_position('zero')
        ax2.autoscale(tight=True)

        table = plt.table(cellText=[[t for _, t in _tokenize(smiles)] for _ in range(attention.shape[0])],
                               cellColours=attention_colors,cellLoc='center')
        table.auto_set_column_width(list(range(num_tokens)))
        table.scale(1, 4)
        table.set_fontsize(26)

        ax.add_table(table)
        ax.axis('off')
        ax.spines['top'].set_position('zero')
        ax.autoscale(tight=True)

        self.counter += 1
        for w in wanted_names:
            fig.savefig(f'/tmp/plots/{w}/{self.counter}.png', transparent=False, bbox_inches='tight', pad_inches=0)
        plt.close()

    @staticmethod
    def mol_to_nx(mol):
        G = nx.Graph()

        for atom in mol.GetAtoms():
            G.add_node(atom.GetIdx(),
                       atomic_num=atom.GetAtomicNum(),
                       formal_charge=atom.GetFormalCharge(),
                       chiral_tag=atom.GetChiralTag(),
                       hybridization=atom.GetHybridization(),
                       num_explicit_hs=atom.GetNumExplicitHs(),
                       is_aromatic=atom.GetIsAromatic())
        for bond in mol.GetBonds():
            G.add_edge(bond.GetBeginAtomIdx(),
                       bond.GetEndAtomIdx(),
                       bond_type=bond.GetBondType())
        return G

    @staticmethod
    def read_smiles_with_index(smiles, explicit_hydrogen=False, zero_order_bonds=True,
                               reinterpret_aromatic=True):
        """
        This is just a re-implementation of pysmiles.read_smiles, that stores token indices
        """
        bond_to_order = {'-': 1, '=': 2, '#': 3, '$': 4, ':': 1.5, '.': 0}
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
                    raise ValueError('Previous bond (order {}) not used. '
                                     'Overwritten by "{}"'.format(next_bond, token))
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
                        raise ValueError('Conflicting bond orders for ring '
                                         'between indices {}'.format(token))
                    # idx is the index of the *next* atom we're adding. So: -1.
                    if mol.has_edge(idx - 1, jdx):
                        raise ValueError('Edge specified by marker {} already '
                                         'exists'.format(token))
                    if idx - 1 == jdx:
                        raise ValueError('Marker {} specifies a bond between an '
                                         'atom and itself'.format(token))
                    if next_bond or zero_order_bonds:
                        mol.add_edge(idx - 1, jdx, order=next_bond)
                    next_bond = None
                    del ring_nums[token]
                else:
                    if idx == 0:
                        raise ValueError("Can't have a marker ({}) before an atom"
                                         "".format(token))
                    # idx is the index of the *next* atom we're adding. So: -1.
                    ring_nums[token] = (idx - 1, next_bond)
                    next_bond = None
            elif tokentype == TokenType.EZSTEREO:
                LOGGER.warning('E/Z stereochemical information, which is specified by "%s", will be discarded', token)
        if ring_nums:
            raise KeyError('Unmatched ring indices {}'.format(list(ring_nums.keys())))

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
            if mol.nodes[n_idx].get('aromatic', False):
                raise ValueError("You specified an aromatic atom outside of a"
                                 " ring. This is impossible")

        mark_aromatic_edges(mol)
        fill_valence(mol)
        if reinterpret_aromatic:
            mark_aromatic_atoms(mol)
            mark_aromatic_edges(mol)
            for idx, jdx in mol.edges:
                if ((not mol.nodes[idx].get('aromatic', False) or
                     not mol.nodes[jdx].get('aromatic', False))
                        and mol.edges[idx, jdx].get('order', 1) == 1.5):
                    mol.edges[idx, jdx]['order'] = 1

        if explicit_hydrogen:
            add_explicit_hydrogens(mol)
        else:
            remove_explicit_hydrogens(mol)
        return mol


class NoRDMolException(Exception):
    pass
