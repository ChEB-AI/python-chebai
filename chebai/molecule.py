from __future__ import absolute_import, division

import logging
from typing import Any, List, Optional, Union

try:
    from rdkit import Chem
except ModuleNotFoundError:
    pass

import networkx as nx
import numpy as np
import torch

logger = logging.getLogger(__name__)


class Molecule:
    """
    Represents a molecule and provides methods for graph creation and feature extraction.

    Attributes:
        max_number_of_parents (int): Maximum number of parent atoms for any given atom.
        smile (str): SMILES representation of the molecule.
        logp (Optional[float]): LogP (partition coefficient) of the molecule.
        atoms (list): List of atom objects.
        no_of_atoms (int): Number of atoms in the molecule.
        graph (nx.Graph): Graph representation of the molecule.
        directed_graphs (np.ndarray): Directed graphs for each atom.
        dag_to_node (dict): Mapping of atom index to its directed acyclic graph (DAG).

    Methods:
        __init__(self, smile, logp=None, contract_rings=False):
            Initializes a Molecule object from SMILES representation.
        create_directed_graphs(self):
            Creates directed graphs (DAGs) for each atom in the molecule.
        create_feature_vectors(self):
            Creates feature vectors based on local environments of atoms.
        get_cycle(self):
            Attempts to find a cycle in the molecule graph.
        collect_atom_features(self):
            Collects atom features into a dictionary.
        get_atom_features(self, node_id):
            Retrieves features of a specific atom.
        get_bond_features(self, node1, node2):
            Retrieves features of a specific bond between two atoms.

    Static Methods:
        atom_features(atom):
            Computes features of an atom based on its properties.
        atom_features_of_contract_rings(degree):
            Computes features of an atom under contract rings condition.
        bond_features_between_contract_rings():
            Computes features of a bond between contracted rings.
        bond_features(bond):
            Computes features of a bond based on its type and properties.
        num_of_features():
            Computes total number of features in the feature vector.
        one_of_k_encoding(x, allowable_set):
            Encodes categorical input into a one-hot vector.
        one_of_k_encoding_unk(x, allowable_set):
            Encodes input into a one-hot vector with an 'unknown' handling.

    """

    max_number_of_parents = 7

    def __init__(
        self, smile: str, logp: Optional[float] = None, contract_rings: bool = False
    ):
        """
        Initializes a Molecule object.

        Args:
            smile (str): SMILES representation of the molecule.
            logp (float, optional): LogP (partition coefficient) of the molecule.
            contract_rings (bool, optional): Whether to contract rings during initialization.
        """
        self.smile = smile
        self.logp = logp
        # logger.info("Parsing Molecule {:},contract rings: {:}".format(smile, contract_rings))
        self.atoms = []

        m = Chem.MolFromSmiles(smile)
        # Chem.Kekulize(self.m)

        self.no_of_atoms = m.GetNumAtoms()
        self.graph = nx.Graph()

        for i in range(self.no_of_atoms):
            atom = m.GetAtomWithIdx(i)
            self.graph.add_node(
                i, attr_dict={"atom_features": Molecule.atom_features(atom)}
            )
            for neighbour in atom.GetNeighbors():
                neighbour_idx = neighbour.GetIdx()
                bond = m.GetBondBetweenAtoms(i, neighbour_idx)
                self.graph.add_edge(
                    i,
                    neighbour_idx,
                    attr_dict={"bond_features": Molecule.bond_features(bond)},
                )

        self.create_directed_graphs()
        # self.create_feature_vectors()

    def create_directed_graphs(self):
        """
        Creates directed graphs (DAGs) for each atom in the molecule.
        """
        self.directed_graphs = np.empty(
            (self.no_of_atoms, self.no_of_atoms - 1, 3), dtype=int
        )

        self.dag_to_node = {}

        # parse all the atoms one by one and get directed graph to that atom
        # as the sink node
        for idx in self.graph.nodes:
            # get shortest path from the root to all the other atoms and then reverse the edges.
            path = nx.single_source_dijkstra_path(self.graph, idx)
            G = nx.DiGraph()
            for i in self.graph.nodes:
                temp = path[i]
                temp.reverse()
                nx.add_path(G, temp)
            self.dag_to_node[idx] = G
            break

    def create_feature_vectors(self):
        """
        Creates feature vectors based on local environments of atoms.

        Note:
            create a three-dimensional matrix I, such that I_{i,j} is the local input vector for jth vertex in ith DAG
        """
        length_of_bond_features = Molecule.num_bond_features()
        length_of_atom_features = Molecule.num_atom_features()

        self.local_input_vector = np.zeros(
            (self.no_of_atoms, self.no_of_atoms, Molecule.num_of_features())
        )

        for idx in range(self.no_of_atoms):
            sorted_path = self.directed_graphs[idx, :, :]

            self.local_input_vector[idx, idx, :length_of_atom_features] = (
                self.get_atom_features(idx)
            )

            no_of_incoming_edges = {}
            for i in range(self.no_of_atoms - 1):
                node1 = sorted_path[i, 0]
                node2 = sorted_path[i, 1]

                self.local_input_vector[idx, node1, :length_of_atom_features] = (
                    self.get_atom_features(node1)
                )

                if node2 in no_of_incoming_edges:
                    index = no_of_incoming_edges[node2]
                    no_of_incoming_edges[node2] += 1
                    if index >= Molecule.max_number_of_parents:
                        continue
                else:
                    index = 0
                    no_of_incoming_edges[node2] = 1

                start = length_of_atom_features + index * length_of_bond_features
                end = start + length_of_bond_features

                self.local_input_vector[idx, node2, start:end] = self.get_bond_features(
                    node1, node2
                )

    def get_cycle(self):
        """
        Attempts to find a cycle in the molecule graph.

        Returns:
            list: List of edges forming a cycle in the graph.
        """
        try:
            return nx.find_cycle(self.graph)
        except:
            return []

    def collect_atom_features(self):
        """
        Collects atom features into a dictionary.
        """
        self.af = {
            node_id: torch.tensor(
                self.graph.nodes[node_id]["attr_dict"]["atom_features"]
            ).float()
            for node_id in range(self.no_of_atoms)
        }

    def get_atom_features(self, node_id: int) -> torch.Tensor:
        """
        Retrieves features of a specific atom.

        Args:
            node_id (int): Index of the atom.

        Returns:
            torch.Tensor: Tensor representing atom features.
        """
        return self.af[node_id]

    def get_bond_features(self, node1: int, node2: int) -> np.ndarray:
        """
        Retrieves features of a specific bond between two atoms.

        Args:
            node1 (int): Index of the first atom.
            node2 (int): Index of the second atom.

        Returns:
            np.ndarray: Array representing bond features.
        """
        attrs = self.graph.get_edge_data(node1, node2)
        return attrs["bond_features"]

    @staticmethod
    def atom_features(atom: Any) -> np.ndarray:
        """
        Computes features of an atom based on its properties.

        Args:
            atom (Chem.Atom): RDKit Atom object.

        Returns:
            np.ndarray: Array representing atom features.
        """
        return np.array(
            Molecule.one_of_k_encoding_unk(
                atom.GetSymbol(),
                [
                    "C",
                    "N",
                    "O",
                    "S",
                    "F",
                    "Si",
                    "P",
                    "Cl",
                    "Br",
                    "Mg",
                    "Na",
                    "Ca",
                    "Fe",
                    "As",
                    "Al",
                    "I",
                    "B",
                    "V",
                    "K",
                    "Tl",
                    "Yb",
                    "Sb",
                    "Sn",
                    "Ag",
                    "Pd",
                    "Co",
                    "Se",
                    "Ti",
                    "Zn",
                    "H",  # H?
                    "Li",
                    "Ge",
                    "Cu",
                    "Au",
                    "Ni",
                    "Cd",
                    "In",
                    "Mn",
                    "Zr",
                    "Cr",
                    "Pt",
                    "Hg",
                    "Pb",
                    "Unknown",
                ],
            )
            + Molecule.one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5])
            + Molecule.one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
            + Molecule.one_of_k_encoding_unk(
                atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5]
            )
            + [atom.GetIsAromatic()]
        )

    @staticmethod
    def atom_features_of_contract_rings(degree: int) -> np.ndarray:
        """
        Computes features of an atom under contract rings condition.

        Args:
            degree (int): Degree of the atom.

        Returns:
            np.ndarray: Array representing atom features under contract rings.
        """
        return np.array(
            Molecule.one_of_k_encoding_unk(
                "Unknown",
                [
                    "C",
                    "N",
                    "O",
                    "S",
                    "F",
                    "Si",
                    "P",
                    "Cl",
                    "Br",
                    "Mg",
                    "Na",
                    "Ca",
                    "Fe",
                    "As",
                    "Al",
                    "I",
                    "B",
                    "V",
                    "K",
                    "Tl",
                    "Yb",
                    "Sb",
                    "Sn",
                    "Ag",
                    "Pd",
                    "Co",
                    "Se",
                    "Ti",
                    "Zn",
                    "H",  # H?
                    "Li",
                    "Ge",
                    "Cu",
                    "Au",
                    "Ni",
                    "Cd",
                    "In",
                    "Mn",
                    "Zr",
                    "Cr",
                    "Pt",
                    "Hg",
                    "Pb",
                    "Unknown",
                ],
            )
            + Molecule.one_of_k_encoding(degree, [0, 1, 2, 3, 4, 5])
            + Molecule.one_of_k_encoding_unk(0, [0, 1, 2, 3, 4])
            + Molecule.one_of_k_encoding_unk(0, [0, 1, 2, 3, 4, 5])
            + [0]
        )

    @staticmethod
    def bond_features_between_contract_rings() -> np.ndarray:
        """
        Computes features of a bond between contracted rings.

        Returns:
            np.ndarray: Array representing bond features between contracted rings.
        """
        return np.array([1, 0, 0, 0, 0, 0])

    @staticmethod
    def bond_features(bond) -> np.ndarray:
        """
        Computes features of a bond based on its type and properties.

        Args:
            bond (Chem.Bond): RDKit Bond object.

        Returns:
            np.ndarray: Array representing bond features.
        """
        bt = bond.GetBondType()
        return np.array(
            [
                bt == Chem.rdchem.BondType.SINGLE,
                bt == Chem.rdchem.BondType.DOUBLE,
                bt == Chem.rdchem.BondType.TRIPLE,
                bt == Chem.rdchem.BondType.AROMATIC,
                bond.GetIsConjugated(),
                bond.IsInRing(),
            ]
        )

    @staticmethod
    def num_of_features() -> int:
        """
        Computes total number of features in the feature vector.

        Returns:
            int: Total number of features.
        """
        return (
            Molecule.max_number_of_parents * Molecule.num_bond_features()
            + Molecule.num_atom_features()
        )

    @staticmethod
    def one_of_k_encoding(x: Union[int, str], allowable_set: List) -> List:
        """
        Encodes categorical input into a one-hot vector.

        Args:
            x: Input value to encode.
            allowable_set (list): List of allowable values.

        Returns:
            list: One-hot encoded vector.
        """
        if x not in allowable_set:
            raise Exception(
                "input {0} not in allowable set{1}:".format(x, allowable_set)
            )
        return list(map(lambda s: x == s, allowable_set))

    @staticmethod
    def one_of_k_encoding_unk(x: Union[int, str], allowable_set: List) -> List:
        """
        Encodes input into a one-hot vector with an 'unknown' handling.

        Note:
            Maps inputs not in the allowable set to the last element.

        Args:
            x: Input value to encode.
            allowable_set (list): List of allowable values.

        Returns:
            list: One-hot encoded vector.
        """
        if x not in allowable_set:
            x = allowable_set[-1]
        return list(map(lambda s: x == s, allowable_set))

    @staticmethod
    def num_atom_features() -> int:
        """
        Computes the length of the atom feature vector using a simple molecule.

        Returns:
            int: Length of atom feature vector.
        """
        m = Chem.MolFromSmiles("CC")
        alist = m.GetAtoms()
        a = alist[0]
        return len(Molecule.atom_features(a))

    @staticmethod
    def num_bond_features() -> int:
        """
        Computes the length of the bond feature vector using a simple molecule.

        Returns:
            int: Length of bond feature vector.
        """
        simple_mol = Chem.MolFromSmiles("CC")
        Chem.SanitizeMol(simple_mol)
        return len(Molecule.bond_features(simple_mol.GetBonds()[0]))


if __name__ == "__main__":
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format)
    logger = logging.getLogger(__name__)
