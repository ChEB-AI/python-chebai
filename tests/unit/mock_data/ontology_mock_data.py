from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Dict, List, Set, Tuple

import networkx as nx
import pandas as pd


class MockOntologyGraphData(ABC):
    """
    Abstract base class for mocking ontology graph data.

    This class provides a set of static methods that must be implemented by subclasses
    to return various elements of an ontology graph such as nodes, edges, and dataframes.
    """

    @staticmethod
    @abstractmethod
    def get_nodes() -> List[int]:
        """
        Get a list of node IDs in the ontology graph.

        Returns:
            List[int]: A list of node IDs.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_number_of_nodes() -> int:
        """
        Get the number of nodes in the ontology graph.

        Returns:
            int: The total number of nodes.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_edges() -> Set[Tuple[int, int]]:
        """
        Get the set of edges in the ontology graph.

        Returns:
            Set[Tuple[int, int]]: A set of tuples where each tuple represents an edge between two nodes.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_number_of_edges() -> int:
        """
        Get the number of edges in the ontology graph.

        Returns:
            int: The total number of edges.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_edges_of_transitive_closure_graph() -> Set[Tuple[int, int]]:
        """
        Get the set of edges in the transitive closure of the ontology graph.

        Returns:
            Set[Tuple[int, int]]: A set of tuples representing the transitive closure edges.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_number_of_transitive_edges() -> int:
        """
        Get the number of edges in the transitive closure of the ontology graph.

        Returns:
            int: The total number of transitive edges.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_obsolete_nodes_ids() -> Set[int]:
        """
        Get the set of obsolete node IDs in the ontology graph.

        Returns:
            Set[int]: A set of obsolete node IDs.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_transitively_closed_graph() -> nx.DiGraph:
        """
        Get the transitive closure of the ontology graph.

        Returns:
            nx.DiGraph: A directed graph representing the transitive closure of the ontology graph.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_data_in_dataframe() -> pd.DataFrame:
        """
        Get the ontology data as a Pandas DataFrame.

        Returns:
            pd.DataFrame: A DataFrame containing ontology data.
        """
        pass


class ChebiMockOntology(MockOntologyGraphData):
    """
    A mock ontology representing a simplified ChEBI (Chemical Entities of Biological Interest) structure.
    This class is used for testing purposes and includes nodes and edges representing chemical compounds
    and their relationships in a graph structure.

    Nodes:
    - CHEBI:12345 (Compound A)
    - CHEBI:54321 (Compound B)
    - CHEBI:67890 (Compound C)
    - CHEBI:11111 (Compound D)
    - CHEBI:22222 (Compound E)
    - CHEBI:99999 (Compound F)
    - CHEBI:77533 (Compound G, Obsolete node)
    - CHEBI:77564 (Compound H, Obsolete node)
    - CHEBI:88888 (Compound I)

    Valid Edges:
    - CHEBI:54321 -> CHEBI:12345
    - CHEBI:67890 -> CHEBI:12345
    - CHEBI:67890 -> CHEBI:88888
    - CHEBI:11111 -> CHEBI:54321
    - CHEBI:22222 -> CHEBI:67890
    - CHEBI:12345 -> CHEBI:99999

    The class also includes methods to retrieve nodes, edges, and transitive closure of the graph.

    Visual Representation Graph with Valid Nodes and Edges:

                                       22222
                                        /
                       11111         67890
                         \\         /  \
                        54321     /    88888
                           \\   /
                           12345
                             \
                            99999
    """

    @staticmethod
    def get_nodes() -> List[int]:
        """
        Get the set of valid node IDs in the mock ontology.

        Returns:
        - Set[int]: A set of integers representing the valid ChEBI node IDs.
        """
        return [11111, 12345, 22222, 54321, 67890, 88888, 99999]

    @staticmethod
    def get_number_of_nodes() -> int:
        """
        Get the number of valid nodes in the mock ontology.

        Returns:
        - int: The number of valid nodes.
        """
        return len(ChebiMockOntology.get_nodes())

    @staticmethod
    def get_edges() -> Set[Tuple[int, int]]:
        """
        Get the set of valid edges in the mock ontology.

        Returns:
        - Set[Tuple[int, int]]: A set of tuples representing the directed edges
          between ChEBI nodes.
        """
        return {
            (54321, 12345),
            (67890, 12345),
            (67890, 88888),
            (11111, 54321),
            (22222, 67890),
            (12345, 99999),
        }

    @staticmethod
    def get_number_of_edges() -> int:
        """
        Get the number of valid edges in the mock ontology.

        Returns:
        - int: The number of valid edges.
        """
        return len(ChebiMockOntology.get_edges())

    @staticmethod
    def get_edges_of_transitive_closure_graph() -> Set[Tuple[int, int]]:
        """
        Get the set of edges derived from the transitive closure of the mock ontology graph.

        Returns:
        - Set[Tuple[int, int]]: A set of tuples representing the directed edges
          in the transitive closure of the ChEBI graph.
        """
        return {
            (54321, 12345),
            (54321, 99999),
            (67890, 12345),
            (67890, 99999),
            (67890, 88888),
            (11111, 54321),
            (11111, 12345),
            (11111, 99999),
            (22222, 67890),
            (22222, 12345),
            (22222, 99999),
            (22222, 88888),
            (12345, 99999),
        }

    @staticmethod
    def get_number_of_transitive_edges() -> int:
        """
        Get the number of edges in the transitive closure of the mock ontology graph.

        Returns:
        - int: The number of edges in the transitive closure graph.
        """
        return len(ChebiMockOntology.get_edges_of_transitive_closure_graph())

    @staticmethod
    def get_obsolete_nodes_ids() -> Set[int]:
        """
        Get the set of obsolete node IDs in the mock ontology.

        Returns:
        - Set[int]: A set of integers representing the obsolete ChEBI node IDs.
        """
        return {77533, 77564}

    @staticmethod
    def get_raw_data() -> str:
        """
        Get the raw data representing the mock ontology in OBO format.

        Returns:
        - str: A string containing the raw OBO data for the mock ChEBI terms.
        """
        return """
        [Term]
        id: CHEBI:12345
        name: Compound A
        subset: 2_STAR
        property_value: http://purl.obolibrary.org/obo/chebi/formula "C26H35ClN4O6S" xsd:string
        property_value: http://purl.obolibrary.org/obo/chebi/charge "0" xsd:string
        property_value: http://purl.obolibrary.org/obo/chebi/monoisotopicmass "566.19658" xsd:string
        property_value: http://purl.obolibrary.org/obo/chebi/mass "567.099" xsd:string
        property_value: http://purl.obolibrary.org/obo/chebi/inchikey "ROXPMFGZZQEKHB-IUKKYPGJSA-N" xsd:string
        property_value: http://purl.obolibrary.org/obo/chebi/smiles "C1=CC=CC=C1" xsd:string
        property_value: http://purl.obolibrary.org/obo/chebi/inchi "InChI=1S/C26H35ClN4O6S/c1-16(2)28-26(34)30(5)14-23-17(3)13-31(18(4)15-32)25(33)21-7-6-8-22(24(21)37-23)29-38(35,36)20-11-9-19(27)10-12-20/h6-12,16-18,23,29,32H,13-15H2,1-5H3,(H,28,34)/t17-,18-,23+/m0/s1" xsd:string
        xref: LINCS:LSM-20139
        is_a: CHEBI:54321
        is_a: CHEBI:67890

        [Term]
        id: CHEBI:54321
        name: Compound B
        property_value: http://purl.obolibrary.org/obo/chebi/smiles "C1=CC=CC=C1O" xsd:string
        is_a: CHEBI:11111
        is_a: CHEBI:77564

        [Term]
        id: CHEBI:67890
        name: Compound C
        property_value: http://purl.obolibrary.org/obo/chebi/smiles "C1=CC=CC=C1N" xsd:string
        is_a: CHEBI:22222

        [Term]
        id: CHEBI:11111
        name: Compound D
        property_value: http://purl.obolibrary.org/obo/chebi/smiles "C1=CC=CC=C1F" xsd:string

        [Term]
        id: CHEBI:22222
        name: Compound E
        property_value: http://purl.obolibrary.org/obo/chebi/smiles "C1=CC=CC=C1Cl" xsd:string

        [Term]
        id: CHEBI:99999
        name: Compound F
        property_value: http://purl.obolibrary.org/obo/chebi/smiles "C1=CC=CC=C1Br" xsd:string
        is_a: CHEBI:12345

        [Term]
        id: CHEBI:77533
        name: Compound G
        is_a: CHEBI:99999
        property_value: http://purl.obolibrary.org/obo/chebi/smiles "C1=C1Br" xsd:string
        is_obsolete: true

        [Term]
        id: CHEBI:77564
        name: Compound H
        property_value: http://purl.obolibrary.org/obo/chebi/smiles "CC=C1Br" xsd:string
        is_obsolete: true

        [Typedef]
        id: has_major_microspecies_at_pH_7_3
        name: has major microspecies at pH 7.3
        is_cyclic: true
        is_transitive: false

        [Term]
        id: CHEBI:88888
        name: Compound I
        property_value: http://purl.obolibrary.org/obo/chebi/smiles "C1=CC=CC=C1[Mg+]" xsd:string
        is_a: CHEBI:67890
        """

    @staticmethod
    def get_data_in_dataframe() -> pd.DataFrame:
        data = OrderedDict(
            id=[
                12345,
                54321,
                67890,
                11111,
                22222,
                99999,
                88888,
            ],
            name=[
                "Compound A",
                "Compound B",
                "Compound C",
                "Compound D",
                "Compound E",
                "Compound F",
                "Compound I",
            ],
            SMILES=[
                "C1=CC=CC=C1",
                "C1=CC=CC=C1O",
                "C1=CC=CC=C1N",
                "C1=CC=CC=C1F",
                "C1=CC=CC=C1Cl",
                "C1=CC=CC=C1Br",
                "C1=CC=CC=C1[Mg+]",
            ],
            **{
                # -row- [12345, 54321, 67890, 11111, 22222, 99999, 88888]
                11111: [True, True, False, True, False, True, False],
                12345: [True, False, False, False, False, True, False],
                22222: [True, False, True, False, True, True, True],
                54321: [True, True, False, False, False, True, False],
                67890: [True, False, True, False, False, True, True],
                88888: [False, False, False, False, False, False, True],
                99999: [False, False, False, False, False, True, False],
            },
        )

        data_df = pd.DataFrame(data)

        # ------------- Code Approach -------
        # ancestors_of_nodes = {}
        # for parent, child in ChebiMockOntology.get_edges_of_transitive_closure_graph():
        #     if child not in ancestors_of_nodes:
        #         ancestors_of_nodes[child] = set()
        #     if parent not in ancestors_of_nodes:
        #         ancestors_of_nodes[parent] = set()
        #     ancestors_of_nodes[child].add(parent)
        #     ancestors_of_nodes[child].add(child)
        #
        # # For each node in the ontology, create a column to check if it's an ancestor of any other node or itself
        # for node in ChebiMockOntology.get_nodes():
        #     data_df[node] = data_df['id'].apply(
        #         lambda x: (x == node) or (node in ancestors_of_nodes[x])
        #     )

        return data_df

    @staticmethod
    def get_transitively_closed_graph() -> nx.DiGraph:
        """
        Create a directed graph, compute its transitive closure, and return it.

        Returns:
            g (nx.DiGraph): A transitively closed directed graph.
        """
        g = nx.DiGraph()

        for node in ChebiMockOntology.get_nodes():
            g.add_node(node, **{"smiles": "test_smiles_placeholder"})

        g.add_edges_from(ChebiMockOntology.get_edges_of_transitive_closure_graph())

        return g
