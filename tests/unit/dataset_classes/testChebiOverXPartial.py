import unittest
from unittest.mock import mock_open, patch

import networkx as nx

from chebai.preprocessing.datasets.chebi import ChEBIOverXPartial
from tests.unit.mock_data.ontology_mock_data import ChebiMockOntology


class TestChEBIOverX(unittest.TestCase):

    @classmethod
    @patch.multiple(ChEBIOverXPartial, __abstractmethods__=frozenset())
    def setUpClass(cls) -> None:
        """
        Set up the ChEBIOverXPartial instance with a mock processed directory path and a test graph.
        """
        cls.chebi_extractor = ChEBIOverXPartial(top_class_id=11111, chebi_version=231)
        cls.test_graph = ChebiMockOntology.get_transitively_closed_graph()

    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data=ChebiMockOntology.get_raw_data(),
    )
    def test_extract_class_hierarchy(self, mock_open: mock_open) -> None:
        """
        Test the extraction of class hierarchy and validate the structure of the resulting graph.
        """
        # Mock the output of fastobo.loads
        self.chebi_extractor.top_class_id = 11111
        graph: nx.DiGraph = self.chebi_extractor.extract_class_hierarchy("fake_path")

        # Validate the graph structure
        self.assertIsInstance(
            graph, nx.DiGraph, "The result should be a directed graph."
        )

        # Check nodes
        expected_nodes = {11111, 54321, 12345, 99999}
        expected_edges = {
            (54321, 12345),
            (54321, 99999),
            (11111, 54321),
            (11111, 12345),
            (11111, 99999),
            (12345, 99999),
        }
        self.assertEqual(
            set(graph.nodes),
            expected_nodes,
            f"The graph nodes do not match the expected nodes for top class {self.chebi_extractor.top_class_id} hierarchy.",
        )

        # Check edges
        self.assertEqual(
            expected_edges,
            set(graph.edges),
            "The graph edges do not match the expected edges.",
        )

        # Check number of nodes and edges
        self.assertEqual(
            len(graph.nodes),
            len(expected_nodes),
            "The number of nodes should match the actual number of nodes in the graph.",
        )

        self.assertEqual(
            len(expected_edges),
            len(graph.edges),
            "The number of transitive edges should match the actual number of transitive edges in the graph.",
        )

        self.chebi_extractor.top_class_id = 22222
        graph = self.chebi_extractor.extract_class_hierarchy("fake_path")

        # Check nodes with top class as 22222
        self.assertEqual(
            set(graph.nodes),
            {67890, 88888, 12345, 99999, 22222},
            f"The graph nodes do not match the expected nodes for top class {self.chebi_extractor.top_class_id} hierarchy.",
        )

    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data=ChebiMockOntology.get_raw_data(),
    )
    def test_extract_class_hierarchy_with_bottom_cls(
        self, mock_open: mock_open
    ) -> None:
        """
        Test the extraction of class hierarchy and validate the structure of the resulting graph.
        """
        self.chebi_extractor.top_class_id = 88888
        graph: nx.DiGraph = self.chebi_extractor.extract_class_hierarchy("fake_path")

        # Check nodes with top class as 88888
        self.assertEqual(
            set(graph.nodes),
            {self.chebi_extractor.top_class_id},
            f"The graph nodes do not match the expected nodes for top class {self.chebi_extractor.top_class_id} hierarchy.",
        )


if __name__ == "__main__":
    unittest.main()
