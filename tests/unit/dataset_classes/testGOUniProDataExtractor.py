import unittest
from collections import OrderedDict
from unittest.mock import PropertyMock, mock_open, patch

import fastobo
import networkx as nx
import pandas as pd

from chebai.preprocessing.datasets.deepGO.go_uniprot import _GOUniProtDataExtractor
from chebai.preprocessing.reader import ProteinDataReader
from tests.unit.mock_data.ontology_mock_data import GOUniProtMockData


class TestGOUniProtDataExtractor(unittest.TestCase):
    """
    Unit tests for the _GOUniProtDataExtractor class.
    """

    @classmethod
    @patch.multiple(_GOUniProtDataExtractor, __abstractmethods__=frozenset())
    @patch.object(_GOUniProtDataExtractor, "base_dir", new_callable=PropertyMock)
    @patch.object(_GOUniProtDataExtractor, "_name", new_callable=PropertyMock)
    @patch("os.makedirs", return_value=None)
    def setUpClass(
        cls,
        mock_makedirs,
        mock_name_property: PropertyMock,
        mock_base_dir_property: PropertyMock,
    ) -> None:
        """
        Class setup for mocking abstract properties of _GOUniProtDataExtractor.
        """
        mock_base_dir_property.return_value = "MockedBaseDirPropGOUniProtDataExtractor"
        mock_name_property.return_value = "MockedNamePropGOUniProtDataExtractor"

        _GOUniProtDataExtractor.READER = ProteinDataReader

        cls.extractor = _GOUniProtDataExtractor()

    def test_term_callback(self) -> None:
        """
        Test the term_callback method for correct parsing and filtering of GO terms.
        """
        self.extractor.go_branch = "all"
        term_mapping = {}
        for term in fastobo.loads(GOUniProtMockData.get_GO_raw_data()):
            if isinstance(term, fastobo.typedef.TypedefFrame):
                continue
            term_mapping[self.extractor._parse_go_id(term.id)] = term

        # Test individual term callback
        term_dict = self.extractor.term_callback(term_mapping[4])
        expected_dict = {"go_id": 4, "parents": [3, 2], "name": "GO_4"}
        self.assertEqual(
            term_dict,
            expected_dict,
            "The term_callback did not return the expected dictionary.",
        )

        # Test filtering valid terms
        valid_terms_docs = set()
        for term_id, term_doc in term_mapping.items():
            if self.extractor.term_callback(term_doc):
                valid_terms_docs.add(term_id)

        self.assertEqual(
            valid_terms_docs,
            set(GOUniProtMockData.get_nodes()),
            "The valid terms do not match expected nodes.",
        )

        # Test that obsolete terms are filtered out
        self.assertFalse(
            any(
                self.extractor.term_callback(term_mapping[obs_id])
                for obs_id in GOUniProtMockData.get_obsolete_nodes_ids()
            ),
            "Obsolete terms should not be present.",
        )

        # Test filtering by GO branch (e.g., BP)
        self.extractor.go_branch = "BP"
        BP_terms = {
            term_id
            for term_id, term in term_mapping.items()
            if self.extractor.term_callback(term)
        }
        self.assertEqual(
            BP_terms, {2, 4}, "The BP terms do not match the expected set."
        )

    @patch(
        "fastobo.load", return_value=fastobo.loads(GOUniProtMockData.get_GO_raw_data())
    )
    def test_extract_class_hierarchy(self, mock_load) -> None:
        """
        Test the extraction of the class hierarchy from the ontology.
        """
        graph = self.extractor._extract_class_hierarchy("fake_path")

        # Validate the graph structure
        self.assertIsInstance(
            graph, nx.DiGraph, "The result should be a directed graph."
        )

        # Check nodes
        actual_nodes = set(graph.nodes)
        self.assertEqual(
            set(GOUniProtMockData.get_nodes()),
            actual_nodes,
            "The graph nodes do not match the expected nodes.",
        )

        # Check edges
        actual_edges = set(graph.edges)
        self.assertEqual(
            GOUniProtMockData.get_edges_of_transitive_closure_graph(),
            actual_edges,
            "The graph edges do not match the expected edges.",
        )

        # Check number of nodes and edges
        self.assertEqual(
            GOUniProtMockData.get_number_of_nodes(),
            len(actual_nodes),
            "The number of nodes should match the actual number of nodes in the graph.",
        )

        self.assertEqual(
            GOUniProtMockData.get_number_of_transitive_edges(),
            len(actual_edges),
            "The number of transitive edges should match the actual number of transitive edges in the graph.",
        )

    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data=GOUniProtMockData.get_UniProt_raw_data(),
    )
    def test_get_swiss_to_go_mapping(self, mock_open) -> None:
        """
        Test the extraction of SwissProt to GO term mapping.
        """
        mapping_df = self.extractor._get_swiss_to_go_mapping()
        expected_df = pd.DataFrame(
            OrderedDict(
                swiss_id=["Swiss_Prot_1", "Swiss_Prot_2"],
                accession=["Q6GZX4", "DCGZX4"],
                go_ids=[[2, 3, 5], [2, 5]],
                sequence=list(GOUniProtMockData.protein_sequences().values()),
            )
        )

        pd.testing.assert_frame_equal(
            mapping_df,
            expected_df,
            obj="The SwissProt to GO mapping DataFrame does not match the expected DataFrame.",
        )

    @patch(
        "fastobo.load", return_value=fastobo.loads(GOUniProtMockData.get_GO_raw_data())
    )
    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data=GOUniProtMockData.get_UniProt_raw_data(),
    )
    @patch.object(
        _GOUniProtDataExtractor,
        "select_classes",
        return_value=GOUniProtMockData.get_nodes(),
    )
    def test_graph_to_raw_dataset(
        self, mock_select_classes, mock_open, mock_load
    ) -> None:
        """
        Test the conversion of the class hierarchy graph to a raw dataset.
        """
        graph = self.extractor._extract_class_hierarchy("fake_path")
        actual_df = self.extractor._graph_to_raw_dataset(graph)
        expected_df = GOUniProtMockData.get_data_in_dataframe()

        pd.testing.assert_frame_equal(
            actual_df,
            expected_df,
            obj="The raw dataset DataFrame does not match the expected DataFrame.",
        )

    @patch("builtins.open", new_callable=mock_open, read_data=b"Mocktestdata")
    @patch("pandas.read_pickle")
    def test_load_dict(
        self, mock_read_pickle: PropertyMock, mock_open: mock_open
    ) -> None:
        """
        Test the loading of the dictionary from a DataFrame.
        """
        mock_df = GOUniProtMockData.get_data_in_dataframe()
        mock_read_pickle.return_value = mock_df

        generator = self.extractor._load_dict("data/tests")
        result = list(generator)

        # Convert NumPy arrays to lists for comparison
        for item in result:
            item["labels"] = list(item["labels"])

        # Expected output for comparison
        expected_result = [
            {
                "features": mock_df["sequence"][0],
                "labels": mock_df.iloc[0, 4:].to_list(),
                "ident": mock_df["swiss_id"][0],
            },
            {
                "features": mock_df["sequence"][1],
                "labels": mock_df.iloc[1, 4:].to_list(),
                "ident": mock_df["swiss_id"][1],
            },
        ]

        self.assertEqual(
            result,
            expected_result,
            "The loaded dictionary does not match the expected structure.",
        )


if __name__ == "__main__":
    unittest.main()
