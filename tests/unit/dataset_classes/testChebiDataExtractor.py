import unittest
from unittest.mock import MagicMock, PropertyMock, mock_open, patch

import networkx as nx
import pandas as pd

from chebai.preprocessing.datasets.chebi import _ChEBIDataExtractor
from tests.unit.mock_data.ontology_mock_data import ChebiMockOntology


class TestChEBIDataExtractor(unittest.TestCase):

    @classmethod
    @patch.multiple(_ChEBIDataExtractor, __abstractmethods__=frozenset())
    @patch.object(_ChEBIDataExtractor, "base_dir", new_callable=PropertyMock)
    @patch.object(_ChEBIDataExtractor, "_name", new_callable=PropertyMock)
    @patch("os.makedirs", return_value=None)
    def setUpClass(
        cls,
        mock_makedirs,
        mock_name_property: PropertyMock,
        mock_base_dir_property: PropertyMock,
    ) -> None:
        """
        Set up a base instance of _ChEBIDataExtractor for testing with mocked properties.
        """
        # Mocking properties
        mock_base_dir_property.return_value = "MockedBaseDirPropertyChebiDataExtractor"
        mock_name_property.return_value = "MockedNamePropertyChebiDataExtractor"

        # Mock Data Reader
        ReaderMock = MagicMock()
        ReaderMock.name.return_value = "MockedReader"
        _ChEBIDataExtractor.READER = ReaderMock

        # Create an instance of the dataset
        cls.extractor: _ChEBIDataExtractor = _ChEBIDataExtractor(
            chebi_version=231, chebi_version_train=200
        )

        # Mock instance for _chebi_version_train_obj
        mock_train_obj = MagicMock()
        mock_train_obj.processed_dir_main = "/mock/path/to/train"
        cls.extractor._chebi_version_train_obj = mock_train_obj

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
        graph = self.extractor._extract_class_hierarchy("fake_path")

        # Validate the graph structure
        self.assertIsInstance(
            graph, nx.DiGraph, "The result should be a directed graph."
        )

        # Check nodes
        actual_nodes = set(graph.nodes)
        self.assertEqual(
            set(ChebiMockOntology.get_nodes()),
            actual_nodes,
            "The graph nodes do not match the expected nodes.",
        )

        # Check edges
        actual_edges = set(graph.edges)
        self.assertEqual(
            ChebiMockOntology.get_edges_of_transitive_closure_graph(),
            actual_edges,
            "The graph edges do not match the expected edges.",
        )

        # Check number of nodes and edges
        self.assertEqual(
            ChebiMockOntology.get_number_of_nodes(),
            len(actual_nodes),
            "The number of nodes should match the actual number of nodes in the graph.",
        )

        self.assertEqual(
            ChebiMockOntology.get_number_of_transitive_edges(),
            len(actual_edges),
            "The number of transitive edges should match the actual number of transitive edges in the graph.",
        )

    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data=ChebiMockOntology.get_raw_data(),
    )
    @patch.object(
        _ChEBIDataExtractor,
        "select_classes",
        return_value=ChebiMockOntology.get_nodes(),
    )
    def test_graph_to_raw_dataset(
        self, mock_select_classes: PropertyMock, mock_open: mock_open
    ) -> None:
        """
        Test conversion of a graph to a raw dataset and compare it with the expected DataFrame.
        """
        graph = self.extractor._extract_class_hierarchy("fake_path")
        data_df = self.extractor._graph_to_raw_dataset(graph)

        pd.testing.assert_frame_equal(
            data_df,
            ChebiMockOntology.get_data_in_dataframe(),
            obj="The DataFrame should match the expected structure.",
        )

    @patch(
        "builtins.open", new_callable=mock_open, read_data=b"Mocktestdata"
    )  # Mocking open as a binary file
    @patch("pandas.read_pickle")
    def test_load_dict(
        self, mock_read_pickle: PropertyMock, mock_open: mock_open
    ) -> None:
        """
        Test loading data from a pickled file and verify the generator output.
        """
        # Mock the DataFrame returned by read_pickle
        mock_df = pd.DataFrame(
            {
                "id": [12345, 67890, 11111, 54321],  # Corrected ID
                "name": ["A", "B", "C", "D"],
                "SMILES": ["C1CCCCC1", "O=C=O", "C1CC=CC1", "C[Mg+]"],
                12345: [True, False, False, True],
                67890: [False, True, True, False],
                11111: [True, False, True, False],
            }
        )
        mock_read_pickle.return_value = mock_df

        generator = self.extractor._load_dict("data/tests")
        result = list(generator)

        # Convert NumPy arrays to lists for comparison
        for item in result:
            item["labels"] = list(item["labels"])

        # Expected output for comparison
        expected_result = [
            {"features": "C1CCCCC1", "labels": [True, False, True], "ident": 12345},
            {"features": "O=C=O", "labels": [False, True, False], "ident": 67890},
            {"features": "C1CC=CC1", "labels": [False, True, True], "ident": 11111},
            {"features": "C[Mg+]", "labels": [True, False, False], "ident": 54321},
        ]

        # Assert if the result matches the expected output
        self.assertEqual(
            result,
            expected_result,
            "The loaded dictionary should match the expected structure.",
        )

    @patch("builtins.open", new_callable=mock_open)
    @patch.object(_ChEBIDataExtractor, "processed_dir_main", new_callable=PropertyMock)
    def test_setup_pruned_test_set(
        self,
        mock_processed_dir_main: PropertyMock,
        mock_open_file: mock_open,
    ) -> None:
        """
        Test the pruning of the test set to match classes in the training set.
        """
        # Mock the content for the two open calls (original classes and new classes)
        mock_orig_classes = "12345\n67890\n88888\n54321\n77777\n"
        mock_new_classes = "12345\n67890\n99999\n77777\n"

        # Use side_effect to simulate the two different file reads
        mock_open_file.side_effect = [
            mock_open(
                read_data=mock_orig_classes
            ).return_value,  # First open() for orig_classes
            mock_open(
                read_data=mock_new_classes
            ).return_value,  # Second open() for new_classes
        ]

        # Mock the attributes used in the method
        mock_processed_dir_main.return_value = "/mock/path/to/current"

        # Mock DataFrame to simulate the test dataset
        mock_df = pd.DataFrame(
            {
                "labels": [
                    [
                        True,
                        False,
                        True,
                        False,
                        True,
                    ],  # First test instance labels (match orig_classes)
                    [False, True, False, True, False],
                ]  # Second test instance labels
            }
        )

        # Call the method under test
        pruned_df = self.extractor._setup_pruned_test_set(mock_df)

        # Expected DataFrame labels after pruning (only "12345", "67890", "77777", and "99999" remain)
        expected_labels = [[True, False, False, True], [False, True, False, False]]

        # Check if the pruned DataFrame still has the same number of rows
        self.assertEqual(
            len(pruned_df),
            len(mock_df),
            "The pruned DataFrame should have the same number of rows.",
        )

        # Check that the labels are correctly pruned
        for i in range(len(pruned_df)):
            self.assertEqual(
                pruned_df.iloc[i]["labels"],
                expected_labels[i],
                f"Row {i}'s labels should be pruned correctly.",
            )


if __name__ == "__main__":
    unittest.main()
