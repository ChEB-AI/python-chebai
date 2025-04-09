import unittest
from typing import List
from unittest.mock import mock_open, patch

import networkx as nx
import pandas as pd

from chebai.preprocessing.datasets.deepGO.go_uniprot import _GOUniProtOverX
from tests.unit.mock_data.ontology_mock_data import GOUniProtMockData


class TestGOUniProtOverX(unittest.TestCase):
    @classmethod
    @patch.multiple(_GOUniProtOverX, __abstractmethods__=frozenset())
    @patch("os.makedirs", return_value=None)
    def setUpClass(cls, mock_makedirs) -> None:
        """
        Set up the class for tests by initializing the extractor, graph, and input DataFrame.
        """
        cls.extractor = _GOUniProtOverX()
        cls.test_graph: nx.DiGraph = GOUniProtMockData.get_transitively_closed_graph()
        cls.input_df: pd.DataFrame = GOUniProtMockData.get_data_in_dataframe().iloc[
            :, :4
        ]

    @patch("builtins.open", new_callable=mock_open)
    def test_select_classes(self, mock_open_file: mock_open) -> None:
        """
        Test the `select_classes` method to ensure it selects classes based on the threshold.

        Args:
            mock_open_file (mock_open): Mocked open function to intercept file operations.
        """
        # Set threshold for testing
        self.extractor.THRESHOLD = 2
        selected_classes: List[int] = self.extractor.select_classes(
            self.test_graph, data_df=self.input_df
        )

        # Expected result: GO terms 1, 2, and 5 should be selected based on the threshold
        expected_selected_classes: List[int] = sorted([1, 2, 5])

        # Check if the selected classes are as expected
        self.assertEqual(
            selected_classes,
            expected_selected_classes,
            msg="The selected classes do not match the expected output for threshold 2.",
        )

        # Expected data as string
        expected_lines: str = "\n".join(map(str, expected_selected_classes)) + "\n"

        # Extract the generator passed to writelines
        written_generator = mock_open_file().writelines.call_args[0][0]
        written_lines: str = "".join(written_generator)

        # Ensure the data matches
        self.assertEqual(
            written_lines,
            expected_lines,
            msg="The written lines do not match the expected lines for the given threshold of 2.",
        )

    @patch("builtins.open", new_callable=mock_open)
    def test_no_classes_meet_threshold(self, mock_open_file: mock_open) -> None:
        """
        Test the `select_classes` method when no nodes meet the successor threshold.

        Args:
            mock_open_file (mock_open): Mocked open function to intercept file operations.
        """
        self.extractor.THRESHOLD = 5
        selected_classes: List[int] = self.extractor.select_classes(
            self.test_graph, data_df=self.input_df
        )

        # Expected result: No classes should meet the threshold of 5
        expected_selected_classes: List[int] = []

        # Check if the selected classes are as expected
        self.assertEqual(
            selected_classes,
            expected_selected_classes,
            msg="The selected classes list should be empty when no nodes meet the threshold of 5.",
        )

        # Expected data as string
        expected_lines: str = ""

        # Extract the generator passed to writelines
        written_generator = mock_open_file().writelines.call_args[0][0]
        written_lines: str = "".join(written_generator)

        # Ensure the data matches
        self.assertEqual(
            written_lines,
            expected_lines,
            msg="The written lines do not match the expected lines when no nodes meet the threshold of 5.",
        )

    @patch("builtins.open", new_callable=mock_open)
    def test_all_nodes_meet_threshold(self, mock_open_file: mock_open) -> None:
        """
        Test the `select_classes` method when all nodes meet the successor threshold.

        Args:
            mock_open_file (mock_open): Mocked open function to intercept file operations.
        """
        self.extractor.THRESHOLD = 0
        selected_classes: List[int] = self.extractor.select_classes(
            self.test_graph, data_df=self.input_df
        )

        # Expected result: All nodes except those not referenced by any protein (4 and 6) should be selected
        expected_classes: List[int] = sorted([1, 2, 3, 5])

        # Check if the returned selected classes match the expected list
        self.assertListEqual(
            selected_classes,
            expected_classes,
            msg="The selected classes do not match the expected output when all nodes meet the threshold of 0.",
        )

        # Expected data as string
        expected_lines: str = "\n".join(map(str, expected_classes)) + "\n"

        # Extract the generator passed to writelines
        written_generator = mock_open_file().writelines.call_args[0][0]
        written_lines: str = "".join(written_generator)

        # Ensure the data matches
        self.assertEqual(
            written_lines,
            expected_lines,
            msg="The written lines do not match the expected lines when all nodes meet the threshold of 0.",
        )


if __name__ == "__main__":
    unittest.main()
