import unittest
from unittest.mock import PropertyMock, mock_open, patch

from chebai.preprocessing.datasets.chebi import ChEBIOverX
from tests.unit.mock_data.ontology_mock_data import ChebiMockOntology


class TestChEBIOverX(unittest.TestCase):
    @classmethod
    @patch.multiple(ChEBIOverX, __abstractmethods__=frozenset())
    @patch.object(ChEBIOverX, "processed_dir_main", new_callable=PropertyMock)
    @patch("os.makedirs", return_value=None)
    def setUpClass(cls, mock_makedirs, mock_processed_dir_main: PropertyMock) -> None:
        """
        Set up the ChEBIOverX instance with a mock processed directory path and a test graph.

        Args:
            mock_makedirs: This patches os.makedirs to do nothing
            mock_processed_dir_main (PropertyMock): Mocked property for the processed directory path.
        """
        mock_processed_dir_main.return_value = "/mock/processed_dir"
        cls.chebi_extractor = ChEBIOverX(chebi_version=231)
        cls.test_graph = ChebiMockOntology.get_transitively_closed_graph()

    @patch("builtins.open", new_callable=mock_open)
    def test_select_classes(self, mock_open_file: mock_open) -> None:
        """
        Test the select_classes method to ensure it correctly selects nodes based on the threshold.

        Args:
            mock_open_file (mock_open): Mocked open function to intercept file operations.
        """
        self.chebi_extractor.THRESHOLD = 3
        selected_classes = self.chebi_extractor.select_classes(self.test_graph)

        # Check if the returned selected classes match the expected list
        expected_classes = sorted([11111, 22222, 67890])
        self.assertListEqual(
            selected_classes,
            expected_classes,
            "The selected classes do not match the expected output for the given threshold of 3.",
        )

        # Expected data as string
        expected_lines = "\n".join(map(str, expected_classes)) + "\n"

        # Extract the generator passed to writelines
        written_generator = mock_open_file().writelines.call_args[0][0]
        written_lines = "".join(written_generator)

        # Ensure the data matches
        self.assertEqual(
            written_lines,
            expected_lines,
            "The written lines do not match the expected lines for the given threshold of 3.",
        )

    @patch("builtins.open", new_callable=mock_open)
    def test_no_classes_meet_threshold(self, mock_open_file: mock_open) -> None:
        """
        Test the select_classes method when no nodes meet the successor threshold.

        Args:
            mock_open_file (mock_open): Mocked open function to intercept file operations.
        """
        self.chebi_extractor.THRESHOLD = 5
        selected_classes = self.chebi_extractor.select_classes(self.test_graph)

        # Expected empty result
        self.assertEqual(
            selected_classes,
            [],
            "The selected classes list should be empty when no nodes meet the threshold of 5.",
        )

        # Expected data as string
        expected_lines = ""

        # Extract the generator passed to writelines
        written_generator = mock_open_file().writelines.call_args[0][0]
        written_lines = "".join(written_generator)

        # Ensure the data matches
        self.assertEqual(
            written_lines,
            expected_lines,
            "The written lines do not match the expected lines when no nodes meet the threshold of 5.",
        )

    @patch("builtins.open", new_callable=mock_open)
    def test_all_nodes_meet_threshold(self, mock_open_file: mock_open) -> None:
        """
        Test the select_classes method when all nodes meet the successor threshold.

        Args:
            mock_open_file (mock_open): Mocked open function to intercept file operations.
        """
        self.chebi_extractor.THRESHOLD = 0
        selected_classes = self.chebi_extractor.select_classes(self.test_graph)

        expected_classes = sorted(ChebiMockOntology.get_nodes())
        # Check if the returned selected classes match the expected list
        self.assertListEqual(
            selected_classes,
            expected_classes,
            "The selected classes do not match the expected output when all nodes meet the threshold of 0.",
        )

        # Expected data as string
        expected_lines = "\n".join(map(str, expected_classes)) + "\n"

        # Extract the generator passed to writelines
        written_generator = mock_open_file().writelines.call_args[0][0]
        written_lines = "".join(written_generator)

        # Ensure the data matches
        self.assertEqual(
            written_lines,
            expected_lines,
            "The written lines do not match the expected lines when all nodes meet the threshold of 0.",
        )


if __name__ == "__main__":
    unittest.main()
