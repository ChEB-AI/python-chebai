import unittest
from typing import List
from unittest.mock import mock_open, patch

from chebai.preprocessing.reader import EMBEDDING_OFFSET, DeepChemDataReader


class TestDeepChemDataReader(unittest.TestCase):
    """
    Unit tests for the DeepChemDataReader class.

    Note: Test methods within a TestCase class are not guaranteed to be executed in any specific order.
    """

    @patch(
        "chebai.preprocessing.reader.open",
        new_callable=mock_open,
        read_data="C\nO\nc\n)",
    )
    def setUp(self, mock_file: mock_open) -> None:
        """
        Set up the test environment by initializing a DeepChemDataReader instance with a mocked token file.

        Args:
            mock_file: Mock object for file operations.
        """
        self.reader = DeepChemDataReader(token_path="/mock/path")
        # After initializing, self.reader.cache should now be set to ['C', 'O', 'c', ')']
        self.assertEqual(self.reader.cache, ["C", "O", "c", ")"])

    def test_read_data(self) -> None:
        """
        Test the _read_data method with a SMILES string to ensure it correctly tokenizes the string.
        """
        raw_data = "c1ccccc1C(Br)(OC)I[Ni-2]"

        # Expected output as per the tokens already in the cache, and new tokens getting added to it.
        expected_output: List[int] = [
            EMBEDDING_OFFSET + 2,  # c
            EMBEDDING_OFFSET + 2,  # c
            EMBEDDING_OFFSET + 2,  # c
            EMBEDDING_OFFSET + 2,  # c
            EMBEDDING_OFFSET + 2,  # c
            EMBEDDING_OFFSET + 2,  # c
            EMBEDDING_OFFSET + len(self.reader.cache),  # 6 (new token)
            EMBEDDING_OFFSET + 0,  # C
            EMBEDDING_OFFSET + len(self.reader.cache) + 1,  # Br (new token)
            EMBEDDING_OFFSET + 3,  # )
            EMBEDDING_OFFSET + 1,  # O
            EMBEDDING_OFFSET + 0,  # C
            EMBEDDING_OFFSET + 3,  # )
            EMBEDDING_OFFSET + 3,  # )
            EMBEDDING_OFFSET + len(self.reader.cache) + 2,  # I (new token)
            EMBEDDING_OFFSET + len(self.reader.cache) + 3,  # [Ni-2] (new token)
        ]
        result = self.reader._read_data(raw_data)
        self.assertEqual(result, expected_output)

    def test_read_data_with_new_token(self) -> None:
        """
        Test the _read_data method with a SMILES string that includes a new token.
        Ensure that the new token is added to the cache and processed correctly.
        """
        raw_data = "[H-]"

        # Determine the index for the new token based on the current size of the cache.
        index_for_last_token = len(self.reader.cache)
        expected_output: List[int] = [EMBEDDING_OFFSET + index_for_last_token]

        result = self.reader._read_data(raw_data)
        self.assertEqual(result, expected_output)

        # Verify that '[H-]' was added to the cache
        self.assertIn("[H-]", self.reader.cache)
        # Ensure it's at the correct index
        self.assertEqual(self.reader.cache.index("[H-]"), index_for_last_token)


if __name__ == "__main__":
    unittest.main()
