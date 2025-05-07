import unittest
from typing import List
from unittest.mock import mock_open, patch

from chebai.preprocessing.reader import EMBEDDING_OFFSET, ChemDataReader


class TestChemDataReader(unittest.TestCase):
    """
    Unit tests for the ChemDataReader class.

    Note: Test methods within a TestCase class are not guaranteed to be executed in any specific order.
    """

    @classmethod
    @patch(
        "chebai.preprocessing.reader.open",
        new_callable=mock_open,
        read_data="C\nO\nN\n=\n1\n(",
    )
    def setUpClass(cls, mock_file: mock_open) -> None:
        """
        Set up the test environment by initializing a ChemDataReader instance with a mocked token file.

        Args:
            mock_file: Mock object for file operations.
        """
        cls.reader = ChemDataReader(token_path="/mock/path")
        # After initializing, cls.reader.cache should now be set to ['C', 'O', 'N', '=', '1', '(']
        assert list(cls.reader.cache.items()) == list(
            {
                "C": 0,
                "O": 1,
                "N": 2,
                "=": 3,
                "1": 4,
                "(": 5,
            }.items()
        ), "Initial cache does not match expected values or the order doesn't match."

    def test_read_data(self) -> None:
        """
        Test the _read_data method with a SMILES string to ensure it correctly tokenizes the string.
        """
        raw_data = "CC(=O)NC1[Mg-2]"
        # Expected output as per the tokens already in the cache, and ")" getting added to it.
        expected_output: List[int] = [
            EMBEDDING_OFFSET + 0,  # C
            EMBEDDING_OFFSET + 0,  # C
            EMBEDDING_OFFSET + 5,  # =
            EMBEDDING_OFFSET + 3,  # O
            EMBEDDING_OFFSET + 1,  # N
            EMBEDDING_OFFSET + len(self.reader.cache),  # (
            EMBEDDING_OFFSET + 2,  # C
            EMBEDDING_OFFSET + 0,  # C
            EMBEDDING_OFFSET + 4,  # 1
            EMBEDDING_OFFSET + len(self.reader.cache) + 1,  # [Mg-2]
        ]
        result = self.reader._read_data(raw_data)
        self.assertEqual(
            result,
            expected_output,
            "The output of _read_data does not match the expected tokenized values.",
        )

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
        self.assertEqual(
            result,
            expected_output,
            "The output for new token '[H-]' does not match the expected values.",
        )

        # Verify that '[H-]' was added to the cache
        self.assertIn(
            "[H-]",
            self.reader.cache,
            "The new token '[H-]' was not added to the cache.",
        )
        # Ensure it's at the correct index
        self.assertEqual(
            self.reader.cache["[H-]"],
            index_for_last_token,
            "The new token '[H-]' was not added at the correct index in the cache.",
        )

    def test_read_data_with_invalid_input(self) -> None:
        """
        Test the _read_data method with an invalid input.
        The invalid token should raise an error or be handled appropriately.
        """
        raw_data = "%INVALID%"

        with self.assertRaises(ValueError):
            self.reader._read_data(raw_data)

    @patch("builtins.open", new_callable=mock_open)
    def test_finish_method_for_new_tokens(self, mock_file: mock_open) -> None:
        """
        Test the on_finish method to ensure it appends only the new tokens to the token file in order.
        """
        # Simulate that some tokens were already loaded
        self.reader._loaded_tokens_count = 6  # 6 tokens already loaded
        self.reader.cache = {
            "C": 0,
            "O": 1,
            "N": 2,
            "=": 3,
            "1": 4,
            "(": 5,
            "[H-]": 6,  # New token 1
            "Br": 7,  # New token 2
            "Cl": 8,  # New token 3
            "Na": 9,  # New token 4
            "Mg": 10,  # New token 5
        }

        # Run the on_finish method
        self.reader.on_finish()

        # Check that the file was opened in append mode ('a')
        mock_file.assert_called_with(self.reader.token_path, "a")

        # Verify the new tokens were written in the correct order
        mock_file().writelines.assert_called_with(
            ["[H-]\n", "Br\n", "Cl\n", "Na\n", "Mg\n"]
        )

    def test_finish_method_no_new_tokens(self) -> None:
        """
        Test the on_finish method when no new tokens are added (cache is the same).
        """
        self.reader._loaded_tokens_count = 6  # No new tokens
        self.reader.cache = {
            "C": 0,
            "O": 1,
            "N": 2,
            "=": 3,
            "1": 4,
            "(": 5,
        }

        with patch("builtins.open", new_callable=mock_open) as mock_file:
            self.reader.on_finish()
            # Check that no new tokens were written
            mock_file().writelines.assert_not_called()


if __name__ == "__main__":
    unittest.main()
