import unittest
from typing import List
from unittest.mock import mock_open, patch

from chebai.preprocessing.reader import EMBEDDING_OFFSET, DeepChemDataReader


class TestDeepChemDataReader(unittest.TestCase):
    """
    Unit tests for the DeepChemDataReader class.

    Note: Test methods within a TestCase class are not guaranteed to be executed in any specific order.
    """

    @classmethod
    @patch(
        "chebai.preprocessing.reader.open",
        new_callable=mock_open,
        read_data="C\nO\nc\n)",
    )
    def setUpClass(cls, mock_file: mock_open) -> None:
        """
        Set up the test environment by initializing a DeepChemDataReader instance with a mocked token file.

        Args:
            mock_file: Mock object for file operations.
        """
        cls.reader = DeepChemDataReader(token_path="/mock/path")
        # After initializing, cls.reader.cache should now be set to ['C', 'O', 'c', ')']
        assert list(cls.reader.cache.items()) == list(
            {
                "C": 0,
                "O": 1,
                "c": 2,
                ")": 3,
            }.items()
        ), "Cache initialization did not match expected tokens or the expected order."

    def test_read_data(self) -> None:
        """
        Test the _read_data method with a SMILES string to ensure it correctly tokenizes the string.
        """
        raw_data = "c1ccccc1C(Br)(OC)I[Ni-2]"

        # benzene is c1ccccc1 in SMILES but cccccc6 in DeepSMILES
        # SMILES C(Br)(OC)I can be converted to the DeepSMILES CBr)OC))I.
        # Resultant String: "cccccc6CBr)OC))I[Ni-2]"
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
        self.assertEqual(
            result,
            expected_output,
            "The _read_data method did not produce the expected tokenized output for the SMILES string.",
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
            "The _read_data method did not produce the expected output for a SMILES string with a new token.",
        )

        # Verify that '[H-]' was added to the cache
        self.assertIn(
            "[H-]",
            self.reader.cache,
            "The new token '[H-]' was not added to the cache as expected.",
        )
        # Ensure it's at the correct index
        self.assertEqual(
            self.reader.cache["[H-]"],
            index_for_last_token,
            "The new token '[H-]' was not added to the correct index in the cache.",
        )

    def test_read_data_with_invalid_input(self) -> None:
        """
        Test the _read_data method with an invalid input string.
        The invalid token should raise an error or be handled appropriately.
        """
        raw_data = "CBr))(OCI"

        with self.assertRaises(Exception):
            self.reader._read_data(raw_data)


if __name__ == "__main__":
    unittest.main()
