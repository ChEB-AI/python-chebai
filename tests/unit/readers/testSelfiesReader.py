import unittest
from typing import List
from unittest.mock import mock_open, patch

from chebai.preprocessing.reader import EMBEDDING_OFFSET, SelfiesReader


class TestSelfiesReader(unittest.TestCase):
    """
    Unit tests for the SelfiesReader class.

    Note: Test methods within a TestCase class are not guaranteed to be executed in any specific order.
    """

    @classmethod
    @patch(
        "chebai.preprocessing.reader.open",
        new_callable=mock_open,
        read_data="[C]\n[O]\n[=C]",
    )
    def setUpClass(cls, mock_file: mock_open) -> None:
        """
        Set up the test environment by initializing a SelfiesReader instance with a mocked token file.

        Args:
            mock_file: Mock object for file operations.
        """
        cls.reader = SelfiesReader(token_path="/mock/path")
        # After initializing, cls.reader.cache should now be set to ['[C]', '[O]', '[=C]']
        assert list(cls.reader.cache.items()) == list(
            {
                "[C]": 0,
                "[O]": 1,
                "[=C]": 2,
            }.items()
        ), "Cache initialization did not match expected tokens or the expected order."

    def test_read_data(self) -> None:
        """
        Test the _read_data method with a SELFIES string to ensure it correctly tokenizes the string.
        """
        raw_data = "c1ccccc1C(Br)(OC)I[Ni-2]"

        # benzene is "c1ccccc1" in SMILES is translated to "[C][=C][C][=C][C][=C][Ring1][=Branch1]" in SELFIES
        # SELFIES translation of SMILES "c1ccccc1C(Br)(OC)I[Ni-2]":
        # "[C][=C][C][=C][C][=C][Ring1][=Branch1][C][Branch1][C][Br][Branch1][Ring1][O][C][I][Ni-2]"
        expected_output: List[int] = [
            EMBEDDING_OFFSET + 0,  # [C] (already in cache)
            EMBEDDING_OFFSET + 2,  # [=C] (already in cache)
            EMBEDDING_OFFSET + 0,  # [C] (already in cache)
            EMBEDDING_OFFSET + 2,  # [=C] (already in cache)
            EMBEDDING_OFFSET + 0,  # [C] (already in cache)
            EMBEDDING_OFFSET + 2,  # [=C] (already in cache)
            EMBEDDING_OFFSET + len(self.reader.cache),  # [Ring1] (new token)
            EMBEDDING_OFFSET + len(self.reader.cache) + 1,  # [=Branch1] (new token)
            EMBEDDING_OFFSET + 0,  # [C] (already in cache)
            EMBEDDING_OFFSET + len(self.reader.cache) + 2,  # [Branch1] (new token)
            EMBEDDING_OFFSET + 0,  # [C] (already in cache)
            EMBEDDING_OFFSET + len(self.reader.cache) + 3,  # [Br] (new token)
            EMBEDDING_OFFSET
            + len(self.reader.cache)
            + 2,  # [Branch1] (reused new token)
            EMBEDDING_OFFSET + len(self.reader.cache),  # [Ring1] (reused new token)
            EMBEDDING_OFFSET + 1,  # [O] (already in cache)
            EMBEDDING_OFFSET + 0,  # [C] (already in cache)
            EMBEDDING_OFFSET + len(self.reader.cache) + 4,  # [I] (new token)
            EMBEDDING_OFFSET + len(self.reader.cache) + 5,  # [Ni-2] (new token)
        ]

        result = self.reader._read_data(raw_data)
        self.assertEqual(
            result,
            expected_output,
            "The _read_data method did not produce the expected tokenized output.",
        )

    def test_read_data_with_new_token(self) -> None:
        """
        Test the _read_data method with a SELFIES string that includes a new token.
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
            "The _read_data method did not correctly handle a new token.",
        )

        # Verify that '[H-1]' was added to the cache, "[H-]" translated to "[H-1]" in SELFIES
        self.assertIn(
            "[H-1]",
            self.reader.cache,
            "The new token '[H-1]' was not added to the cache.",
        )
        # Ensure it's at the correct index
        self.assertEqual(
            self.reader.cache["[H-1]"],
            index_for_last_token,
            "The new token '[H-1]' was not added at the correct index in the cache.",
        )

    def test_read_data_with_invalid_selfies(self) -> None:
        """
        Test the _read_data method with an invalid SELFIES string to ensure error handling works.
        """
        raw_data = "[C][O][INVALID][N]"

        result = self.reader._read_data(raw_data)
        self.assertIsNone(
            result,
            "The _read_data method did not return None for an invalid SELFIES string.",
        )

        # Verify that the error count was incremented
        self.assertEqual(
            self.reader.error_count,
            1,
            "The error count was not incremented for an invalid SELFIES string.",
        )


if __name__ == "__main__":
    unittest.main()
