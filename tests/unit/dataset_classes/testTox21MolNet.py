import os
import unittest
from typing import Dict, List
from unittest.mock import MagicMock, mock_open, patch

import torch
from sklearn.model_selection import GroupShuffleSplit

from chebai.preprocessing.datasets.tox21 import Tox21MolNet
from tests.unit.mock_data.tox_mock_data import Tox21MolNetMockData


class TestTox21MolNet(unittest.TestCase):

    @classmethod
    @patch("os.makedirs", return_value=None)
    def setUpClass(cls, mock_makedirs) -> None:
        """Initialize a Tox21MolNet instance for testing."""
        ReaderMock = MagicMock()
        ReaderMock.name.return_value = "MockedReaderTox21MolNet"
        Tox21MolNet.READER = ReaderMock
        cls.data_module = Tox21MolNet()

    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data=Tox21MolNetMockData.get_raw_data(),
    )
    def test_load_data_from_file(self, mock_open_file: mock_open) -> None:
        """
        Test the `_load_data_from_file` method for correct CSV parsing.

        Args:
            mock_open_file (mock_open): Mocked open function to simulate file reading.
        """
        expected_data = Tox21MolNetMockData.get_processed_data()
        actual_data = self.data_module._load_data_from_file("fake/file/path.csv")

        self.assertEqual(
            list(actual_data),
            expected_data,
            "The loaded data does not match the expected output.",
        )

    @patch.object(
        Tox21MolNet,
        "_load_data_from_file",
        return_value=Tox21MolNetMockData.get_processed_data(),
    )
    @patch("torch.save")
    def test_setup_processed_simple_split(
        self, mock_load_data: MagicMock, mock_torch_save: MagicMock
    ) -> None:
        """
        Test the `setup_processed` method for basic data splitting and saving.

        Args:
            mock_load_data (MagicMock): Mocked `_load_data_from_file` method to provide controlled data.
            mock_torch_save (MagicMock): Mocked `torch.save` function to avoid actual file writes.
        """
        # Facing technical error here
        self.data_module.setup_processed()


if __name__ == "__main__":
    unittest.main()
