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
    def setUpClass(cls) -> None:
        """Initialize a Tox21MolNet instance for testing."""
        ReaderMock = MagicMock()
        ReaderMock.name.return_value = "MockedReaderTox21MolNet"
        Tox21MolNet.READER = ReaderMock
        cls.data_module = Tox21MolNet()
        # cls.data_module.raw_dir = "/mock/raw_dir"
        # cls.data_module.processed_dir = "/mock/processed_dir"

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
        self.data_module.setup_processed()

        # # Check that torch.save was called for train, test, and validation splits
        # self.assertEqual(
        #     mock_torch_save.call_count,
        #     3,
        #     "torch.save should have been called exactly three times for train, test, and validation splits."
        # )

    # @patch("os.path.isfile", return_value=False)
    # @patch.object(Tox21MolNet,
    #               "_load_data_from_file",
    #               return_value= Tox21MolNetMockData.get_processed_grouped_data())
    # @patch("torch.save")
    # @patch("torch.load")
    # @patch("chebai.preprocessing.datasets.tox21.GroupShuffleSplit")
    # def test_setup_processed_group_split(
    #         self,
    #         mock_group_split: MagicMock,
    #         mock_torch_load: MagicMock,
    #         mock_save: MagicMock,
    #         mock_load_data: MagicMock,
    #         mock_isfile: MagicMock
    # ) -> None:
    #     """
    #     Test the `setup_processed` method for group-based data splitting and saving.
    #
    #     Args:
    #         mock_save (MagicMock): Mocked `torch.save` function to avoid file writes.
    #         mock_load_data (MagicMock): Mocked `_load_data_from_file` method to provide controlled data.
    #         mock_isfile (MagicMock): Mocked `os.path.isfile` function to simulate file presence.
    #         mock_group_split (MagicMock): Mocked `GroupShuffleSplit` to control data splitting behavior.
    #     """
    #     mock_group_split.return_value = GroupShuffleSplit(n_splits=1, train_size=0.7)
    #     self.data_module.setup_processed()
    #
    #     # Load the test split
    #     test_split_path = os.path.join(self.data_module.processed_dir, "test.pt")
    #     test_split = torch.load(test_split_path)
    #
    #     # Check if torch.save was called with correct arguments
    #     mock_save.assert_any_call([mock_data[1]], "/mock/processed_dir/test.pt")
    #     mock_save.assert_any_call([mock_data[0]], "/mock/processed_dir/train.pt")
    #     mock_save.assert_any_call([mock_data[1]], "/mock/processed_dir/validation.pt")
    #     # Check that torch.save was called for train, test, and validation splits
    #     self.assertEqual(
    #         mock_torch_save.call_count,
    #         3,
    #         "torch.save should have been called exactly three times for train, test, and validation splits."
    #     )


if __name__ == "__main__":
    unittest.main()
