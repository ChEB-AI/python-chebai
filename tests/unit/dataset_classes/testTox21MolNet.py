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

    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data=Tox21MolNetMockData.get_raw_data(),
    )
    @patch("torch.save")
    def test_setup_processed_simple_split(
        self,
        mock_torch_save,
        mock_open_file: mock_open,
    ) -> None:
        """
        Test the `setup_processed` method for basic data splitting and saving.

        Args:
            mock_torch_save : Mocked `torch.save` function to avoid actual file writes.
            mock_open_file (mock_open): Mocked `open` builtin-method to provide custom data.
        """
        self.data_module.setup_processed()

        # Verify if torch.save was called for each split
        self.assertEqual(mock_torch_save.call_count, 3)
        call_args_list = mock_torch_save.call_args_list
        self.assertIn("test", call_args_list[0][0][1])
        self.assertIn("train", call_args_list[1][0][1])
        self.assertIn("validation", call_args_list[2][0][1])

        # Check for non-overlap between train, test, and validation
        test_split = [d["ident"] for d in call_args_list[0][0][0]]
        train_split = [d["ident"] for d in call_args_list[1][0][0]]
        validation_split = [d["ident"] for d in call_args_list[2][0][0]]

        # Assert no overlap between splits
        self.assertTrue(
            set(train_split).isdisjoint(test_split),
            "There is an overlap between the train and test splits.",
        )
        self.assertTrue(
            set(train_split).isdisjoint(validation_split),
            "There is an overlap between the train and validation splits.",
        )
        self.assertTrue(
            set(test_split).isdisjoint(validation_split),
            "There is an overlap between the test and validation splits.",
        )


if __name__ == "__main__":
    unittest.main()
