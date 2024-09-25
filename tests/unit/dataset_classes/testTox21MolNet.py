import unittest
from typing import List
from unittest.mock import MagicMock, mock_open, patch

from chebai.preprocessing.datasets.tox21 import Tox21MolNet
from tests.unit.mock_data.tox_mock_data import Tox21MolNetMockData


class TestTox21MolNet(unittest.TestCase):
    @classmethod
    @patch("os.makedirs", return_value=None)
    def setUpClass(cls, mock_makedirs: MagicMock) -> None:
        """
        Initialize a Tox21MolNet instance for testing.

        Args:
            mock_makedirs (MagicMock): Mocked `os.makedirs` function.
        """
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
            "The loaded data does not match the expected output from the file.",
        )

    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data=Tox21MolNetMockData.get_raw_data(),
    )
    @patch("torch.save")
    def test_setup_processed_simple_split(
        self,
        mock_torch_save: MagicMock,
        mock_open_file: mock_open,
    ) -> None:
        """
        Test the `setup_processed` method for basic data splitting and saving.

        Args:
            mock_torch_save (MagicMock): Mocked `torch.save` function to avoid actual file writes.
            mock_open_file (mock_open): Mocked `open` function to simulate file reading.
        """
        self.data_module.setup_processed()

        # Verify if torch.save was called for each split (train, test, validation)
        self.assertEqual(
            mock_torch_save.call_count, 3, "Expected torch.save to be called 3 times."
        )
        call_args_list = mock_torch_save.call_args_list
        self.assertIn("test", call_args_list[0][0][1], "Missing 'test' split.")
        self.assertIn("train", call_args_list[1][0][1], "Missing 'train' split.")
        self.assertIn(
            "validation", call_args_list[2][0][1], "Missing 'validation' split."
        )

        # Check for non-overlap between train, test, and validation splits
        test_split: List[str] = [d["ident"] for d in call_args_list[0][0][0]]
        train_split: List[str] = [d["ident"] for d in call_args_list[1][0][0]]
        validation_split: List[str] = [d["ident"] for d in call_args_list[2][0][0]]

        self.assertTrue(
            set(train_split).isdisjoint(test_split),
            "Overlap detected between the train and test splits.",
        )
        self.assertTrue(
            set(train_split).isdisjoint(validation_split),
            "Overlap detected between the train and validation splits.",
        )
        self.assertTrue(
            set(test_split).isdisjoint(validation_split),
            "Overlap detected between the test and validation splits.",
        )

    @patch.object(
        Tox21MolNet,
        "_load_data_from_file",
        return_value=Tox21MolNetMockData.get_processed_grouped_data(),
    )
    @patch("torch.save")
    def test_setup_processed_with_group_split(
        self, mock_torch_save: MagicMock, mock_load_file: MagicMock
    ) -> None:
        """
        Test the `setup_processed` method for group-based splitting and saving.

        Args:
            mock_torch_save (MagicMock): Mocked `torch.save` function to avoid actual file writes.
            mock_load_file (MagicMock): Mocked `_load_data_from_file` to provide custom data.
        """
        self.data_module.train_split = 0.5
        self.data_module.setup_processed()

        # Verify if torch.save was called for each split
        self.assertEqual(
            mock_torch_save.call_count, 3, "Expected torch.save to be called 3 times."
        )
        call_args_list = mock_torch_save.call_args_list
        self.assertIn("test", call_args_list[0][0][1], "Missing 'test' split.")
        self.assertIn("train", call_args_list[1][0][1], "Missing 'train' split.")
        self.assertIn(
            "validation", call_args_list[2][0][1], "Missing 'validation' split."
        )

        # Check for non-overlap between train, test, and validation splits (based on 'ident')
        test_split: List[str] = [d["ident"] for d in call_args_list[0][0][0]]
        train_split: List[str] = [d["ident"] for d in call_args_list[1][0][0]]
        validation_split: List[str] = [d["ident"] for d in call_args_list[2][0][0]]

        self.assertTrue(
            set(train_split).isdisjoint(test_split),
            "Overlap detected between the train and test splits (based on 'ident').",
        )
        self.assertTrue(
            set(train_split).isdisjoint(validation_split),
            "Overlap detected between the train and validation splits (based on 'ident').",
        )
        self.assertTrue(
            set(test_split).isdisjoint(validation_split),
            "Overlap detected between the test and validation splits (based on 'ident').",
        )

        # Check for non-overlap between train, test, and validation splits (based on 'group')
        test_split_grp: List[str] = [d["group"] for d in call_args_list[0][0][0]]
        train_split_grp: List[str] = [d["group"] for d in call_args_list[1][0][0]]
        validation_split_grp: List[str] = [d["group"] for d in call_args_list[2][0][0]]

        self.assertTrue(
            set(train_split_grp).isdisjoint(test_split_grp),
            "Overlap detected between the train and test splits (based on 'group').",
        )
        self.assertTrue(
            set(train_split_grp).isdisjoint(validation_split_grp),
            "Overlap detected between the train and validation splits (based on 'group').",
        )
        self.assertTrue(
            set(test_split_grp).isdisjoint(validation_split_grp),
            "Overlap detected between the test and validation splits (based on 'group').",
        )


if __name__ == "__main__":
    unittest.main()
