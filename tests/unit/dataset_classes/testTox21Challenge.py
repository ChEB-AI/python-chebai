import unittest
from unittest.mock import mock_open, patch

from rdkit import Chem

from chebai.preprocessing.datasets.tox21 import Tox21Challenge
from chebai.preprocessing.reader import ChemDataReader
from tests.unit.mock_data.tox_mock_data import (
    Tox21ChallengeMockData,
    Tox21MolNetMockData,
)


class TestTox21Challenge(unittest.TestCase):
    """
    Unit tests for the Tox21Challenge class.
    """

    @classmethod
    @patch("os.makedirs", return_value=None)
    def setUpClass(cls, mock_makedirs) -> None:
        """
        Set up the Tox21Challenge instance and mock data for testing.
        This is run once for the test class.
        """
        Tox21Challenge.READER = ChemDataReader
        cls.tox21 = Tox21Challenge()

    @patch("rdkit.Chem.SDMolSupplier")
    def test_load_data_from_file(self, mock_sdmol_supplier: patch) -> None:
        """
        Test the `_load_data_from_file` method to ensure it correctly loads data from an SDF file.

        Args:
            mock_sdmol_supplier (patch): A mock of the RDKit SDMolSupplier.
        """
        # Use ForwardSDMolSupplier to read the mock data from the binary string
        mock_file = mock_open(read_data=Tox21ChallengeMockData.get_raw_train_data())
        with patch("builtins.open", mock_file):
            with open(
                r"fake/path",
                "rb",
            ) as f:
                suppl = Chem.ForwardSDMolSupplier(f)

        mock_sdmol_supplier.return_value = suppl

        actual_data = self.tox21._load_data_from_file("fake/path")
        expected_data = Tox21ChallengeMockData.data_in_dict_format()

        self.assertEqual(
            actual_data,
            expected_data,
            "The loaded data from file does not match the expected data.",
        )

    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data=Tox21MolNetMockData.get_raw_data(),
    )
    def test_load_dict(self, mock_open_file: mock_open) -> None:
        """
        Test the `_load_dict` method to ensure correct CSV parsing.

        Args:
            mock_open_file (mock_open): Mocked open function to simulate file reading.
        """
        expected_data = Tox21MolNetMockData.get_processed_data()
        for item in expected_data:
            item.pop("group", None)

        actual_data = self.tox21._load_dict("fake/file/path.csv")

        self.assertEqual(
            list(actual_data),
            expected_data,
            "The loaded data from CSV does not match the expected processed data.",
        )

    @patch.object(Tox21Challenge, "_load_data_from_file", return_value="test")
    @patch("builtins.open", new_callable=mock_open)
    @patch("torch.save")
    @patch("os.path.join")
    def test_setup_processed(
        self,
        mock_join: patch,
        mock_torch_save: patch,
        mock_open_file: mock_open,
        mock_load_file: patch,
    ) -> None:
        """
        Test the `setup_processed` method to ensure it processes and saves data correctly.

        Args:
            mock_join (patch): Mock of os.path.join to simulate file path joining.
            mock_torch_save (patch): Mock of torch.save to simulate saving processed data.
            mock_open_file (mock_open): Mocked open function to simulate file reading.
            mock_load_file (patch): Mocked data loading method.
        """
        # Simulated raw and processed directories
        path_str = "fake/test/path"
        mock_join.return_value = path_str

        # Mock the file content for test.smiles and score.txt
        mock_open_file.side_effect = [
            mock_open(
                read_data=Tox21ChallengeMockData.get_raw_smiles_data()
            ).return_value,
            mock_open(
                read_data=Tox21ChallengeMockData.get_raw_score_txt_data()
            ).return_value,
        ]

        # Call setup_processed to simulate the data processing workflow
        self.tox21.setup_processed()

        # Assert that torch.save was called with the correct processed data
        expected_test_data = Tox21ChallengeMockData.get_setup_processed_output_data()
        mock_torch_save.assert_called_with(expected_test_data, path_str)

        self.assertTrue(
            mock_torch_save.called, "The processed data was not saved as expected."
        )


if __name__ == "__main__":
    unittest.main()
