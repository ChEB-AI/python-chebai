import os
import unittest
from unittest.mock import MagicMock, mock_open, patch

from rdkit import Chem

from chebai.preprocessing.datasets.tox21 import Tox21Challenge
from chebai.preprocessing.reader import ChemDataReader
from tests.unit.mock_data.tox_mock_data import Tox21ChallengeMockData


class TestTox21Challenge(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Set up the Tox21Challenge instance and mock data for testing.
        """
        Tox21Challenge.READER = ChemDataReader
        cls.tox21 = Tox21Challenge()

    @patch("rdkit.Chem.SDMolSupplier")
    def test_load_data_from_file(self, mock_sdmol_supplier) -> None:
        """
        Test the _load_data_from_file method to ensure it correctly loads data from an SDF file.
        """
        # Use ForwardSDMolSupplier to read the mock data from the binary string
        mock_file = mock_open(read_data=Tox21ChallengeMockData.get_raw_train_data())
        with patch("builtins.open", mock_file):
            with open(
                r"G:\github-aditya0by0\chebai_data\tox21_challenge\tox21_10k_data_all.sdf\tox21_10k_data_all.sdf",
                "rb",
            ) as f:
                suppl = Chem.ForwardSDMolSupplier(f)

        mock_sdmol_supplier.return_value = suppl

        actual_data = self.tox21._load_data_from_file("fake/path")
        self.assertEqual(Tox21ChallengeMockData.data_in_dict_format(), actual_data)


if __name__ == "__main__":
    unittest.main()
