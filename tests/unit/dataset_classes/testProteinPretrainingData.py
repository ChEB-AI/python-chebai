import unittest
from unittest.mock import PropertyMock, mock_open, patch

from chebai.preprocessing.datasets.deepGO.protein_pretraining import (
    _ProteinPretrainingData,
)
from chebai.preprocessing.reader import ProteinDataReader
from tests.unit.mock_data.ontology_mock_data import GOUniProtMockData


class TestProteinPretrainingData(unittest.TestCase):
    """
    Unit tests for the _ProteinPretrainingData class.
    Tests focus on data parsing and validation checks for protein pretraining.
    """

    @classmethod
    @patch.multiple(_ProteinPretrainingData, __abstractmethods__=frozenset())
    @patch.object(_ProteinPretrainingData, "base_dir", new_callable=PropertyMock)
    @patch.object(_ProteinPretrainingData, "_name", new_callable=PropertyMock)
    @patch("os.makedirs", return_value=None)
    def setUpClass(
        cls,
        mock_makedirs,
        mock_name_property: PropertyMock,
        mock_base_dir_property: PropertyMock,
    ) -> None:
        """
        Class setup for mocking abstract properties of _ProteinPretrainingData.

        Mocks the required abstract properties and sets up the data extractor.
        """
        mock_base_dir_property.return_value = "MockedBaseDirPropProteinPretrainingData"
        mock_name_property.return_value = "MockedNameProp_ProteinPretrainingData"

        # Set the READER class for the pretraining data
        _ProteinPretrainingData.READER = ProteinDataReader

        # Initialize the extractor instance
        cls.extractor = _ProteinPretrainingData()

    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data=GOUniProtMockData.get_UniProt_raw_data(),
    )
    def test_parse_protein_data_for_pretraining(
        self, mock_open_file: mock_open
    ) -> None:
        """
        Tests the _parse_protein_data_for_pretraining method.

        Verifies that:
        - The parsed DataFrame contains the expected protein IDs.
        - The protein sequences are not empty.
        """
        # Parse the pretraining data
        pretrain_df = self.extractor._parse_protein_data_for_pretraining()
        list_of_pretrain_swiss_ids = GOUniProtMockData.proteins_for_pretraining()

        # Assert that all expected Swiss-Prot IDs are present in the DataFrame
        self.assertEqual(
            set(pretrain_df["swiss_id"]),
            set(list_of_pretrain_swiss_ids),
            msg="The parsed DataFrame does not contain the expected Swiss-Prot IDs for pretraining.",
        )

        # Assert that all sequences are not empty
        self.assertTrue(
            pretrain_df["sequence"].str.len().gt(0).all(),
            msg="Some protein sequences in the pretraining DataFrame are empty.",
        )


if __name__ == "__main__":
    unittest.main()
