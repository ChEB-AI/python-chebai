import unittest
from unittest.mock import PropertyMock, patch

from chebai.preprocessing.datasets.base import XYBaseDataModule
from chebai.preprocessing.reader import ProteinDataReader


class TestXYBaseDataModule(unittest.TestCase):
    """
    Unit tests for the methods of the XYBaseDataModule class.
    """

    @classmethod
    @patch.object(XYBaseDataModule, "_name", new_callable=PropertyMock)
    def setUpClass(cls, mock_name_property) -> None:
        """
        Set up a base instance of XYBaseDataModule for testing.
        """

        # Mock the _name property of XYBaseDataModule
        mock_name_property.return_value = "MockedXYBaseDataModule"

        # Assign a static variable READER with ProteinDataReader (to get rid of default Abstract DataReader)
        XYBaseDataModule.READER = ProteinDataReader

        # Initialize the module with a label_filter
        cls.module = XYBaseDataModule(
            label_filter=1,  # Provide a label_filter
            balance_after_filter=1.0,  # Balance ratio
        )

    def test_filter_labels_valid_index(self) -> None:
        """
        Test the _filter_labels method with a valid label_filter index.
        """
        self.module.label_filter = 1
        row = {
            "features": ["feature1", "feature2"],
            "labels": [0, 3, 1, 2],  # List of labels
        }
        filtered_row = self.module._filter_labels(row)
        expected_labels = [3]  # Only the label at index 1 should be kept

        self.assertEqual(filtered_row["labels"], expected_labels)

        row = {
            "features": ["feature1", "feature2"],
            "labels": [True, False, True, True],
        }
        self.assertEqual(self.module._filter_labels(row)["labels"], [False])

    def test_filter_labels_no_filter(self) -> None:
        """
        Test the _filter_labels method with no label_filter index.
        """
        # Update the module to have no label filter
        self.module.label_filter = None
        row = {"features": ["feature1", "feature2"], "labels": [False, True]}
        # Handle the case where the index is out of bounds
        with self.assertRaises(TypeError):
            self.module._filter_labels(row)

    def test_filter_labels_invalid_index(self) -> None:
        """
        Test the _filter_labels method with an invalid label_filter index.
        """
        # Set an invalid label filter index (e.g., greater than the number of labels)
        self.module.label_filter = 10
        row = {"features": ["feature1", "feature2"], "labels": [False, True]}
        # Handle the case where the index is out of bounds
        with self.assertRaises(IndexError):
            self.module._filter_labels(row)


if __name__ == "__main__":
    unittest.main()
