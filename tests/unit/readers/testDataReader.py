import unittest
from typing import Any, Dict, List

from chebai.preprocessing.reader import DataReader


class TestDataReader(unittest.TestCase):
    """
    Unit tests for the DataReader class.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """
        Set up the test environment by initializing a DataReader instance.
        """
        cls.reader = DataReader()

    def test_to_data(self) -> None:
        """
        Test the to_data method to ensure it correctly processes the input row
        and formats it according to the expected output.

        This method tests the conversion of raw data into a processed format,
        including extracting features, labels, ident, group, and additional
        keyword arguments.
        """
        features_list: List[int] = [10, 20, 30]
        labels_list: List[bool] = [True, False, True]
        ident_no: int = 123

        row: Dict[str, Any] = {
            "features": features_list,
            "labels": labels_list,
            "ident": ident_no,
            "group": "group_data",
            "additional_kwargs": {"extra_key": "extra_value"},
        }

        expected: Dict[str, Any] = {
            "features": features_list,
            "labels": labels_list,
            "ident": ident_no,
            "group": "group_data",
            "extra_key": "extra_value",
        }

        self.assertEqual(
            self.reader.to_data(row),
            expected,
            "The to_data method did not process the input row as expected.",
        )


if __name__ == "__main__":
    unittest.main()
