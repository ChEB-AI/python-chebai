import unittest
from typing import Dict, List

from chebai.preprocessing.collate import DefaultCollator
from chebai.preprocessing.structures import XYData


class TestDefaultCollator(unittest.TestCase):
    """
    Unit tests for the DefaultCollator class.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """
        Set up the test environment by initializing a DefaultCollator instance.
        """
        cls.collator = DefaultCollator()

    def test_call_with_valid_data(self) -> None:
        """
        Test the __call__ method with valid data to ensure features and labels are correctly extracted.
        """
        data: List[Dict] = [
            {"features": [1.0, 2.0], "labels": [True, False, True]},
            {"features": [3.0, 4.0], "labels": [False, False, True]},
        ]

        result: XYData = self.collator(data)
        self.assertIsInstance(result, XYData)

        expected_x = ([1.0, 2.0], [3.0, 4.0])
        expected_y = ([True, False, True], [False, False, True])

        self.assertEqual(result.x, expected_x)
        self.assertEqual(result.y, expected_y)

    def test_call_with_empty_data(self) -> None:
        """
        Test the __call__ method with an empty list to ensure it handles the edge case correctly.
        """
        data: List[Dict] = []

        with self.assertRaises(ValueError) as context:
            self.collator(data)

        self.assertEqual(
            str(context.exception), "not enough values to unpack (expected 2, got 0)"
        )


if __name__ == "__main__":
    unittest.main()
