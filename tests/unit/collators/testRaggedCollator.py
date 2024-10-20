import unittest
from typing import Dict, List, Tuple

import torch

from chebai.preprocessing.collate import RaggedCollator
from chebai.preprocessing.structures import XYData


class TestRaggedCollator(unittest.TestCase):
    """
    Unit tests for the RaggedCollator class.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """
        Set up the test environment by initializing a RaggedCollator instance.
        """
        cls.collator = RaggedCollator()

    def test_call_with_valid_data(self) -> None:
        """
        Test the __call__ method with valid ragged data to ensure features, labels, and masks are correctly handled.
        """
        data: List[Dict] = [
            {"features": [1, 2], "labels": [True, False], "ident": "sample1"},
            {"features": [3, 4, 5], "labels": [False, True, True], "ident": "sample2"},
            {"features": [6], "labels": [True], "ident": "sample3"},
        ]

        result: XYData = self.collator(data)

        expected_x = torch.tensor([[1, 2, 0], [3, 4, 5], [6, 0, 0]])
        expected_y = torch.tensor(
            [[True, False, False], [False, True, True], [True, False, False]]
        )
        expected_mask_for_x = torch.tensor(
            [[True, True, False], [True, True, True], [True, False, False]]
        )
        expected_lens_for_x = torch.tensor([2, 3, 1])

        self.assertTrue(
            torch.equal(result.x, expected_x),
            "The feature tensor 'x' does not match the expected output.",
        )
        self.assertTrue(
            torch.equal(result.y, expected_y),
            "The label tensor 'y' does not match the expected output.",
        )
        self.assertTrue(
            torch.equal(
                result.additional_fields["model_kwargs"]["mask"], expected_mask_for_x
            ),
            "The mask tensor does not match the expected output.",
        )
        self.assertTrue(
            torch.equal(
                result.additional_fields["model_kwargs"]["lens"], expected_lens_for_x
            ),
            "The lens tensor does not match the expected output.",
        )
        self.assertEqual(
            result.additional_fields["idents"],
            ("sample1", "sample2", "sample3"),
            "The identifiers do not match the expected output.",
        )

    def test_call_with_missing_entire_labels(self) -> None:
        """
        Test the __call__ method with data where some samples are missing labels.
        """
        data: List[Dict] = [
            {"features": [1, 2], "labels": [True, False], "ident": "sample1"},
            {"features": [3, 4, 5], "labels": None, "ident": "sample2"},
            {"features": [6], "labels": [True], "ident": "sample3"},
        ]

        result: XYData = self.collator(data)

        # https://github.com/ChEB-AI/python-chebai/pull/48#issuecomment-2324393829
        expected_x = torch.tensor([[1, 2, 0], [3, 4, 5], [6, 0, 0]])
        expected_y = torch.tensor(
            [[True, False], [True, False]]
        )  # True -> 1, False -> 0
        expected_mask_for_x = torch.tensor(
            [[True, True, False], [True, True, True], [True, False, False]]
        )
        expected_lens_for_x = torch.tensor([2, 3, 1])

        self.assertTrue(
            torch.equal(result.x, expected_x),
            "The feature tensor 'x' does not match the expected output when labels are missing.",
        )
        self.assertTrue(
            torch.equal(result.y, expected_y),
            "The label tensor 'y' does not match the expected output when labels are missing.",
        )
        self.assertTrue(
            torch.equal(
                result.additional_fields["model_kwargs"]["mask"], expected_mask_for_x
            ),
            "The mask tensor does not match the expected output when labels are missing.",
        )
        self.assertTrue(
            torch.equal(
                result.additional_fields["model_kwargs"]["lens"], expected_lens_for_x
            ),
            "The lens tensor does not match the expected output when labels are missing.",
        )
        self.assertEqual(
            result.additional_fields["loss_kwargs"]["non_null_labels"],
            [0, 2],
            "The non-null labels list does not match the expected output.",
        )
        self.assertEqual(
            len(result.additional_fields["loss_kwargs"]["non_null_labels"]),
            result.y.shape[1],
            "The length of non null labels list must match with target label variable size",
        )
        self.assertEqual(
            result.additional_fields["idents"],
            ("sample1", "sample2", "sample3"),
            "The identifiers do not match the expected output when labels are missing.",
        )

    def test_call_with_none_in_labels(self) -> None:
        """
        Test the __call__ method with data where one of the elements in the labels is None.
        """
        data: List[Dict] = [
            {"features": [1, 2], "labels": [None, True], "ident": "sample1"},
            {"features": [3, 4, 5], "labels": [True, False], "ident": "sample2"},
            {"features": [6], "labels": [True], "ident": "sample3"},
        ]

        result: XYData = self.collator(data)

        expected_x = torch.tensor([[1, 2, 0], [3, 4, 5], [6, 0, 0]])
        expected_y = torch.tensor(
            [[False, True], [True, False], [True, False]]
        )  # None -> False
        expected_mask_for_x = torch.tensor(
            [[True, True, False], [True, True, True], [True, False, False]]
        )
        expected_lens_for_x = torch.tensor([2, 3, 1])

        self.assertTrue(
            torch.equal(result.x, expected_x),
            "The feature tensor 'x' does not match the expected output when labels contain None.",
        )
        self.assertTrue(
            torch.equal(result.y, expected_y),
            "The label tensor 'y' does not match the expected output when labels contain None.",
        )
        self.assertTrue(
            torch.equal(
                result.additional_fields["model_kwargs"]["mask"], expected_mask_for_x
            ),
            "The mask tensor does not match the expected output when labels contain None.",
        )
        self.assertTrue(
            torch.equal(
                result.additional_fields["model_kwargs"]["lens"], expected_lens_for_x
            ),
            "The lens tensor does not match the expected output when labels contain None.",
        )
        self.assertEqual(
            result.additional_fields["idents"],
            ("sample1", "sample2", "sample3"),
            "The identifiers do not match the expected output when labels contain None.",
        )

    def test_call_with_empty_data(self) -> None:
        """
        Test the __call__ method with an empty list to ensure it raises an error.
        """
        data: List[Dict] = []

        with self.assertRaises(
            Exception, msg="Expected an Error when no data is provided"
        ):
            self.collator(data)

    def test_process_label_rows(self) -> None:
        """
        Test the process_label_rows method to ensure it pads label sequences correctly.
        """
        labels: Tuple = ([True, False], [False, True, True], [True])

        result: torch.Tensor = self.collator.process_label_rows(labels)

        expected_output = torch.tensor(
            [[True, False, False], [False, True, True], [True, False, False]]
        )

        self.assertTrue(
            torch.equal(result, expected_output),
            "The processed label rows tensor does not match the expected output.",
        )


if __name__ == "__main__":
    unittest.main()
