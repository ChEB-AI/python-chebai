import unittest
from typing import Tuple
from unittest.mock import MagicMock, PropertyMock, patch

import pandas as pd

from chebai.preprocessing.datasets.base import _DynamicDataset


class TestDynamicDataset(unittest.TestCase):
    """
    Test case for _DynamicDataset functionality, ensuring correct data splits and integrity
    of train, validation, and test datasets.
    """

    @classmethod
    @patch.multiple(_DynamicDataset, __abstractmethods__=frozenset())
    @patch.object(_DynamicDataset, "base_dir", new_callable=PropertyMock)
    @patch.object(_DynamicDataset, "_name", new_callable=PropertyMock)
    @patch("os.makedirs", return_value=None)
    def setUpClass(
        cls,
        mock_makedirs,
        mock_base_dir_property: PropertyMock,
        mock_name_property: PropertyMock,
    ) -> None:
        """
        Set up a base instance of _DynamicDataset for testing with mocked properties.
        """

        # Mocking properties
        mock_base_dir_property.return_value = "MockedBaseDirPropertyDynamicDataset"
        mock_name_property.return_value = "MockedNamePropertyDynamicDataset"

        # Mock Data Reader
        ReaderMock = MagicMock()
        ReaderMock.name.return_value = "MockedReader"
        _DynamicDataset.READER = ReaderMock

        # Creating an instance of the dataset
        cls.dataset: _DynamicDataset = _DynamicDataset()

        # Dataset with a balanced distribution of labels
        X = [
            [1, 2],
            [3, 4],
            [5, 6],
            [7, 8],
            [9, 10],
            [11, 12],
            [13, 14],
            [15, 16],
            [17, 18],
            [19, 20],
            [21, 22],
            [23, 24],
            [25, 26],
            [27, 28],
            [29, 30],
            [31, 32],
        ]
        y = [
            [False, False],
            [False, True],
            [True, False],
            [True, True],
            [False, False],
            [False, True],
            [True, False],
            [True, True],
            [False, False],
            [False, True],
            [True, False],
            [True, True],
            [False, False],
            [False, True],
            [True, False],
            [True, True],
        ]
        cls.data_df = pd.DataFrame(
            {"ident": [f"id{i + 1}" for i in range(len(X))], "features": X, "labels": y}
        )

    def test_get_test_split_valid(self) -> None:
        """
        Test splitting the dataset into train and test sets and verify balance and non-overlap.
        """
        self.dataset.train_split = 0.5
        # Test size will be 0.25 * 16 = 4
        train_df, test_df = self.dataset.get_test_split(self.data_df, seed=0)

        # Assert the correct number of rows in train and test sets
        self.assertEqual(len(train_df), 12, "Train set should contain 12 samples.")
        self.assertEqual(len(test_df), 4, "Test set should contain 4 samples.")

        # Check positive and negative label counts in train and test sets
        train_pos_count, train_neg_count = self.get_positive_negative_labels_counts(
            train_df
        )
        test_pos_count, test_neg_count = self.get_positive_negative_labels_counts(
            test_df
        )

        # Ensure that the train and test sets have balanced positives and negatives
        self.assertEqual(
            train_pos_count, train_neg_count, "Train set labels should be balanced."
        )
        self.assertEqual(
            test_pos_count, test_neg_count, "Test set labels should be balanced."
        )

        # Assert there is no overlap between train and test sets
        train_idents = set(train_df["ident"])
        test_idents = set(test_df["ident"])
        self.assertEqual(
            len(train_idents.intersection(test_idents)),
            0,
            "Train and test sets should not overlap.",
        )

    def test_get_test_split_missing_labels(self) -> None:
        """
        Test the behavior when the 'labels' column is missing in the dataset.
        """
        df_missing_labels = pd.DataFrame({"ident": ["id1", "id2"]})
        with self.assertRaises(
            KeyError, msg="Expected KeyError when 'labels' column is missing."
        ):
            self.dataset.get_test_split(df_missing_labels)

    def test_get_test_split_seed_consistency(self) -> None:
        """
        Test that splitting the dataset with the same seed produces consistent results.
        """
        train_df1, test_df1 = self.dataset.get_test_split(self.data_df, seed=42)
        train_df2, test_df2 = self.dataset.get_test_split(self.data_df, seed=42)

        pd.testing.assert_frame_equal(
            train_df1,
            train_df2,
            obj="Train sets should be identical for the same seed.",
        )
        pd.testing.assert_frame_equal(
            test_df1, test_df2, obj="Test sets should be identical for the same seed."
        )

    def test_get_train_val_splits_given_test(self) -> None:
        """
        Test splitting the dataset into train and validation sets and verify balance and non-overlap.
        """
        self.dataset.use_inner_cross_validation = False
        self.dataset.train_split = 0.5
        df_train_main, test_df = self.dataset.get_test_split(self.data_df, seed=0)
        train_df, val_df = self.dataset.get_train_val_splits_given_test(
            df_train_main, test_df, seed=42
        )

        # Ensure there is no overlap between train and test sets
        train_idents = set(train_df["ident"])
        test_idents = set(test_df["ident"])
        self.assertEqual(
            len(train_idents.intersection(test_idents)),
            0,
            "Train and test sets should not overlap.",
        )

        # Ensure there is no overlap between validation and test sets
        val_idents = set(val_df["ident"])
        self.assertEqual(
            len(val_idents.intersection(test_idents)),
            0,
            "Validation and test sets should not overlap.",
        )

        # Ensure there is no overlap between train and validation sets
        self.assertEqual(
            len(train_idents.intersection(val_idents)),
            0,
            "Train and validation sets should not overlap.",
        )

        # Check positive and negative label counts in train and validation sets
        train_pos_count, train_neg_count = self.get_positive_negative_labels_counts(
            train_df
        )
        val_pos_count, val_neg_count = self.get_positive_negative_labels_counts(val_df)

        # Ensure that the train and validation sets have balanced positives and negatives
        self.assertEqual(
            train_pos_count, train_neg_count, "Train set labels should be balanced."
        )
        self.assertEqual(
            val_pos_count, val_neg_count, "Validation set labels should be balanced."
        )

    def test_get_train_val_splits_given_test_consistency(self) -> None:
        """
        Test that splitting the dataset into train and validation sets with the same seed produces consistent results.
        """
        test_df = self.data_df.iloc[12:]  # Assume rows 12 onward are for testing
        train_df1, val_df1 = self.dataset.get_train_val_splits_given_test(
            self.data_df, test_df, seed=42
        )
        train_df2, val_df2 = self.dataset.get_train_val_splits_given_test(
            self.data_df, test_df, seed=42
        )

        pd.testing.assert_frame_equal(
            train_df1,
            train_df2,
            obj="Train sets should be identical for the same seed.",
        )
        pd.testing.assert_frame_equal(
            val_df1,
            val_df2,
            obj="Validation sets should be identical for the same seed.",
        )

    def test_get_test_split_stratification(self) -> None:
        """
        Test that the split into train and test sets maintains the stratification of labels.
        """
        self.dataset.train_split = 0.5
        train_df, test_df = self.dataset.get_test_split(self.data_df, seed=0)

        number_of_labels = len(self.data_df["labels"][0])

        # Check the label distribution in the original dataset
        original_pos_count, original_neg_count = (
            self.get_positive_negative_labels_counts(self.data_df)
        )
        total_count = len(self.data_df) * number_of_labels

        # Calculate the expected proportions
        original_pos_proportion = original_pos_count / total_count
        original_neg_proportion = original_neg_count / total_count

        # Check the label distribution in the train set
        train_pos_count, train_neg_count = self.get_positive_negative_labels_counts(
            train_df
        )
        train_total_count = len(train_df) * number_of_labels

        # Calculate the train set proportions
        train_pos_proportion = train_pos_count / train_total_count
        train_neg_proportion = train_neg_count / train_total_count

        # Assert that the proportions are similar to the original dataset
        self.assertAlmostEqual(
            train_pos_proportion,
            original_pos_proportion,
            places=1,
            msg="Train set labels should maintain original positive label proportion.",
        )
        self.assertAlmostEqual(
            train_neg_proportion,
            original_neg_proportion,
            places=1,
            msg="Train set labels should maintain original negative label proportion.",
        )

        # Check the label distribution in the test set
        test_pos_count, test_neg_count = self.get_positive_negative_labels_counts(
            test_df
        )
        test_total_count = len(test_df) * number_of_labels

        # Calculate the test set proportions
        test_pos_proportion = test_pos_count / test_total_count
        test_neg_proportion = test_neg_count / test_total_count

        # Assert that the proportions are similar to the original dataset
        self.assertAlmostEqual(
            test_pos_proportion,
            original_pos_proportion,
            places=1,
            msg="Test set labels should maintain original positive label proportion.",
        )
        self.assertAlmostEqual(
            test_neg_proportion,
            original_neg_proportion,
            places=1,
            msg="Test set labels should maintain original negative label proportion.",
        )

    def test_get_train_val_splits_given_test_stratification(self) -> None:
        """
        Test that the split into train and validation sets maintains the stratification of labels.
        """
        self.dataset.use_inner_cross_validation = False
        self.dataset.train_split = 0.5
        df_train_main, test_df = self.dataset.get_test_split(self.data_df, seed=0)
        train_df, val_df = self.dataset.get_train_val_splits_given_test(
            df_train_main, test_df, seed=42
        )

        number_of_labels = len(self.data_df["labels"][0])

        # Check the label distribution in the original dataset
        original_pos_count, original_neg_count = (
            self.get_positive_negative_labels_counts(self.data_df)
        )
        total_count = len(self.data_df) * number_of_labels

        # Calculate the expected proportions
        original_pos_proportion = original_pos_count / total_count
        original_neg_proportion = original_neg_count / total_count

        # Check the label distribution in the train set
        train_pos_count, train_neg_count = self.get_positive_negative_labels_counts(
            train_df
        )
        train_total_count = len(train_df) * number_of_labels

        # Calculate the train set proportions
        train_pos_proportion = train_pos_count / train_total_count
        train_neg_proportion = train_neg_count / train_total_count

        # Assert that the proportions are similar to the original dataset
        self.assertAlmostEqual(
            train_pos_proportion,
            original_pos_proportion,
            places=1,
            msg="Train set labels should maintain original positive label proportion.",
        )
        self.assertAlmostEqual(
            train_neg_proportion,
            original_neg_proportion,
            places=1,
            msg="Train set labels should maintain original negative label proportion.",
        )

        # Check the label distribution in the validation set
        val_pos_count, val_neg_count = self.get_positive_negative_labels_counts(val_df)
        val_total_count = len(val_df) * number_of_labels

        # Calculate the validation set proportions
        val_pos_proportion = val_pos_count / val_total_count
        val_neg_proportion = val_neg_count / val_total_count

        # Assert that the proportions are similar to the original dataset
        self.assertAlmostEqual(
            val_pos_proportion,
            original_pos_proportion,
            places=1,
            msg="Validation set labels should maintain original positive label proportion.",
        )
        self.assertAlmostEqual(
            val_neg_proportion,
            original_neg_proportion,
            places=1,
            msg="Validation set labels should maintain original negative label proportion.",
        )

    @staticmethod
    def get_positive_negative_labels_counts(df: pd.DataFrame) -> Tuple[int, int]:
        """
        Count the number of True and False values within the labels column.

        Args:
            df (pd.DataFrame): The DataFrame containing the 'labels' column.

        Returns:
            Tuple[int, int]: A tuple containing the counts of True and False values, respectively.
        """
        true_count = sum(sum(label) for label in df["labels"])
        false_count = sum(len(label) - sum(label) for label in df["labels"])
        return true_count, false_count


if __name__ == "__main__":
    unittest.main()
