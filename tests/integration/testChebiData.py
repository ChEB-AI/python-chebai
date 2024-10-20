import unittest
from typing import List

import pandas as pd

from chebai.preprocessing.datasets.chebi import ChEBIOver50


class TestChebiData(unittest.TestCase):
    """
    Test case for ChEBIOver50 dataset integrity, focusing on data splits and overlaps.

    Attributes:
        overlaps_train_val (List): List of overlapping entities between train and validation splits based on SMILES.
        overlaps_train_test (List): List of overlapping entities between train and test splits based on SMILES.
        overlaps_val_test (List): List of overlapping entities between validation and test splits based on SMILES.
        overlaps_train_val_ids (List): List of overlapping entity IDs between train and validation splits.
        overlaps_train_test_ids (List): List of overlapping entity IDs between train and test splits.
        overlaps_val_test_ids (List): List of overlapping entity IDs between validation and test splits.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """
        Set up class method to initialize ChEBIOver50 instance and generate data splits and overlaps.
        """
        cls.getDataSplitsOverlaps()

    @classmethod
    def getDataSplitsOverlaps(cls) -> None:
        """
        Get the overlap between data splits based on SMILES and IDs.
        """
        chebi_class_obj = ChEBIOver50()
        # Get the raw/processed data if missing
        chebi_class_obj.prepare_data()
        chebi_class_obj.setup()

        train_set = chebi_class_obj.dynamic_split_dfs["train"]
        val_set = chebi_class_obj.dynamic_split_dfs["validation"]
        test_set = chebi_class_obj.dynamic_split_dfs["test"]

        train_smiles, train_smiles_ids = cls.get_features_ids(train_set)
        val_smiles, val_smiles_ids = cls.get_features_ids(val_set)
        test_smiles, test_smiles_ids = cls.get_features_ids(test_set)

        # Get the overlap between data splits based on smiles tokens/features
        cls.overlaps_train_val = cls.get_overlaps(train_smiles, val_smiles)
        cls.overlaps_train_test = cls.get_overlaps(train_smiles, test_smiles)
        cls.overlaps_val_test = cls.get_overlaps(val_smiles, test_smiles)

        # Get the overlap between data splits based on IDs
        cls.overlaps_train_val_ids = cls.get_overlaps(train_smiles_ids, val_smiles_ids)
        cls.overlaps_train_test_ids = cls.get_overlaps(
            train_smiles_ids, test_smiles_ids
        )
        cls.overlaps_val_test_ids = cls.get_overlaps(val_smiles_ids, test_smiles_ids)

    @staticmethod
    def get_features_ids(data_split_df: pd.DataFrame) -> tuple[List, List]:
        """
        Returns SMILES features/tokens and SMILES IDs from the data.

        Args:
            data_split_df: DataFrame containing data to extract features and IDs from.

        Returns:
            Tuple of lists: SMILES features/tokens list and SMILES IDs list.
        """
        smiles_features = data_split_df["features"].tolist()
        smiles_ids = data_split_df["ident"].tolist()

        return smiles_features, smiles_ids

    @staticmethod
    def get_overlaps(list_1: List, list_2: List) -> List:
        """
        Get overlaps between two lists.

        Args:
            list_1: First list.
            list_2: Second list.

        Returns:
            List: List of elements present in both lists.
        """
        overlap = []
        for element in list_1:
            if element in list_2:
                overlap.append(element)
        return overlap

    @unittest.expectedFailure
    def test_train_val_overlap_based_on_smiles(self) -> None:
        """
        Check that train-val splits are performed correctly i.e.every entity
        only appears in one of the train and validation set based on smiles tokens/features
        """
        self.assertEqual(
            len(self.overlaps_train_val),
            0,
            "Duplicate entities present in Train and Validation set based on SMILES",
        )

    @unittest.expectedFailure
    def test_train_test_overlap_based_on_smiles(self) -> None:
        """
        Check that train-test splits are performed correctly i.e.every entity
        only appears in one of the train and test set based on smiles tokens/features
        """
        self.assertEqual(
            len(self.overlaps_train_test),
            0,
            "Duplicate entities present in Train and Test set based on SMILES",
        )

    @unittest.expectedFailure
    def test_val_test_overlap_based_on_smiles(self) -> None:
        """
        Check that val-test splits are performed correctly i.e.every entity
        only appears in one of the validation and test set based on smiles tokens/features
        """
        self.assertEqual(
            len(self.overlaps_val_test),
            0,
            "Duplicate entities present in Validation and Test set based on SMILES",
        )

    def test_train_val_overlap_based_on_ids(self) -> None:
        """
        Check that train-val splits are performed correctly i.e.every entity
        only appears in one of the train and validation set based on smiles IDs
        """
        self.assertEqual(
            len(self.overlaps_train_val_ids),
            0,
            "Duplicate entities present in Train and Validation set based on IDs",
        )

    def test_train_test_overlap_based_on_ids(self) -> None:
        """
        Check that train-test splits are performed correctly i.e.every entity
        only appears in one of the train and test set based on smiles IDs
        """
        self.assertEqual(
            len(self.overlaps_train_test_ids),
            0,
            "Duplicate entities present in Train and Test set based on IDs",
        )

    def test_val_test_overlap_based_on_ids(self) -> None:
        """
        Check that val-test splits are performed correctly i.e.every entity
        only appears in one of the validation and test set based on smiles IDs
        """
        self.assertEqual(
            len(self.overlaps_val_test_ids),
            0,
            "Duplicate entities present in Validation and Test set based on IDs",
        )


if __name__ == "__main__":
    unittest.main()
