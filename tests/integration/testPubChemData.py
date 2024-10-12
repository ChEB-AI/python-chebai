import os
import unittest
from typing import Dict, List, Tuple

import torch

from chebai.preprocessing.datasets.pubchem import PubChem


class TestPubChemData(unittest.TestCase):
    """
    Unit tests for PubChem dataset preprocessing.

    Attributes:
        pubChem (PubChem): Instance of PubChem dataset handler.
        overlaps_train_val (List): List of overlaps between training and validation sets based on SMILES features.
        overlaps_train_test (List): List of overlaps between training and test sets based on SMILES features.
        overlaps_val_test (List): List of overlaps between validation and test sets based on SMILES features.
        overlaps_train_val_ids (List): List of overlaps between training and validation sets based on SMILES IDs.
        overlaps_train_test_ids (List): List of overlaps between training and test sets based on SMILES IDs.
        overlaps_val_test_ids (List): List of overlaps between validation and test sets based on SMILES IDs.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """
        Set up PubChem dataset and compute overlaps between data splits.
        """
        cls.pubChem = PubChem()
        cls.getDataSplitsOverlaps()

    @classmethod
    def getDataSplitsOverlaps(cls) -> None:
        """
        Get the overlap between data splits based on SMILES features and IDs.
        """
        processed_path = os.path.join(os.getcwd(), cls.pubChem.processed_dir)
        print(f"Checking Data from - {processed_path}")

        train_set = torch.load(
            os.path.join(processed_path, "train.pt"), weights_only=False
        )
        val_set = torch.load(
            os.path.join(processed_path, "validation.pt"), weights_only=False
        )
        test_set = torch.load(
            os.path.join(processed_path, "test.pt"), weights_only=False
        )

        train_smiles, train_smiles_ids = cls.get_features_ids(train_set)
        val_smiles, val_smiles_ids = cls.get_features_ids(val_set)
        test_smiles, test_smiles_ids = cls.get_features_ids(test_set)

        # Get overlaps based on SMILES features
        cls.overlaps_train_val = cls.get_overlaps(train_smiles, val_smiles)
        cls.overlaps_train_test = cls.get_overlaps(train_smiles, test_smiles)
        cls.overlaps_val_test = cls.get_overlaps(val_smiles, test_smiles)

        # Get overlaps based on SMILES IDs
        cls.overlaps_train_val_ids = cls.get_overlaps(train_smiles_ids, val_smiles_ids)
        cls.overlaps_train_test_ids = cls.get_overlaps(
            train_smiles_ids, test_smiles_ids
        )
        cls.overlaps_val_test_ids = cls.get_overlaps(val_smiles_ids, test_smiles_ids)

    @staticmethod
    def get_features_ids(data_split: List[Dict]) -> Tuple[List, List]:
        """
        Returns SMILES features/tokens and SMILES IDs from the data.

        Args:
            data_split (List[Dict]): List of dictionaries containing SMILES features and IDs.

        Returns:
            Tuple[List, List]: Tuple of lists containing SMILES features and SMILES IDs.
        """
        smiles_features, smiles_ids = [], []
        for entry in data_split:
            smiles_features.append(entry["features"])
            smiles_ids.append(entry["ident"])

        return smiles_features, smiles_ids

    @staticmethod
    def get_overlaps(list_1: List, list_2: List) -> List:
        """
        Get overlaps between two lists.

        Args:
            list_1 (List): First list.
            list_2 (List): Second list.

        Returns:
            List: List of elements common to both input lists.
        """
        overlap = []
        for element in list_1:
            if element in list_2:
                overlap.append(element)
        return overlap

    def test_train_val_overlap_based_on_smiles(self) -> None:
        """
        Check that train-val splits are performed correctly based on SMILES features.
        """
        self.assertEqual(
            len(self.overlaps_train_val),
            0,
            "Duplicate entities present in Train and Validation set based on SMILES",
        )

    def test_train_test_overlap_based_on_smiles(self) -> None:
        """
        Check that train-test splits are performed correctly based on SMILES features.
        """
        self.assertEqual(
            len(self.overlaps_train_test),
            0,
            "Duplicate entities present in Train and Test set based on SMILES",
        )

    def test_val_test_overlap_based_on_smiles(self) -> None:
        """
        Check that val-test splits are performed correctly based on SMILES features.
        """
        self.assertEqual(
            len(self.overlaps_val_test),
            0,
            "Duplicate entities present in Validation and Test set based on SMILES",
        )

    def test_train_val_overlap_based_on_ids(self) -> None:
        """
        Check that train-val splits are performed correctly based on SMILES IDs.
        """
        self.assertEqual(
            len(self.overlaps_train_val_ids),
            0,
            "Duplicate entities present in Train and Validation set based on IDs",
        )

    def test_train_test_overlap_based_on_ids(self) -> None:
        """
        Check that train-test splits are performed correctly based on SMILES IDs.
        """
        self.assertEqual(
            len(self.overlaps_train_test_ids),
            0,
            "Duplicate entities present in Train and Test set based on IDs",
        )

    def test_val_test_overlap_based_on_ids(self) -> None:
        """
        Check that val-test splits are performed correctly based on SMILES IDs.
        """
        self.assertEqual(
            len(self.overlaps_val_test_ids),
            0,
            "Duplicate entities present in Validation and Test set based on IDs",
        )


if __name__ == "__main__":
    unittest.main()
