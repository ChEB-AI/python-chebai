import hashlib
import unittest
from typing import Any, List, Tuple

import numpy as np
import pandas as pd

from chebai.preprocessing.datasets.chebi import ChEBIOver50


class TestChebiDynamicDataSplits(unittest.TestCase):
    """
    Test dynamic splits implementation's consistency for ChEBIOver50 dataset.

    Attributes:
        chebi_50_v231 (ChEBIOver50): Instance of ChEBIOver50 with ChEBI version 231.
        chebi_50_v231_vt200 (ChEBIOver50): Instance of ChEBIOver50 with ChEBI version 231 and train version 200.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """
        Set up class method to initialize instances of ChEBIOver50 and generate data.
        """
        cls.chebi_50_v231 = ChEBIOver50(chebi_version=231)
        cls.chebi_50_v231_vt200 = ChEBIOver50(
            chebi_version=231, chebi_version_train=200
        )
        cls._generate_chebi_class_data(cls.chebi_50_v231)
        cls._generate_chebi_class_data(cls.chebi_50_v231_vt200)

    def testDynamicDataSplitsConsistency(self) -> None:
        """
        Test Dynamic Data Splits consistency across multiple runs.
        """
        # Dynamic Data Splits in Run 1
        train_hash_1, val_hash_1, test_hash_1 = self._get_hashed_splits()

        self.chebi_50_v231.dynamic_df_train = None
        # Dynamic Data Splits in Run 2
        train_hash_2, val_hash_2, test_hash_2 = self._get_hashed_splits()

        # Check all splits are matching in both runs
        self.assertEqual(train_hash_1, train_hash_2, "Train data hashes do not match.")
        self.assertEqual(val_hash_1, val_hash_2, "Validation data hashes do not match.")
        self.assertEqual(test_hash_1, test_hash_2, "Test data hashes do not match.")

    def test_same_ids_and_in_test_sets(self) -> None:
        """
        Check if test sets of both classes have the same IDs.
        """
        v231_ids = set(self.chebi_50_v231.dynamic_split_dfs["test"]["ident"])
        v231_vt200_ids = set(
            self.chebi_50_v231_vt200.dynamic_split_dfs["test"]["ident"]
        )

        self.assertEqual(
            v231_ids, v231_vt200_ids, "Test sets do not have the same IDs."
        )

    def test_labels_vector_size_in_test_sets(self) -> None:
        """
        Check if test sets of both classes have different sizes/shapes of labels.
        """
        v231_labels_shape = len(
            self.chebi_50_v231.dynamic_split_dfs["test"]["labels"].iloc[0]
        )
        v231_vt200_label_shape = len(
            self.chebi_50_v231_vt200.dynamic_split_dfs["test"]["labels"].iloc[0]
        )

        self.assertEqual(
            v231_labels_shape,
            v231_vt200_label_shape,
            "Test sets have different sizes of labels",
        )

    def test_no_overlaps_in_chebi_v231_vt200(self) -> None:
        """
        Test the overlaps for the ChEBIOver50(chebi_version=231, chebi_version_train=200) dataset.
        """
        train_set = self.chebi_50_v231_vt200.dynamic_split_dfs["train"]
        val_set = self.chebi_50_v231_vt200.dynamic_split_dfs["validation"]
        test_set = self.chebi_50_v231_vt200.dynamic_split_dfs["test"]

        train_set_ids = train_set["ident"].tolist()
        val_set_ids = val_set["ident"].tolist()
        test_set_ids = test_set["ident"].tolist()

        # Get the overlap between data splits based on IDs
        self.overlaps_train_val_ids = self.get_overlaps(train_set_ids, val_set_ids)
        self.overlaps_train_test_ids = self.get_overlaps(train_set_ids, test_set_ids)
        self.overlaps_val_test_ids = self.get_overlaps(val_set_ids, test_set_ids)

        self.assertEqual(
            len(self.overlaps_train_val_ids),
            0,
            "Duplicate entities present in Train and Validation set based on IDs",
        )
        self.assertEqual(
            len(self.overlaps_train_test_ids),
            0,
            "Duplicate entities present in Train and Test set based on IDs",
        )
        self.assertEqual(
            len(self.overlaps_val_test_ids),
            0,
            "Duplicate entities present in Validation and Test set based on IDs",
        )

    def _get_hashed_splits(self) -> Tuple[str, str, str]:
        """
        Returns hashed dynamic data splits.

        Returns:
            Tuple[str, str, str]: Hashes for train, validation, and test data splits.
        """
        chebi_class_obj = self.chebi_50_v231

        # Get dynamic splits from class variables
        train_data = chebi_class_obj.dynamic_split_dfs["train"]
        val_data = chebi_class_obj.dynamic_split_dfs["validation"]
        test_data = chebi_class_obj.dynamic_split_dfs["test"]

        # Get hashes for each split
        train_hash = self.compute_hash(train_data)
        val_hash = self.compute_hash(val_data)
        test_hash = self.compute_hash(test_data)

        return train_hash, val_hash, test_hash

    @staticmethod
    def compute_hash(data: pd.DataFrame) -> str:
        """
        Returns hash for the given data partition.

        Args:
            data (pd.DataFrame): DataFrame containing data to be hashed.

        Returns:
            str: Hash computed for the DataFrame.
        """
        data_for_hashing = data.map(TestChebiDynamicDataSplits.convert_to_hashable)
        return hashlib.md5(
            pd.util.hash_pandas_object(data_for_hashing, index=True).values
        ).hexdigest()

    @staticmethod
    def convert_to_hashable(item: Any) -> Any:
        """
        Convert lists and numpy arrays within the DataFrame to tuples for hashing.

        Args:
            item (Any): Item to convert to a hashable form.

        Returns:
            Any: Hashable representation of the input item.
        """
        if isinstance(item, list):
            return tuple(item)
        elif isinstance(item, np.ndarray):
            return tuple(item.tolist())
        else:
            return item

    @staticmethod
    def _generate_chebi_class_data(chebi_class_obj: ChEBIOver50) -> None:
        """
        Generate ChEBI class data if not already generated.

        Args:
            chebi_class_obj (ChEBIOver50): Instance of ChEBIOver50 class.
        """
        chebi_class_obj.prepare_data()
        chebi_class_obj.setup()

    @staticmethod
    def get_overlaps(list_1: List[Any], list_2: List[Any]) -> List[Any]:
        """
        Get overlaps between two lists.

        Args:
            list_1 (List[Any]): First list.
            list_2 (List[Any]): Second list.

        Returns:
            List[Any]: List of elements present in both lists.
        """
        overlap = []
        for element in list_1:
            if element in list_2:
                overlap.append(element)
        return overlap


if __name__ == "__main__":
    unittest.main()
