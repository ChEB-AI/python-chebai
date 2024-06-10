import hashlib
import unittest

import numpy as np
import pandas as pd

from chebai.preprocessing.datasets.chebi import ChEBIOver50


class TestChebiDynamicDataSplits(unittest.TestCase):
    """Test dynamic splits implementation's consistency"""

    @classmethod
    def setUpClass(cls):
        cls.chebi_50_v231 = ChEBIOver50(chebi_version=231)
        cls.chebi_50_v231_vt200 = ChEBIOver50(
            chebi_version=231, chebi_version_train=200
        )
        cls._generate_chebi_class_data(cls.chebi_50_v231)
        cls._generate_chebi_class_data(cls.chebi_50_v231_vt200)

    def testDynamicDataSplitsConsistency(self):
        """Test Dynamic Data Splits consistency across every run"""

        # Dynamic Data Splits in Run 1
        train_hash_1, val_hash_1, test_hash_1 = self._get_hashed_splits()

        # Dynamic Data Splits in Run 2
        train_hash_2, val_hash_2, test_hash_2 = self._get_hashed_splits()

        # Check all splits are matching in both runs
        self.assertEqual(train_hash_1, train_hash_2, "Train data hashes do not match.")
        self.assertEqual(val_hash_1, val_hash_2, "Validation data hashes do not match.")
        self.assertEqual(test_hash_1, test_hash_2, "Test data hashes do not match.")

    def test_same_ids_and_in_test_sets(self):
        """Check if test sets of both classes have same IDs"""

        v231_ids = set(self.chebi_50_v231.dynamic_split_dfs["test"]["ident"])
        v231_vt200_ids = set(
            self.chebi_50_v231_vt200.dynamic_split_dfs["test"]["ident"]
        )

        self.assertEqual(
            v231_ids, v231_vt200_ids, "Test sets do not have the same IDs."
        )

    def test_labels_vector_size_in_test_sets(self):
        """Check if test sets of both classes have different size/shape of labels"""

        v231_labels_shape = len(
            self.chebi_50_v231.dynamic_split_dfs["test"]["labels"].iloc[0]
        )
        v231_vt200_label_shape = len(
            self.chebi_50_v231_vt200.dynamic_split_dfs["test"]["labels"].iloc[0]
        )

        self.assertEqual(
            v231_labels_shape,
            v231_vt200_label_shape,
            "Test sets have the different size of labels",
        )

    def test_no_overlaps_in_chebi_v231_vt200(self):
        """Test the overlaps for the ChEBIOver50(chebi_version=231, chebi_version_train=200)"""
        train_set = self.chebi_50_v231_vt200.dynamic_split_dfs["train"]
        val_set = self.chebi_50_v231_vt200.dynamic_split_dfs["validation"]
        test_set = self.chebi_50_v231_vt200.dynamic_split_dfs["test"]

        train_set_ids = train_set["ident"].tolist()
        val_set_ids = val_set["ident"].tolist()
        test_set_ids = test_set["ident"].tolist()

        # ----- Get the overlap between data splits based on IDs -----
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

    def _get_hashed_splits(self):
        """Returns hashed dynamic data splits"""

        # Get the raw/processed data if missing
        chebi_class_obj = ChEBIOver50(seed=42)
        self._generate_chebi_class_data(chebi_class_obj)

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
    def compute_hash(data):
        """Returns hash for the given data partition"""
        data_for_hashing = data.map(TestChebiDynamicDataSplits.convert_to_hashable)
        return hashlib.md5(
            pd.util.hash_pandas_object(data_for_hashing, index=True).values
        ).hexdigest()

    @staticmethod
    def convert_to_hashable(item):
        """To Convert lists and numpy arrays within the DataFrame to tuples for hashing"""
        if isinstance(item, list):
            return tuple(item)
        elif isinstance(item, np.ndarray):
            return tuple(item.tolist())
        else:
            return item

    @staticmethod
    def _generate_chebi_class_data(chebi_class_obj):
        # Get the raw/processed data if missing
        chebi_class_obj.prepare_data()
        chebi_class_obj.setup()

    @staticmethod
    def get_overlaps(list_1, list_2):
        overlap = []
        for element in list_1:
            if element in list_2:
                overlap.append(element)
        return overlap


if __name__ == "__main__":
    unittest.main()
