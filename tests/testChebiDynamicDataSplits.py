import unittest
import hashlib
import pandas as pd
import numpy as np
from chebai.preprocessing.datasets.chebi import ChEBIOver50


class TestChebiDynamicDataSplits(unittest.TestCase):

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

    def _get_hashed_splits(self):
        """Returns hashed dynamic data splits"""

        # Get the raw/processed data if missing
        chebi_class_obj = ChEBIOver50(seed=42)
        chebi_class_obj.prepare_data()
        chebi_class_obj.setup()

        # Get dynamic splits from class variables
        train_data = chebi_class_obj.dynamic_split_class_variables_df["train"]
        val_data = chebi_class_obj.dynamic_split_class_variables_df["validation"]
        test_data = chebi_class_obj.dynamic_split_class_variables_df["test"]

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


if __name__ == "__main__":
    unittest.main()
