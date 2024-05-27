import unittest
import os
import yaml
import hashlib
import pandas as pd
import numpy as np


class TestChebiDynamicDataSplits(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.chebi_class_object = cls.getChebiDataClassConfig()

    def testDynamicDataSplitsConsistency(self):
        """Test Dynamic Data Splits consistency across every run"""

        # Dynamic Data Splits in Run 1
        train_data_1, val_data_1, test_data_1 = self.get_train_val_test_splits()
        train_hash_1 = self.compute_hash(train_data_1)
        val_hash_1 = self.compute_hash(val_data_1)
        test_hash_1 = self.compute_hash(test_data_1)

        # Dynamic Data Splits in Run 2
        train_data_2, val_data_2, test_data_2 = self.get_train_val_test_splits()
        train_hash_2 = self.compute_hash(train_data_2)
        val_hash_2 = self.compute_hash(val_data_2)
        test_hash_2 = self.compute_hash(test_data_2)

        # Check all splits are matching in both runs
        self.assertEqual(train_hash_1, train_hash_2, "Train data hashes do not match.")
        self.assertEqual(val_hash_1, val_hash_2, "Validation data hashes do not match.")
        self.assertEqual(test_hash_1, test_hash_2, "Test data hashes do not match.")

    def get_train_val_test_splits(self):
        """Returns Dynamic Data splits consisting of train, validation and test set"""
        data = self.chebi_class_object.load_processed_data("data")
        df = pd.DataFrame(data)
        train_df, df_test = self.chebi_class_object.get_test_split(
            df, seed=self.chebi_class_object.data_split_seed
        )
        df_train, df_val = self.chebi_class_object.get_train_val_splits_given_test(
            train_df, df_test, seed=self.chebi_class_object.data_split_seed
        )
        return df_train, df_val, df_test

    @staticmethod
    def compute_hash(data):
        """Returns hash for the given data partition"""
        data_for_hashing = data.applymap(TestChebiDynamicDataSplits.convert_to_hashable)
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
    def getChebiDataClassConfig():
        """Import the respective class and instantiate with given version from the config"""
        CONFIG_FILE_NAME = "chebi50.yml"
        with open(
            os.path.join("configs", "data", f"{CONFIG_FILE_NAME}"), "r"
        ) as yaml_file:
            config = yaml.safe_load(yaml_file)

        class_path = config["class_path"]
        init_args = config.get("init_args", {})

        module, class_name = class_path.rsplit(".", 1)
        module = __import__(module, fromlist=[class_name])
        class_ = getattr(module, class_name)

        return class_(**init_args)


if __name__ == "__main__":
    unittest.main()
