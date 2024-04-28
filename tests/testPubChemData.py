import unittest
import os
import torch
from chebai.preprocessing.datasets.pubchem import PubChem


class TestPubChemData(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.pubChem = PubChem()
        cls.getDataSplitsOverlaps()

    @classmethod
    def getDataSplitsOverlaps(cls):
        """Get the overlap between data splits"""
        processed_path = os.path.join(os.getcwd(), cls.pubChem.processed_dir)
        print(f"Checking Data from - {processed_path}")

        train_set = torch.load(os.path.join(processed_path, "train.pt"))
        val_set = torch.load(os.path.join(processed_path, "validation.pt"))
        test_set = torch.load(os.path.join(processed_path, "test.pt"))

        train_smiles, train_smiles_ids = cls.get_features_ids(train_set)
        val_smiles, val_smiles_ids = cls.get_features_ids(val_set)
        test_smiles, test_smiles_ids = cls.get_features_ids(test_set)

        # ----- Get the overlap between data splits based on smiles tokens/features -----

        # train_smiles.append(val_smiles[0])
        # train_smiles.append(test_smiles[0])
        # val_smiles.append(test_smiles[0])

        cls.overlaps_train_val = cls.get_overlaps(train_smiles, val_smiles)
        cls.overlaps_train_test = cls.get_overlaps(train_smiles, test_smiles)
        cls.overlaps_val_test = cls.get_overlaps(val_smiles, test_smiles)

        # ----- Get the overlap between data splits based on IDs -----

        # train_smiles_ids.append(val_smiles_ids[0])
        # train_smiles_ids.append(test_smiles_ids[0])
        # val_smiles_ids.append(test_smiles_ids[0])

        cls.overlaps_train_val_ids = cls.get_overlaps(train_smiles_ids, val_smiles_ids)
        cls.overlaps_train_test_ids = cls.get_overlaps(
            train_smiles_ids, test_smiles_ids
        )
        cls.overlaps_val_test_ids = cls.get_overlaps(val_smiles_ids, test_smiles_ids)

    @staticmethod
    def get_features_ids(data_split):
        """Returns SMILES features/tokens and SMILES IDs from the data"""
        smiles_features, smiles_ids = [], []
        for entry in data_split:
            smiles_features.append(entry["features"])
            smiles_ids.append(entry["ident"])

        return smiles_features, smiles_ids

    @staticmethod
    def get_overlaps(list_1, list_2):
        overlap = []
        for element in list_1:
            if element in list_2:
                overlap.append(element)
        return overlap

    def test_train_val_overlap_based_on_smiles(self):
        """Check that train-val splits are performed correctly i.e.every entity
        only appears in one of the train and validation set based on smiles tokens/features
        """
        self.assertEqual(
            len(self.overlaps_train_val),
            0,
            "Duplicate entities present in Train and Validation set based on SMILES",
        )

    def test_train_test_overlap_based_on_smiles(self):
        """Check that train-test splits are performed correctly i.e.every entity
        only appears in one of the train and test set based on smiles tokens/features"""
        self.assertEqual(
            len(self.overlaps_train_test),
            0,
            "Duplicate entities present in Train and Test set based on SMILES",
        )

    def test_val_test_overlap_based_on_smiles(self):
        """Check that val-test splits are performed correctly i.e.every entity
        only appears in one of the validation and test set based on smiles tokens/features
        """
        self.assertEqual(
            len(self.overlaps_val_test),
            0,
            "Duplicate entities present in Validation and Test set based on SMILES",
        )

    def test_train_val_overlap_based_on_ids(self):
        """Check that train-val splits are performed correctly i.e.every entity
        only appears in one of the train and validation set based on smiles IDs"""
        self.assertEqual(
            len(self.overlaps_train_val_ids),
            0,
            "Duplicate entities present in Train and Validation set based on IDs",
        )

    def test_train_test_overlap_based_on_ids(self):
        """Check that train-test splits are performed correctly i.e.every entity
        only appears in one of the train and test set based on smiles IDs"""
        self.assertEqual(
            len(self.overlaps_train_test_ids),
            0,
            "Duplicate entities present in Train and Test set based on IDs",
        )

    def test_val_test_overlap_based_on_ids(self):
        """Check that val-test splits are performed correctly i.e.every entity
        only appears in one of the validation and test set based on smiles IDs"""
        self.assertEqual(
            len(self.overlaps_val_test_ids),
            0,
            "Duplicate entities present in Validation and Test set based on IDs",
        )


if __name__ == "__main__":
    unittest.main()
