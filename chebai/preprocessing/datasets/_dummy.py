# This file is for developers only

__all__ = []  # Nothing should be imported from this file


import random

import numpy as np
from torch.utils.data import DataLoader, Dataset

from chebai.preprocessing.datasets import XYBaseDataModule
from chebai.preprocessing.reader import ChemDataReader


class _DummyDataModule(XYBaseDataModule):

    READER = ChemDataReader

    def __init__(self, num_of_labels: int, feature_vector_size: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._num_of_labels = num_of_labels
        self._feature_vector_size = feature_vector_size
        assert self._num_of_labels is not None
        assert self._feature_vector_size is not None

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        pass

    @property
    def num_of_labels(self):
        return self._num_of_labels

    @property
    def feature_vector_size(self):
        return self._feature_vector_size

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        dataset = _DummyDataset(100, self.num_of_labels, self.feature_vector_size)
        return DataLoader(
            dataset,
            collate_fn=self.reader.collator,
            batch_size=self.batch_size,
            **kwargs,
        )

    def test_dataloader(self, *args, **kwargs) -> DataLoader:
        dataset = _DummyDataset(20, self.num_of_labels, self.feature_vector_size)
        return DataLoader(
            dataset,
            collate_fn=self.reader.collator,
            batch_size=self.batch_size,
            **kwargs,
        )

    def val_dataloader(self, *args, **kwargs) -> DataLoader:
        dataset = _DummyDataset(10, self.num_of_labels, self.feature_vector_size)
        return DataLoader(
            dataset,
            collate_fn=self.reader.collator,
            batch_size=self.batch_size,
            **kwargs,
        )

    @property
    def _name(self) -> str:
        return "_DummyDataModule"


class _DummyDataset(Dataset):
    def __init__(self, num_samples: int, num_labels: int, feature_vector_size: int):
        self.num_samples = num_samples
        self.num_labels = num_labels
        self.feature_vector_size = feature_vector_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            "features": np.random.randint(
                10, 100, size=self.feature_vector_size
            ),  # Random feature vector
            "labels": np.random.choice(
                [False, True], size=self.num_labels
            ),  # Random boolean labels
            "ident": random.randint(1, 40000),  # Random identifier
            "group": None,  # Default group value
        }


if __name__ == "__main__":
    dataset = _DummyDataset(num_samples=100, num_labels=5, feature_vector_size=20)
    for i in range(10):
        print(dataset[i])
