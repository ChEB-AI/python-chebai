import os

import torch
from lightning.pytorch.core.datamodule import LightningDataModule
from torch.utils.data import DataLoader

from chebai.preprocessing.collate import RaggedCollator


class MyLightningDataModule(LightningDataModule):
    def __init__(self):
        super().__init__()
        self._num_of_labels = None
        self._feature_vector_size = None
        self.collator = RaggedCollator()

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        self._num_of_labels = 10
        self._feature_vector_size = 20
        print(f"Number of labels: {self._num_of_labels}")
        print(f"Number of features: {self._feature_vector_size}")

    @property
    def num_of_labels(self):
        return self._num_of_labels

    @property
    def feature_vector_size(self):
        return self._feature_vector_size

    @property
    def classes_txt_file_path(self) -> str:
        return os.path.join("tests", "unit", "cli", "classification_labels.txt")

    def train_dataloader(self):
        assert self.feature_vector_size is not None, "feature_vector_size must be set"
        # Dummy dataset for example purposes

        datalist = [
            {
                "features": torch.randn(self._feature_vector_size),
                "labels": torch.randint(0, 2, (self._num_of_labels,), dtype=torch.bool),
                "ident": i,
                "group": None,
            }
            for i in range(100)
        ]

        return DataLoader(datalist, batch_size=32, collate_fn=self.collator)
