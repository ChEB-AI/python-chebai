from typing import List, Union
import multiprocessing as mp
import os

from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch
import tqdm

from chebai.preprocessing import reader as dr


class XYBaseDataModule(pl.LightningDataModule):
    READER = dr.DataReader

    def __init__(self, batch_size=1, tran_split=0.85, reader_kwargs=None, **kwargs):
        super().__init__(**kwargs)
        if reader_kwargs is None:
            reader_kwargs = dict()
        self.reader = self.READER(**reader_kwargs)
        self.train_split = tran_split
        self.batch_size = batch_size
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)

    @property
    def identifier(self):
        return (self.reader.name(),)

    @property
    def full_identifier(self):
        return (self._name, *self.identifier)

    @property
    def processed_dir(self):
        return os.path.join("data", self._name, "processed", *self.identifier)

    @property
    def raw_dir(self):
        return os.path.join("data", self._name, "raw")

    @property
    def _name(self):
        raise NotImplementedError

    def dataloader(self, kind, **kwargs):

        dataset = torch.load(os.path.join(self.processed_dir, f"{kind}.pt"))

        return DataLoader(
            dataset,
            collate_fn=self.reader.collater,
            batch_size=self.batch_size,
            **kwargs,
        )

    @staticmethod
    def _load_dict(input_file_path):
        with open(input_file_path, "r") as input_file:
            for row in input_file:
                smiles, labels = row.split("\t")
                yield dict(features=smiles, labels=labels)

    @staticmethod
    def _get_data_size(input_file_path):
        with open(input_file_path, "r") as f:
            return sum(1 for _ in f)

    def _load_data_from_file(self, path):
        lines = self._get_data_size(path)
        print(f"Processing {lines} lines...")
        data = [self.reader.to_data(d) for d in tqdm.tqdm(self._load_dict(path), total=lines) if d["features"] is not None]
        return data

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return self.dataloader("train", shuffle=True, **kwargs)

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return self.dataloader("validation", shuffle=False, **kwargs)

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return self.dataloader("test", shuffle=False, **kwargs)

    def setup(self, **kwargs):
        if any(
            not os.path.isfile(os.path.join(self.processed_dir, f))
            for f in self.processed_file_names
        ):
            self.setup_processed()

    def setup_processed(self):
        raise NotImplementedError

    @property
    def processed_file_names(self):
        raise NotImplementedError

    @property
    def label_number(self):
        """
        Number of labels
        :return:
        Returns -1 for seq2seq encoding, otherwise the number of labels
        """
        raise NotImplementedError
