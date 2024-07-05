import os
import random
import typing
from typing import List, Union

import lightning as pl
import torch
import tqdm
from lightning.pytorch.core.datamodule import LightningDataModule
from lightning_utilities.core.rank_zero import rank_zero_info
from torch.utils.data import DataLoader

from chebai.preprocessing import reader as dr


class XYBaseDataModule(LightningDataModule):
    """
    Base class for data modules.

    This class provides a base implementation for loading and preprocessing datasets.
    It inherits from `LightningDataModule` and defines common properties and methods for data loading and processing.

    Args:
        batch_size (int): The batch size for data loading. Default is 1.
        train_split (float): The ratio of training data to total data and of test data to (validation + test) data. Default is 0.85.
        reader_kwargs (dict): Additional keyword arguments to be passed to the data reader. Default is None.
        prediction_kind (str): The kind of prediction to be performed (only relevant for the predict_dataloader). Default is "test".
        data_limit (int): The maximum number of data samples to load. If set to None, the complete dataset will be used. Default is None.
        label_filter (int): The index of the label to filter. Default is None.
        balance_after_filter (float): The ratio of negative samples to positive samples after filtering. Default is None.
        num_workers (int): The number of worker processes for data loading. Default is 1.
        chebi_version (int): The version of ChEBI to use. Default is 200.
        inner_k_folds (int): The number of folds for inner cross-validation. Use -1 to disable inner cross-validation. Default is -1.
        fold_index (int): The index of the fold to use for training and validation. Default is None.
        base_dir (str): The base directory for storing processed and raw data. Default is None.
        **kwargs: Additional keyword arguments.

    Attributes:
        READER (DataReader): The data reader class to use.
        reader (DataReader): An instance of the data reader class.
        train_split (float): The ratio of training data to total data.
        batch_size (int): The batch size for data loading.
        prediction_kind (str): The kind of prediction to be performed.
        data_limit (int): The maximum number of data samples to load.
        label_filter (int): The index of the label to filter.
        balance_after_filter (float): The ratio of negative samples to positive samples after filtering.
        num_workers (int): The number of worker processes for data loading.
        chebi_version (int): The version of ChEBI to use.
        inner_k_folds (int): The number of folds for inner cross-validation. If it is less than to, no cross-validation will be performed.
        fold_index (int): The index of the fold to use for training and validation (only relevant for cross-validation)
        _base_dir (str): The base directory for storing processed and raw data.
        raw_dir (str): The directory for storing raw data.
        processed_dir (str): The directory for storing processed data.
        fold_dir (str): The name of the directory where the folds from inner cross-validation are stored.
        _name (str): The name of the data module.

    """

    READER = dr.DataReader

    def __init__(
        self,
        batch_size=1,
        train_split=0.85,
        reader_kwargs=None,
        prediction_kind="test",
        data_limit: typing.Optional[int] = None,
        label_filter: typing.Optional[int] = None,
        balance_after_filter: typing.Optional[float] = None,
        num_workers: int = 1,
        chebi_version: int = 200,
        inner_k_folds: int = -1,  # use inner cross-validation if > 1
        fold_index: typing.Optional[int] = None,
        base_dir=None,
        **kwargs,
    ):
        super().__init__()
        if reader_kwargs is None:
            reader_kwargs = dict()
        self.reader = self.READER(**reader_kwargs)
        self.train_split = train_split
        self.batch_size = batch_size
        self.prediction_kind = prediction_kind
        self.data_limit = data_limit
        self.label_filter = label_filter
        assert (balance_after_filter is not None) or (
            self.label_filter is None
        ), "Filter balancing requires a filter"
        self.balance_after_filter = balance_after_filter
        self.num_workers = num_workers
        self.chebi_version = chebi_version
        assert type(inner_k_folds) is int
        self.inner_k_folds = inner_k_folds
        self.use_inner_cross_validation = (
            inner_k_folds > 1
        )  # only use cv if there are at least 2 folds
        assert (
            fold_index is None or self.use_inner_cross_validation is not None
        ), "fold_index can only be set if cross validation is used"
        if fold_index is not None and self.inner_k_folds is not None:
            assert (
                fold_index < self.inner_k_folds
            ), "fold_index can't be larger than the total number of folds"
        self.fold_index = fold_index
        self._base_dir = base_dir
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        if self.use_inner_cross_validation:
            os.makedirs(os.path.join(self.raw_dir, self.fold_dir), exist_ok=True)
            os.makedirs(os.path.join(self.processed_dir, self.fold_dir), exist_ok=True)
        self.save_hyperparameters()

    @property
    def identifier(self):
        """Identifier for the dataset."""
        return (self.reader.name(),)

    @property
    def full_identifier(self):
        """Full identifier for the dataset."""
        return (self._name, *self.identifier)

    @property
    def base_dir(self):
        """Common base directory for processed and raw directories"""
        if self._base_dir is not None:
            return self._base_dir
        return os.path.join("data", self._name)

    @property
    def processed_dir(self):
        """name of dir where the processed data is stored"""
        return os.path.join(self.base_dir, "processed", *self.identifier)

    @property
    def raw_dir(self):
        """name of dir where the raw data is stored"""
        return os.path.join(self.base_dir, "raw")

    @property
    def fold_dir(self):
        """name of dir where the folds from inner cross-validation (i.e., the train and val sets) are stored"""
        return f"cv_{self.inner_k_folds}_fold"

    @property
    def _name(self):
        raise NotImplementedError

    def _filter_labels(self, row):
        """filter labels based on label_filter"""
        row["labels"] = [row["labels"][self.label_filter]]
        return row

    def load_processed_data(self, kind: str = None, filename: str = None) -> List:
        """
        Load processed data from a file.

        Args:
            kind (str, optional): The kind of dataset to load such as "train", "val" or "test". Defaults to None.
            filename (str, optional): The name of the file to load the dataset from. Defaults to None.

        Returns:
            List: The loaded processed data.

        Raises:
            ValueError: If both kind and filename are None.
        """
        if kind is None and filename is None:
            raise ValueError(
                "Either kind or filename is required to load the correct dataset, both are None"
            )
        # if both kind and filename are given, use filename
        if kind is not None and filename is None:
            try:
                # processed_file_names_dict is only implemented for _ChEBIDataExtractor
                if self.use_inner_cross_validation and kind != "test":
                    filename = self.processed_file_names_dict[
                        f"fold_{self.fold_index}_{kind}"
                    ]
                else:
                    filename = self.processed_file_names_dict[kind]
            except NotImplementedError:
                filename = f"{kind}.pt"
        return torch.load(os.path.join(self.processed_dir, filename))

    def dataloader(self, kind, **kwargs) -> DataLoader:
        """
        Returns a DataLoader object for the specified kind (train, val or test) of data.

        Args:
            kind (str): The kind indicates whether it is a train, val or test data to load.
            **kwargs: Additional keyword arguments.

        Returns:
            DataLoader: A DataLoader object.

        """
        dataset = self.load_processed_data(kind)
        if "ids" in kwargs:
            ids = kwargs.pop("ids")
            _dataset = []
            for i in range(len(dataset)):
                if i in ids:
                    _dataset.append(dataset[i])
            dataset = _dataset
        if self.label_filter is not None:
            original_len = len(dataset)
            dataset = [self._filter_labels(r) for r in dataset]
            positives = [r for r in dataset if r["labels"][0]]
            negatives = [r for r in dataset if not r["labels"][0]]
            if self.balance_after_filter is not None:
                negative_length = min(
                    original_len, int(len(positives) * self.balance_after_filter)
                )
                dataset = positives + negatives[:negative_length]
            else:
                dataset = positives + negatives
            random.shuffle(dataset)
        if self.data_limit is not None:
            dataset = dataset[: self.data_limit]
        return DataLoader(
            dataset,
            collate_fn=self.reader.collater,
            batch_size=self.batch_size,
            **kwargs,
        )

    @staticmethod
    def _load_dict(input_file_path):
        """
        Load data from a file and return a dictionary.

        Args:
            input_file_path (str): The path to the input file.

        Yields:
            dict: A dictionary containing the features and labels.
        """
        with open(input_file_path, "r") as input_file:
            for row in input_file:
                smiles, labels = row.split("\t")
                yield dict(features=smiles, labels=labels)

    @staticmethod
    def _get_data_size(input_file_path):
        with open(input_file_path, "r") as f:
            return sum(1 for _ in f)

    def _load_data_from_file(self, path):
        """
        Load data from a file and return a list of dictionaries.

        Args:
            path (str): The path to the input file.

        Returns:
            List: A list of dictionaries containing the features and labels.
        """
        lines = self._get_data_size(path)
        print(f"Processing {lines} lines...")
        data = [
            self.reader.to_data(d)
            for d in tqdm.tqdm(self._load_dict(path), total=lines)
            if d["features"] is not None
        ]
        # filter for missing features in resulting data
        data = [val for val in data if val["features"] is not None]

        return data

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return self.dataloader(
            "train",
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
            **kwargs,
        )

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return self.dataloader(
            "validation",
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
            **kwargs,
        )

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return self.dataloader("test", shuffle=False, **kwargs)

    def predict_dataloader(
        self, *args, **kwargs
    ) -> Union[DataLoader, List[DataLoader]]:
        return self.dataloader(self.prediction_kind, shuffle=False, **kwargs)

    def setup(self, **kwargs):
        rank_zero_info(f"Check for processed data in {self.processed_dir}")
        rank_zero_info(f"Cross-validation enabled: {self.use_inner_cross_validation}")
        if any(
            not os.path.isfile(os.path.join(self.processed_dir, f))
            for f in self.processed_file_names
        ):
            self.setup_processed()

        if not ("keep_reader" in kwargs and kwargs["keep_reader"]):
            self.reader.on_finish()

    def setup_processed(self):
        raise NotImplementedError

    @property
    def processed_file_names(self):
        raise NotImplementedError

    @property
    def raw_file_names(self):
        raise NotImplementedError

    @property
    def processed_file_names_dict(self) -> dict:
        raise NotImplementedError

    @property
    def raw_file_names_dict(self) -> dict:
        raise NotImplementedError

    @property
    def label_number(self):
        """
        Number of labels
        :return:
        Returns -1 for seq2seq encoding, otherwise the number of labels
        """
        raise NotImplementedError


class MergedDataset(XYBaseDataModule):
    MERGED = []

    @property
    def _name(self):
        return "+".join(s._name for s in self.subsets)

    def __init__(self, batch_size=1, train_split=0.85, reader_kwargs=None, **kwargs):
        if reader_kwargs is None:
            reader_kwargs = [None for _ in self.MERGED]
        self.train_split = train_split
        self.batch_size = batch_size
        self.subsets = [
            s(train_split=train_split, reader_kwargs=kws)
            for s, kws in zip(self.MERGED, reader_kwargs)
        ]
        self.reader = self.subsets[0].reader
        os.makedirs(self.processed_dir, exist_ok=True)
        super(pl.LightningDataModule, self).__init__(**kwargs)

    def prepare_data(self):
        for s in self.subsets:
            s.prepare_data()

    def setup(self, **kwargs):
        for s in self.subsets:
            s.setup(**kwargs)

    def dataloader(self, kind, **kwargs):
        subdatasets = [
            torch.load(os.path.join(s.processed_dir, f"{kind}.pt"))
            for s in self.subsets
        ]
        dataset = [
            self._process_data(i, d)
            for i, (s, lim) in enumerate(zip(subdatasets, self.limits))
            for d in (s if lim is None else s[:lim])
        ]
        return DataLoader(
            dataset,
            collate_fn=self.reader.collater,
            batch_size=self.batch_size,
            **kwargs,
        )

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return self.dataloader("train", shuffle=True, **kwargs)

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return self.dataloader("validation", shuffle=False, **kwargs)

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return self.dataloader("test", shuffle=False, **kwargs)

    def _process_data(self, subset_id, data):
        return dict(
            features=data["features"], labels=data["labels"], ident=data["ident"]
        )

    def setup_processed(self):
        pass

    @property
    def processed_file_names(self):
        return ["test.pt", "train.pt", "validation.pt"]

    @property
    def label_number(self):
        return self.subsets[0].label_number

    @property
    def limits(self):
        return None
