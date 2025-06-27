import os
import random
from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import lightning as pl
import networkx as nx
import pandas as pd
import torch
import tqdm
from iterstrat.ml_stratifiers import (
    MultilabelStratifiedKFold,
    MultilabelStratifiedShuffleSplit,
)
from lightning.pytorch.core.datamodule import LightningDataModule
from lightning_utilities.core.rank_zero import rank_zero_info
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader

from chebai.preprocessing import reader as dr


class XYBaseDataModule(LightningDataModule):
    """
    Base class for data modules.

    This class provides a base implementation for loading and preprocessing datasets.
    It inherits from `LightningDataModule` and defines common properties and methods for data loading and processing.

    Args:
        batch_size (int): The batch size for data loading. Default is 1.
        test_split (float): The ratio of test data to total data. Default is 0.1.
        validation_split (float): The ratio of validation data to total data. Default is 0.05.
        reader_kwargs (dict): Additional keyword arguments to be passed to the data reader. Default is None.
        prediction_kind (str): The kind of prediction to be performed (only relevant for the predict_dataloader). Default is "test".
        data_limit (Optional[int]): The maximum number of data samples to load. If set to None, the complete dataset will be used. Default is None.
        label_filter (Optional[int]): The index of the label to filter. Default is None.
        balance_after_filter (Optional[float]): The ratio of negative samples to positive samples after filtering. Default is None.
        num_workers (int): The number of worker processes for data loading. Default is 1.
        chebi_version (int): The version of ChEBI to use. Default is 200.
        inner_k_folds (int): The number of folds for inner cross-validation. Use -1 to disable inner cross-validation. Default is -1.
        fold_index (Optional[int]): The index of the fold to use for training and validation. Default is None.
        base_dir (Optional[str]): The base directory for storing processed and raw data. Default is None.
        **kwargs: Additional keyword arguments.

    Attributes:
        READER (DataReader): The data reader class to use.
        reader (DataReader): An instance of the data reader class.
        test_split (float): The ratio of test data to total data.
        validation_split (float): The ratio of validation data to total data.
        batch_size (int): The batch size for data loading.
        prediction_kind (str): The kind of prediction to be performed.
        data_limit (Optional[int]): The maximum number of data samples to load.
        label_filter (Optional[int]): The index of the label to filter.
        balance_after_filter (Optional[float]): The ratio of negative samples to positive samples after filtering.
        num_workers (int): The number of worker processes for data loading.
        chebi_version (int): The version of ChEBI to use.
        inner_k_folds (int): The number of folds for inner cross-validation. If it is less than to, no cross-validation will be performed.
        fold_index (Optional[int]): The index of the fold to use for training and validation (only relevant for cross-validation).
        _base_dir (Optional[str]): The base directory for storing processed and raw data.
        raw_dir (str): The directory for storing raw data.
        processed_dir (str): The directory for storing processed data.
        fold_dir (str): The name of the directory where the folds from inner cross-validation are stored.
        _name (str): The name of the data module.

    """

    READER = dr.DataReader

    def __init__(
        self,
        batch_size: int = 1,
        test_split: Optional[float] = 0.1,
        validation_split: Optional[float] = 0.05,
        reader_kwargs: Optional[dict] = None,
        prediction_kind: str = "test",
        data_limit: Optional[int] = None,
        label_filter: Optional[int] = None,
        balance_after_filter: Optional[float] = None,
        num_workers: int = 1,
        chebi_version: int = 200,
        inner_k_folds: int = -1,  # use inner cross-validation if > 1
        fold_index: Optional[int] = None,
        base_dir: Optional[str] = None,
        n_token_limit: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        if reader_kwargs is None:
            reader_kwargs = dict()
        self.reader = self.READER(**reader_kwargs)
        self.test_split = test_split
        self.validation_split = validation_split

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
        self.n_token_limit = n_token_limit
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        if self.use_inner_cross_validation:
            os.makedirs(os.path.join(self.raw_dir, self.fold_dir), exist_ok=True)
            os.makedirs(os.path.join(self.processed_dir, self.fold_dir), exist_ok=True)
        self.save_hyperparameters()

        self._num_of_labels = None
        self._feature_vector_size = None
        self._prepare_data_flag = 1
        self._setup_data_flag = 1

    @property
    def num_of_labels(self):
        assert self._num_of_labels is not None, "num of labels must be set"
        return self._num_of_labels

    @property
    def feature_vector_size(self):
        assert (
            self._feature_vector_size is not None
        ), "size of feature vector must be set"
        return self._feature_vector_size

    @property
    def identifier(self) -> tuple:
        """Identifier for the dataset."""
        return (self.reader.name(),)

    @property
    def full_identifier(self) -> tuple:
        """Full identifier for the dataset."""
        return (self._name, *self.identifier)

    @property
    def base_dir(self) -> str:
        """Common base directory for processed and raw directories."""
        if self._base_dir is not None:
            return self._base_dir
        return os.path.join("data", self._name)

    @property
    def processed_dir_main(self) -> str:
        """Name of the directory where processed (but not tokenized) data is stored."""
        return os.path.join(self.base_dir, "processed")

    @property
    def processed_dir(self) -> str:
        """Name of the directory where the processed and tokenized data is stored."""
        return os.path.join(self.processed_dir_main, *self.identifier)

    @property
    def raw_dir(self) -> str:
        """Name of the directory where the raw data is stored."""
        return os.path.join(self.base_dir, "raw")

    @property
    def fold_dir(self) -> str:
        """Name of the directory where the folds from inner cross-validation (i.e., the train and val sets) are stored."""
        return f"cv_{self.inner_k_folds}_fold"

    @property
    @abstractmethod
    def _name(self) -> str:
        """
        Abstract property representing the name of the data module.

        This property should be implemented in subclasses to provide a unique name for the data module.
        The name is used to create subdirectories within the base directory or `processed_dir_main`
        for storing relevant data associated with this module.

        Returns:
            str: The name of the data module.
        """
        pass

    def _filter_labels(self, row: dict) -> dict:
        """
        Filter labels based on `label_filter`.
        This method selects specific labels from the `labels` list within the row dictionary
        according to the index or indices provided by the `label_filter` attribute of the class.

        Args:
            row (dict): A dictionary containing the row data.

        Returns:
            dict: The filtered row data.
        """
        row["labels"] = [row["labels"][self.label_filter]]
        return row

    def load_processed_data(
        self, kind: Optional[str] = None, filename: Optional[str] = None
    ) -> List:
        """
        Load processed data from a file. Either the kind or the filename has to be provided. If both are provided, the
        filename is used.

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
        return torch.load(
            os.path.join(self.processed_dir, filename), weights_only=False
        )

    def dataloader(self, kind: str, **kwargs) -> DataLoader:
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
            collate_fn=self.reader.collator,
            batch_size=self.batch_size,
            **kwargs,
        )

    @staticmethod
    def _load_dict(
        input_file_path: str,
    ) -> Generator[Dict[str, Any], None, None]:
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
    def _get_data_size(input_file_path: str) -> int:
        """
        Get the number of lines in a file.

        Args:
            input_file_path (str): The path to the input file.

        Returns:
            int: The number of lines in the file.
        """
        with open(input_file_path, "r") as f:
            return sum(1 for _ in f)

    def _load_data_from_file(self, path: str) -> List[Dict[str, Any]]:
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
        # filter for missing features in resulting data, keep features length below token limit
        data = [
            val
            for val in data
            if val["features"] is not None
            and (
                self.n_token_limit is None or len(val["features"]) <= self.n_token_limit
            )
        ]

        return data

    def train_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        """
        Returns the train DataLoader.

        Args:
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            DataLoader: A DataLoader object for training data.
        """
        return self.dataloader(
            "train",
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
            **kwargs,
        )

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        """
        Returns the validation DataLoader.

        Args:
            *args: Additional positional arguments (unused).
            **kwargs: Additional keyword arguments, passed to dataloader().

        Returns:
            Union[DataLoader, List[DataLoader]]: A DataLoader object for validation data.
        """
        return self.dataloader(
            "validation",
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
            **kwargs,
        )

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        """
        Returns the test DataLoader.

        Args:
            *args: Additional positional arguments (unused).
            **kwargs: Additional keyword arguments, passed to dataloader().

        Returns:
            Union[DataLoader, List[DataLoader]]: A DataLoader object for test data.
        """
        return self.dataloader("test", shuffle=False, **kwargs)

    def predict_dataloader(
        self, *args, **kwargs
    ) -> Union[DataLoader, List[DataLoader]]:
        """
        Returns the predict DataLoader.

        Args:
            *args: Additional positional arguments (unused).
            **kwargs: Additional keyword arguments, passed to dataloader().

        Returns:
            Union[DataLoader, List[DataLoader]]: A DataLoader object for prediction data.
        """
        return self.dataloader(self.prediction_kind, shuffle=False, **kwargs)

    def prepare_data(self, *args, **kwargs) -> None:
        if self._prepare_data_flag != 1:
            return

        self._prepare_data_flag += 1
        self._perform_data_preparation(*args, **kwargs)

    def _perform_data_preparation(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def setup(self, *args, **kwargs) -> None:
        """
        Setup the data module.

        This method checks for the processed data and sets up the data module for training, validation, and testing.

        Args:
            **kwargs: Additional keyword arguments.
        """
        if self._setup_data_flag != 1:
            return

        self._setup_data_flag += 1

        rank_zero_info(f"Check for processed data in {self.processed_dir}")
        rank_zero_info(f"Cross-validation enabled: {self.use_inner_cross_validation}")
        if any(
            not os.path.isfile(os.path.join(self.processed_dir, f))
            for f in self.processed_file_names
        ):
            self.setup_processed()

        self._after_setup(**kwargs)

    def _after_setup(self, **kwargs):
        """
        Finalize the setup process after ensuring the processed data is available.

        This method performs post-setup tasks like finalizing the reader and setting internal properties.
        """
        self.reader.on_finish()
        self._set_processed_data_props()

    def _set_processed_data_props(self):
        """
        Load processed data and extract metadata.

        Sets:
            - self._num_of_labels: Number of target labels in the dataset.
            - self._feature_vector_size: Maximum feature vector length across all data points.
        """
        data_pt = torch.load(
            os.path.join(self.processed_dir, self.processed_file_names_dict["data"]),
            weights_only=False,
        )

        self._num_of_labels = len(data_pt[0]["labels"])
        self._feature_vector_size = max(len(d["features"]) for d in data_pt)

        print(f"Number of labels for loaded data: {self._num_of_labels}")
        print(f"Feature vector size: {self._feature_vector_size}")

    def setup_processed(self):
        """
        Setup the processed data.

        This method should be implemented by subclasses to handle the specific setup of processed data.
        """
        raise NotImplementedError

    @property
    def processed_main_file_names_dict(self) -> dict:
        """
        Returns a dictionary mapping processed data file names.

        Returns:
            dict: A dictionary mapping dataset key to their respective file names.
                  For example, {"data": "data.pkl"}.
        """
        raise NotImplementedError

    @property
    def processed_main_file_names(self) -> List[str]:
        """
        Returns a list of file names for processed data (before tokenization).

        Returns:
            List[str]: A list of file names corresponding to the processed data.
        """
        return list(self.processed_main_file_names_dict.values())

    @property
    def processed_file_names_dict(self) -> dict:
        """
        Returns a dictionary for the processed and tokenized data files.

        Returns:
            dict: A dictionary mapping dataset keys to their respective file names.
                  For example, {"data": "data.pt"}.
        """
        raise NotImplementedError

    @property
    def processed_file_names(self) -> List[str]:
        """
        Returns a list of file names for processed data.

        Returns:
            List[str]: A list of file names corresponding to the processed data.
        """
        return list(self.processed_file_names_dict.values())

    @property
    def raw_file_names(self) -> List[str]:
        """
        Returns the list of raw file names.

        Returns:
            List[str]: The list of raw file names.
        """
        return list(self.raw_file_names_dict.values())

    @property
    def raw_file_names_dict(self) -> dict:
        """
        Returns the dictionary of raw file names (i.e., files that are directly obtained from an external source).

        This property should be implemented by subclasses to provide the dictionary of raw file names.

        Returns:
            dict: The dictionary of raw file names.
        """
        raise NotImplementedError


class MergedDataset(XYBaseDataModule):
    MERGED = []

    @property
    def _name(self) -> str:
        """
        Returns a concatenated name of all subset names.
        """
        return "+".join(s._name for s in self.subsets)

    def __init__(
        self,
        batch_size: int = 1,
        train_split: float = 0.85,
        reader_kwargs: Union[None, List[dict]] = None,
        **kwargs,
    ):
        """
        Args:
            batch_size (int): Batch size for data loaders.
            train_split (float): Fraction of data to use for training.
            reader_kwargs (Union[None, List[dict]]): Optional arguments for subset readers.
            **kwargs: Additional arguments to pass to LightningDataModule.
        """
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

    def _perform_data_preparation(self):
        """
        Placeholder for data preparation logic.
        """
        for s in self.subsets:
            s.prepare_data()

    def setup(self, **kwargs):
        """
        Setup the data module.

        This method checks for the processed data and sets up the data module for training, validation, and testing.

        Args:
            **kwargs: Additional keyword arguments.
        """
        if self._setup_data_flag != 1:
            return

        self._setup_data_flag += 1
        for s in self.subsets:
            s.setup(**kwargs)

        self._set_processed_data_props()

    def dataloader(self, kind: str, **kwargs) -> DataLoader:
        """
        Creates a DataLoader for a specific subset.

        Args:
            kind (str): Kind of data loader ('train', 'validation', or 'test').
            **kwargs: Additional arguments passed to DataLoader.

        Returns:
            DataLoader: DataLoader object for the specified subset.
        """
        subdatasets = [
            torch.load(os.path.join(s.processed_dir, f"{kind}.pt"), weights_only=False)
            for s in self.subsets
        ]
        dataset = [
            self._process_data(i, d)
            for i, (s, lim) in enumerate(zip(subdatasets, self.limits))
            for d in (s if lim is None else s[:lim])
        ]
        return DataLoader(
            dataset,
            collate_fn=self.reader.collator,
            batch_size=self.batch_size,
            **kwargs,
        )

    def train_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        """
        Returns the training DataLoader.
        """
        return self.dataloader("train", shuffle=True, **kwargs)

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        """
        Returns the validation DataLoader.
        """
        return self.dataloader("validation", shuffle=False, **kwargs)

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        """
        Returns the test DataLoader.
        """
        return self.dataloader("test", shuffle=False, **kwargs)

    def _process_data(self, subset_id: int, data: dict) -> dict:
        """
        Processes data from a subset.

        Args:
            subset_id (int): Index of the subset.
            data (dict): Data from the subset.

        Returns:
            dict: Processed data with 'features', 'labels', and 'ident' keys.
        """
        return dict(
            features=data["features"], labels=data["labels"], ident=data["ident"]
        )

    def setup_processed(self):
        """
        Placeholder for setup logic after data processing.
        """
        pass

    @property
    def processed_file_names(self) -> List[str]:
        """
        Returns the list of processed file names.
        """
        return ["test.pt", "train.pt", "validation.pt"]

    @property
    def limits(self):
        """
        Returns None, assuming no limits on data slicing.
        """
        return None


class _DynamicDataset(XYBaseDataModule, ABC):
    """
    A class for extracting and processing data from the given dataset.

    The processed and transformed data is stored in `data.pkl` and `data.pt` format as a whole respectively,
    rather than as separate  train, validation, and test splits, with dynamic splitting of data.pt occurring at runtime.
    The `_DynamicDataset` class manages data splits by either generating them during execution or retrieving them from
    a CSV file.
    If no split file path is provided, `_generate_dynamic_splits` creates the training, validation, and test splits
    from the encoded/transformed data, storing them in `_dynamic_df_train`, `_dynamic_df_val`, and `_dynamic_df_test`.
    When a split file path is provided, `_retrieve_splits_from_csv` loads splits from the CSV file, which must
    include 'id' and 'split' columns.
    The `dynamic_split_dfs` property ensures that the necessary splits are loaded as required.

    Args:
        dynamic_data_split_seed (int, optional): The seed for random data splitting. Defaults to 42.
        splits_file_path (str, optional): Path to the splits CSV file. Defaults to None.
        **kwargs: Additional keyword arguments passed to XYBaseDataModule.

    Attributes:
        dynamic_data_split_seed (int): The seed for random data splitting, default is 42.
        splits_file_path (Optional[str]): Path to the CSV file containing split assignments.
    """

    # ---- Index for columns of processed `data.pkl` (should be derived from `_graph_to_raw_dataset` method) ------
    _ID_IDX: int = None
    _DATA_REPRESENTATION_IDX: int = None
    _LABELS_START_IDX: int = None

    def __init__(
        self,
        **kwargs,
    ):
        super(_DynamicDataset, self).__init__(**kwargs)
        self.dynamic_data_split_seed = int(kwargs.get("seed", 42))  # default is 42
        # Class variables to store the dynamics splits
        self._dynamic_df_train = None
        self._dynamic_df_test = None
        self._dynamic_df_val = None
        # Path of csv file which contains a list of ids & their assignment to a dataset (either train,
        # validation or test).
        self.splits_file_path = self._validate_splits_file_path(
            kwargs.get("splits_file_path", None)
        )

    @staticmethod
    def _validate_splits_file_path(splits_file_path: Optional[str]) -> Optional[str]:
        """
        Validates the file in provided splits file path.

        Args:
            splits_file_path (Optional[str]): Path to the splits CSV file.

        Returns:
            Optional[str]: Validated splits file path if checks pass, None if splits_file_path is None.

        Raises:
            FileNotFoundError: If the splits file does not exist.
            ValueError: If splits file is empty or missing required columns ('id' and/or 'split'), or not a CSV file.
        """
        if splits_file_path is None:
            return None

        if not os.path.isfile(splits_file_path):
            raise FileNotFoundError(f"File {splits_file_path} does not exist")

        file_size = os.path.getsize(splits_file_path)
        if file_size == 0:
            raise ValueError(f"File {splits_file_path} is empty")

        # Check if the file has a CSV extension
        if not splits_file_path.lower().endswith(".csv"):
            raise ValueError(f"File {splits_file_path} is not a CSV file")

        # Read the first row of CSV file into a DataFrame
        splits_df = pd.read_csv(splits_file_path, nrows=1)

        # Check if 'id' and 'split' columns are in the DataFrame
        required_columns = {"id", "split"}
        if not required_columns.issubset(splits_df.columns):
            raise ValueError(
                f"CSV file {splits_file_path} is missing required columns ('id' and/or 'split')."
            )

        return splits_file_path

    # ------------------------------ Phase: Prepare data -----------------------------------
    def _perform_data_preparation(self, *args: Any, **kwargs: Any) -> None:
        """
        Prepares the data for the dataset.

        This method checks for the presence of raw data in the specified directory.
        If the raw data is missing, it fetches the ontology and creates a dataframe and saves it to a data.pkl file.

        The resulting dataframe/pickle file is expected to contain columns with the following structure:
            - Column at index `self._ID_IDX`: ID of data instance
            - Column at index `self._DATA_REPRESENTATION_IDX`: Sequence representation of the protein
            - Column from index `self._LABELS_START_IDX` onwards: Labels

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            None
        """
        print("Checking for processed data in", self.processed_dir_main)

        processed_name = self.processed_main_file_names_dict["data"]
        if not os.path.isfile(os.path.join(self.processed_dir_main, processed_name)):
            print(f"Missing processed data file (`{processed_name}` file)")
            os.makedirs(self.processed_dir_main, exist_ok=True)
            data_path = self._download_required_data()
            g = self._extract_class_hierarchy(data_path)
            data_df = self._graph_to_raw_dataset(g)
            self.save_processed(data_df, processed_name)

    @abstractmethod
    def _download_required_data(self) -> str:
        """
        Downloads the required raw data.

        Returns:
            str: Path to the downloaded data.
        """
        pass

    @abstractmethod
    def _extract_class_hierarchy(self, data_path: str) -> nx.DiGraph:
        """
        Extracts the class hierarchy from the data.
        Constructs a directed graph (DiGraph) using NetworkX, where nodes are annotated with fields/terms from
        the term documents.

        Args:
            data_path (str): Path to the data.

        Returns:
            nx.DiGraph: The class hierarchy graph.
        """
        pass

    @abstractmethod
    def _graph_to_raw_dataset(self, graph: nx.DiGraph) -> pd.DataFrame:
        """
        Converts the graph to a raw dataset.
        Uses the graph created by `_extract_class_hierarchy` method to extract the
        raw data in Dataframe format with additional columns corresponding to each multi-label class.

        Args:
            graph (nx.DiGraph): The class hierarchy graph.

        Returns:
            pd.DataFrame: The raw dataset.
        """
        pass

    @abstractmethod
    def select_classes(self, g: nx.DiGraph, *args, **kwargs) -> List:
        """
        Selects classes from the dataset based on a specified criteria.

        Args:
            g (nx.Graph): The graph representing the dataset.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            List: A sorted list of node IDs that meet the specified criteria.
        """
        pass

    def save_processed(self, data: pd.DataFrame, filename: str) -> None:
        """
        Save the processed dataset to a pickle file.

        Args:
            data (pd.DataFrame): The processed dataset to be saved.
            filename (str): The filename for the pickle file.
        """
        pd.to_pickle(data, open(os.path.join(self.processed_dir_main, filename), "wb"))

    # ------------------------------ Phase: Setup data -----------------------------------
    def setup_processed(self) -> None:
        """
        Transforms `data.pkl` into a model input data format (`data.pt`), ensuring that the data is in a format
        compatible for input to the model.
        The transformed data contains the following keys: `ident`, `features`, `labels`, and `group`.
        This method uses a subclass of Data Reader to perform the transformation.

        Returns:
            None
        """
        os.makedirs(self.processed_dir, exist_ok=True)
        transformed_file_name = self.processed_file_names_dict["data"]
        print(
            f"Missing transformed data (`{transformed_file_name}` file). Transforming data.... "
        )
        torch.save(
            self._load_data_from_file(
                os.path.join(
                    self.processed_dir_main,
                    self.processed_main_file_names_dict["data"],
                )
            ),
            os.path.join(self.processed_dir, transformed_file_name),
        )

    @staticmethod
    def _get_data_size(input_file_path: str) -> int:
        """
        Get the size of the data from a pickled file.

        Args:
            input_file_path (str): The path to the file.

        Returns:
            int: The size of the data.
        """
        with open(input_file_path, "rb") as f:
            return len(pd.read_pickle(f))

    @abstractmethod
    def _load_dict(self, input_file_path: str) -> Generator[Dict[str, Any], None, None]:
        """
        Loads data from given pickled file and yields individual dictionaries for each row.

        This method is used by `_load_data_from_file` to generate dictionaries that are then
        processed and converted into a list of dictionaries containing the features and labels.

        Args:
            input_file_path (str): The path to the pickled input file.

        Yields:
            Generator[Dict[str, Any], None, None]: Generator yielding dictionaries.

        """
        pass

    # ------------------------------ Phase: Dynamic Splits -----------------------------------
    @property
    def dynamic_split_dfs(self) -> Dict[str, pd.DataFrame]:
        """
        Property to retrieve dynamic train, validation, and test splits.

        This property checks if dynamic data splits (`_dynamic_df_train`, `_dynamic_df_val`, `_dynamic_df_test`)
        are already loaded. If any of them is None, it either generates them dynamically or retrieves them
        from data file with help of pre-existing split csv file (`splits_file_path`) containing splits assignments.

        Returns:
            dict: A dictionary containing the dynamic train, validation, and test DataFrames.
                Keys are 'train', 'validation', and 'test'.
        """
        if any(
            split is None
            for split in [
                self._dynamic_df_test,
                self._dynamic_df_val,
                self._dynamic_df_train,
            ]
        ):
            if self.splits_file_path is None:
                # Generate splits based on given seed, create csv file to records the splits
                self._generate_dynamic_splits()
            else:
                # If user has provided splits file path, use it to get the splits from the data
                self._retrieve_splits_from_csv()
        return {
            "train": self._dynamic_df_train,
            "validation": self._dynamic_df_val,
            "test": self._dynamic_df_test,
        }

    def _generate_dynamic_splits(self) -> None:
        """
        Generate data splits during runtime and save them in class variables.

        This method loads encoded data and generates train, validation, and test splits based on the loaded data.
        """
        print("\nGenerate dynamic splits...")
        df_train, df_val, df_test = self._get_data_splits()

        # Generate splits.csv file to store ids of each corresponding split
        split_assignment_list: List[pd.DataFrame] = [
            pd.DataFrame({"id": df_train["ident"], "split": "train"}),
            pd.DataFrame({"id": df_val["ident"], "split": "validation"}),
            pd.DataFrame({"id": df_test["ident"], "split": "test"}),
        ]

        combined_split_assignment = pd.concat(split_assignment_list, ignore_index=True)
        combined_split_assignment.to_csv(
            os.path.join(self.processed_dir_main, "splits.csv"), index=False
        )

        # Store the splits in class variables
        self._dynamic_df_train = df_train
        self._dynamic_df_val = df_val
        self._dynamic_df_test = df_test

    @abstractmethod
    def _get_data_splits(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Retrieve the train, validation, and test data splits for the dataset.

        This method returns data splits according to specific criteria implemented
        in the subclasses.

        Returns:
            tuple: A tuple containing DataFrames for train, validation, and test splits.
        """
        pass

    def get_test_split(
        self, df: pd.DataFrame, seed: Optional[int] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split the input DataFrame into training and testing sets based on multilabel stratified sampling.

        This method uses MultilabelStratifiedShuffleSplit to split the data such that the distribution of labels
        in the training and testing sets is approximately the same. The split is based on the "labels" column
        in the DataFrame.

        Args:
            df (pd.DataFrame): The input DataFrame containing the data to be split. It must contain a column
                               named "labels" with the multilabel data.
            seed (int, optional): The random seed to be used for reproducibility. Default is None.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the training set and testing set DataFrames.

        Raises:
            ValueError: If the DataFrame does not contain a column named "labels".
        """
        print("Get test data split")

        labels_list = df["labels"].tolist()

        if len(labels_list[0]) > 1:
            splitter = MultilabelStratifiedShuffleSplit(
                n_splits=1, test_size=self.test_split, random_state=seed
            )
        else:
            splitter = StratifiedShuffleSplit(
                n_splits=1, test_size=self.test_split, random_state=seed
            )

        train_indices, test_indices = next(splitter.split(labels_list, labels_list))

        df_train = df.iloc[train_indices]
        df_test = df.iloc[test_indices]
        return df_train, df_test

    def get_train_val_splits_given_test(
        self, df: pd.DataFrame, test_df: pd.DataFrame, seed: int = None
    ) -> Union[Dict[str, pd.DataFrame], Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Split the dataset into train and validation sets, given a test set.
        Use test set (e.g., loaded from another source or generated in get_test_split), to avoid overlap

        Args:
            df (pd.DataFrame): The original dataset.
            test_df (pd.DataFrame): The test dataset.
            seed (int, optional): The random seed to be used for reproducibility. Default is None.

        Returns:
            Union[Dict[str, pd.DataFrame], Tuple[pd.DataFrame, pd.DataFrame]]: A dictionary containing train and
                validation sets if self.use_inner_cross_validation is True, otherwise a tuple containing the train
                and validation DataFrames. The keys are the names of the train and validation sets, and the values
                are the corresponding DataFrames.
        """
        print(f"Split dataset into train / val with given test set")

        test_ids = test_df["ident"].tolist()
        df_trainval = df[~df["ident"].isin(test_ids)]
        labels_list_trainval = df_trainval["labels"].tolist()

        if self.use_inner_cross_validation:
            folds = {}
            kfold = MultilabelStratifiedKFold(
                n_splits=self.inner_k_folds, random_state=seed
            )
            for fold, (train_ids, val_ids) in enumerate(
                kfold.split(
                    labels_list_trainval,
                    labels_list_trainval,
                )
            ):
                df_validation = df_trainval.iloc[val_ids]
                df_train = df_trainval.iloc[train_ids]
                folds[self.raw_file_names_dict[f"fold_{fold}_train"]] = df_train
                folds[self.raw_file_names_dict[f"fold_{fold}_validation"]] = (
                    df_validation
                )

            return folds

        if len(labels_list_trainval[0]) > 1:
            splitter = MultilabelStratifiedShuffleSplit(
                n_splits=1,
                test_size=self.validation_split / (1 - self.test_split),
                random_state=seed,
            )
        else:
            splitter = StratifiedShuffleSplit(
                n_splits=1,
                test_size=self.validation_split / (1 - self.test_split),
                random_state=seed,
            )

        train_indices, validation_indices = next(
            splitter.split(labels_list_trainval, labels_list_trainval)
        )

        df_validation = df_trainval.iloc[validation_indices]
        df_train = df_trainval.iloc[train_indices]
        return df_train, df_validation

    def _retrieve_splits_from_csv(self) -> None:
        """
        Retrieve previously saved data splits from splits.csv file or from provided file path.

        This method loads the splits.csv file located at `self.splits_file_path`.
        It then loads the encoded data (`data.pt`) and filters it based on the IDs retrieved from
        splits.csv to reconstruct the train, validation, and test splits.
        """
        print(f"\nLoading splits from {self.splits_file_path}...")
        splits_df = pd.read_csv(self.splits_file_path)

        filename = self.processed_file_names_dict["data"]
        data = self.load_processed_data_from_file(
            os.path.join(self.processed_dir, filename)
        )
        df_data = pd.DataFrame(data)

        train_ids = splits_df[splits_df["split"] == "train"]["id"]
        validation_ids = splits_df[splits_df["split"] == "validation"]["id"]
        test_ids = splits_df[splits_df["split"] == "test"]["id"]

        self._dynamic_df_train = df_data[df_data["ident"].isin(train_ids)]
        self._dynamic_df_val = df_data[df_data["ident"].isin(validation_ids)]
        self._dynamic_df_test = df_data[df_data["ident"].isin(test_ids)]

    # ------------------------------ Phase: DataLoaders -----------------------------------
    def load_processed_data(
        self, kind: Optional[str] = None, filename: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Loads processed data from a specified dataset type or file.

        This method retrieves processed data based on the dataset type (`kind`) such as "train",
        "val", or "test", or directly from a provided filename. When `kind` is specified, the method
        leverages the `dynamic_split_dfs` property to dynamically generate or retrieve the corresponding
        data splits if they are not already loaded. If both `kind` and `filename` are provided, `filename`
        takes precedence.

        Args:
            kind (str, optional): The type of dataset to load ("train", "val", or "test").
                If `filename` is provided, this argument is ignored. Defaults to None.
            filename (str, optional): The name of the file to load the dataset from.
                If provided, this takes precedence over `kind`. Defaults to None.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, where each dictionary contains
            the processed data for an individual data point.

        Raises:
            ValueError: If both `kind` and `filename` are None, as one of them is required to load the dataset.
            KeyError: If the specified `kind` does not exist in the `dynamic_split_dfs` property or
                `processed_file_names_dict`, when expected.
            FileNotFoundError: If the file corresponding to the provided `filename` does not exist.
        """
        if kind is None and filename is None:
            raise ValueError(
                "Either kind or filename is required to load the correct dataset, both are None"
            )

        # If both kind and filename are given, use filename
        if kind is not None and filename is None:
            if self.use_inner_cross_validation and kind != "test":
                filename = self.processed_file_names_dict[
                    f"fold_{self.fold_index}_{kind}"
                ]
            else:
                data_df = self.dynamic_split_dfs[kind]
                return data_df.to_dict(orient="records")

        # If filename is provided
        return self.load_processed_data_from_file(filename)

    def load_processed_data_from_file(self, filename):
        return torch.load(os.path.join(filename), weights_only=False)

    # ------------------------------ Phase: Raw Properties -----------------------------------
    @property
    @abstractmethod
    def base_dir(self) -> str:
        """
        Returns the base directory path for storing data.

        Returns:
            str: The path to the base directory.
        """
        pass

    @property
    def processed_dir_main(self) -> str:
        """
        Returns the main directory path where processed data is stored.

        Returns:
            str: The path to the main processed data directory, based on the base directory and the instance's name.
        """
        return os.path.join(
            self.base_dir,
            self._name,
            "processed",
        )

    @property
    def processed_main_file_names_dict(self) -> dict:
        """
        Returns a dictionary mapping processed data file names.

        Returns:
            dict: A dictionary mapping dataset key to their respective file names.
                  For example, {"data": "data.pkl"}.
        """
        return {"data": "data.pkl"}

    @property
    def raw_file_names(self) -> List[str]:
        """
        Returns a list of raw file names.

        Returns:
            List[str]: A list of file names corresponding to the raw data.
        """
        return list(self.raw_file_names_dict.values())

    @property
    def processed_file_names_dict(self) -> dict:
        """
        Returns a dictionary for the processed and tokenized data files.

        Returns:
            dict: A dictionary mapping dataset keys to their respective file names.
                  For example, {"data": "data.pt"}.
        """
        if self.n_token_limit is not None:
            return {"data": f"data_maxlen{self.n_token_limit}.pt"}
        return {"data": "data.pt"}
