import gzip
import os
import random
import shutil
import tempfile
from datetime import datetime
from typing import Any, Dict, Generator, List, Optional, Tuple, Type, Union

import pandas as pd
import requests
import torch
import tqdm

from chebai.preprocessing import reader as dr
from chebai.preprocessing.datasets.base import (
    DataLoader,
    XYBaseDataModule,
    _DynamicDataset,
)
from chebai.preprocessing.datasets.chebi import (
    ChEBIOver50,
    ChEBIOver100,
    ChEBIOverX,
)


class PubChem(_DynamicDataset):
    """
    Dataset module for PubChem compounds.
    """

    SMILES_INDEX = 0
    LABEL_INDEX = 1
    FULL = 0
    UNLABELED = True
    READER = dr.StaticSMILESReader

    # Column indices in data.pkl
    _ID_IDX: int = 0
    _DATA_REPRESENTATION_IDX: int = 1
    _LABELS_START_IDX: int = 2

    def __init__(self, *args, n_samples: Optional[int] = 100000, **kwargs):
        """
        Args:
            n_samples (Optional[int]): Number of samples to use. Set to `PubChem.FULL` for full dataset.
            *args: Additional arguments for superclass initialization.
            **kwargs: Additional keyword arguments for superclass initialization.
        """
        self._n_samples = n_samples
        current_year = datetime.today().year
        current_month = datetime.today().month
        self.pubchem_url = f"https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Monthly/{current_year}-{current_month:02d}-01/Extras/CID-SMILES.gz"

        super(PubChem, self).__init__(*args, **kwargs)

    @property
    def _name(self) -> str:
        """
        Returns:
            str: Name of the dataset.
        """
        return "Pubchem"

    @property
    def base_dir(self) -> str:
        """
        Returns:
            str: Base directory for this dataset.
        """
        if self._base_dir is not None:
            return self._base_dir
        return os.path.join("data", self._name)

    @property
    def identifier(self) -> tuple:
        """
        Returns:
            tuple: Tuple containing only the reader name (split is encoded in processed_dir_main).
        """
        return (self.reader.name(),)

    @property
    def split_label(self) -> str:
        """
        Returns:
            str: Label indicating the split of the dataset ('full' or a specific number).
        """
        if self._n_samples and self._n_samples != self.FULL:
            return str(self._n_samples)
        else:
            return "full"

    @property
    def processed_dir_main(self) -> str:
        """
        Returns:
            str: Directory where data.pkl and splits.csv are stored (split-specific).
        """
        return os.path.join(self.base_dir, "processed", self.split_label)

    @property
    def raw_dir(self) -> str:
        """
        Returns:
            str: Directory path where raw data is stored.
        """
        return os.path.join(self.base_dir, "raw", self.split_label)

    @property
    def _raw_data_source_path(self) -> str:
        """Path to the raw text file used to build data.pkl."""
        return os.path.join(self.raw_dir, "smiles.txt")

    @staticmethod
    def _parse_raw_line(line: str) -> dict:
        """Parse a single tab-separated line from a raw smiles text file."""
        ident, smiles = line.split("\t")
        return dict(id=ident.strip(), smiles=smiles.strip())

    def _load_dict(self, input_file_path: str) -> Generator[dict, None, None]:
        """
        Load data from data.pkl and yield dicts with features, labels and ident.

        Args:
            input_file_path (str): Path to the data.pkl file.

        Yields:
            dict: Dictionary containing 'features', 'labels' (None), and 'ident' fields.
        """
        with open(input_file_path, "rb") as f:
            df = pd.read_pickle(f)
        for _, row in df.iterrows():
            yield dict(features=row["smiles"], labels=None, ident=str(row["id"]))

    def _download_required_data(self) -> str:
        """Download raw data and return the path to the source file."""
        self.download()
        return self._raw_data_source_path

    def _graph_to_raw_dataset(self, graph):
        raise NotImplementedError(
            "PubChem does not use a graph-based data preparation pipeline."
        )

    def download(self):
        """
        Downloads PubChem data based on `_k` parameter.
        """
        if not os.path.isfile(os.path.join(self.raw_dir, "smiles.txt")):
            if self._n_samples == PubChem.FULL:
                print("Download from", self.pubchem_url)
                r = requests.get(self.pubchem_url, allow_redirects=True)
                with tempfile.NamedTemporaryFile() as tf:
                    tf.write(r.content)
                    print("Unpacking...")
                    tf.seek(0)
                    with gzip.open(tf, "rb") as f_in:
                        with open(
                            os.path.join(self.raw_dir, "smiles.txt"), "wb"
                        ) as f_out:
                            shutil.copyfileobj(f_in, f_out)
            else:
                full_dataset = self.__class__(n_samples=PubChem.FULL)
                full_dataset.download()
                with open(
                    os.path.join(full_dataset.raw_dir, "smiles.txt"), "r"
                ) as f_in:
                    lines = sum(1 for _ in f_in)
                    selected = frozenset(
                        random.sample(list(range(lines)), k=self._n_samples)
                    )
                    f_in.seek(0)
                    selected_lines = list(
                        filter(
                            lambda x: x[0] in selected,
                            enumerate(tqdm.tqdm(f_in, total=lines)),
                        )
                    )
                with open(os.path.join(self.raw_dir, "smiles.txt"), "w") as f_out:
                    f_out.writelines([line for _, line in selected_lines])

    @property
    def raw_file_names(self) -> List[str]:
        """
        Returns:
            List[str]: List of raw data file names.
        """
        return ["smiles.txt"]

    def _set_processed_data_props(self):
        """
        Self-supervised learning with PubChem does not use this metadata, therefore set them to zero.

        Sets:
            - self._num_of_labels: 0
            - self._feature_vector_size: 0.
        """

        self._num_of_labels = 0
        self._feature_vector_size = 0

        print(
            f"Number of labels and feature vector size set to: {self._num_of_labels} / {self._feature_vector_size} (default values, not used for self-supervised learning)"
        )

    def _perform_data_preparation(self, *args, **kwargs):
        """
        Checks for raw data, downloads if necessary, then builds data.pkl.
        """
        print(
            f"Check for raw data ({', '.join(self.raw_file_names)}) in {self.raw_dir}..."
        )
        if any(
            not os.path.isfile(os.path.join(self.raw_dir, f))
            for f in self.raw_file_names
        ):
            print("Downloading data. This may take some time...")
            self.download()
            print("Done")

        pkl_path = os.path.join(
            self.processed_dir_main, self.processed_main_file_names_dict["data"]
        )
        if not os.path.isfile(pkl_path):
            os.makedirs(self.processed_dir_main, exist_ok=True)
            print(f"Building data.pkl from {self._raw_data_source_path}...")
            rows = []
            with open(self._raw_data_source_path, "r") as f:
                for line in tqdm.tqdm(f):
                    line = line.rstrip("\n")
                    if line:
                        rows.append(self._parse_raw_line(line))
            df = pd.DataFrame(rows, columns=["id", "smiles"])
            pd.to_pickle(df, pkl_path)
            print(f"Saved {len(df)} entries to {pkl_path}")

    def _get_data_splits(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load encoded data and split into train, validation and test.
        """
        from sklearn.model_selection import train_test_split

        filename = self.processed_file_names_dict["data"]
        data = self.load_processed_data_from_file(filename)
        df = pd.DataFrame(data)

        train, rest = train_test_split(
            df,
            train_size=1 - (self.validation_split + self.test_split),
            random_state=self.dynamic_data_split_seed,
        )
        val, test = train_test_split(
            rest,
            train_size=self.validation_split
            / (self.validation_split + self.test_split),
            random_state=self.dynamic_data_split_seed,
        )
        return train, val, test


class PubChemBatched(PubChem):
    """Store train data as batches of 10m, validation and test should each be 100k max"""

    READER: Type[dr.DataReader] = dr.StaticSMILESReader

    def __init__(self, train_batch_size=1_000_000, *args, **kwargs):
        super(PubChemBatched, self).__init__(*args, **kwargs)
        self.curr_epoch = 0
        self.train_batch_size = train_batch_size
        if self._n_samples != self.FULL:
            self.val_batch_size = (
                100_000
                if self.validation_split * self._n_samples > 100_000
                else int(self.validation_split * self._n_samples)
            )
            self.test_batch_size = (
                100_000
                if self.test_split * self._n_samples > 100_000
                else int(self.test_split * self._n_samples)
            )
        else:
            self.val_batch_size = 100_000
            self.test_batch_size = 100_000

    @property
    def processed_file_names_dict(self) -> Dict[str, str]:
        """
        Returns:
            Dict[str, str]: Dictionary of processed data file names.
        """
        train_samples = (
            self._n_samples
            if self._n_samples != self.FULL
            else 120_000_000  # estimated PubChem size
        )  # estimate size
        train_samples -= self.val_batch_size + self.test_batch_size
        train_batches = (
            {"train": "train.pt"}
            if train_samples <= self.train_batch_size
            else {
                f"train_{i}": f"train_{i}.pt"
                for i in range(train_samples // self.train_batch_size)
            }
        )
        train_batches["test"] = "test.pt"
        train_batches["validation"] = "validation.pt"
        return train_batches

    def _tokenize_batched(self, data):
        """
        Load data from a file and return a list of dictionaries, batched in 1,000,000 entries.

        Args:
            path (str): The path to the input file.
            batch_size (int): The size of each batch.
            batch_idx (int): The index of the batch to load.

        Returns:
            List: A list of dictionaries containing the features and labels.
        """
        print(f"Processing {len(data)} lines...")
        batch = []
        for i, d in enumerate(tqdm.tqdm(data, total=len(data))):
            if d["features"] is not None:
                batch.append(self.reader.to_data(d))
            if i % self.train_batch_size == 0 and i > 0:
                print(f"Generating batch {i // self.train_batch_size - 1}")
                batch = [b for b in batch if b["features"] is not None]
                if self.n_token_limit is not None:
                    batch = [
                        b for b in batch if len(b["features"]) <= self.n_token_limit
                    ]
                yield batch
                batch = []
        print("Generating final batch")
        batch = [b for b in batch if b["features"] is not None]
        if self.n_token_limit is not None:
            batch = [b for b in batch if len(b["features"]) <= self.n_token_limit]
        yield batch

    def setup_processed(self):
        """
        Prepares processed data and saves them as Torch tensors.
        """
        from sklearn.model_selection import train_test_split

        pkl_path = os.path.join(
            self.processed_dir_main, self.processed_main_file_names_dict["data"]
        )
        print("Load data from file", pkl_path)
        data_not_tokenized = list(self._load_dict(pkl_path))
        print("Create splits")
        train, test = train_test_split(
            data_not_tokenized, test_size=self.test_batch_size + self.val_batch_size
        )
        del data_not_tokenized
        test, val = train_test_split(test, train_size=self.test_batch_size)
        # Save first (and only) test batch
        torch.save(
            next(self._tokenize_batched(test)),
            os.path.join(self.processed_dir, self.processed_file_names_dict["test"]),
        )
        # save first (and only) validation batch
        torch.save(
            next(self._tokenize_batched(val)),
            os.path.join(
                self.processed_dir, self.processed_file_names_dict["validation"]
            ),
        )

        # batch training if necessary
        if len(train) > self.train_batch_size:
            for i, batch in enumerate(self._tokenize_batched(train)):
                torch.save(batch, os.path.join(self.processed_dir, f"train_{i}.pt"))
        else:
            torch.save(
                next(self._tokenize_batched(train)),
                os.path.join(self.processed_dir, "train.pt"),
            )

        self.reader.on_finish()

    def train_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        """
        Returns the train DataLoader. This swaps the training batch for each epoch.

        Args:
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            DataLoader: A DataLoader object for training data.
        """
        return self.dataloader(
            (
                "train"
                if "train" in self.processed_file_names_dict
                else f"train_{self.curr_epoch}"
            ),
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
            **kwargs,
        )

    def load_processed_data(
        self, kind: Optional[str] = None, filename: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Loads processed data from a specified dataset type or file. Loads data directly from file instead of
        using the dynamic_splits_df property. This ensures that a new training batch is loaded for each epoch.
        """
        if kind is None and filename is None:
            raise ValueError(
                "Either kind or filename is required to load the correct dataset, both are None"
            )

        # If both kind and filename are given, use filename
        if kind is not None and filename is None:
            return self.load_processed_data_from_file(
                self.processed_file_names_dict[kind]
            )

        # If filename is provided
        return self.load_processed_data_from_file(filename)


class LabeledUnlabeledMixed(XYBaseDataModule):
    """
    Mixed dataset combining labeled and unlabeled data.

    Inherits from XYBaseDataModule.

    Attributes:
        READER (type): Data reader type for chemical data (ChemDataReader).
    """

    READER: Type[dr.ChemDataReader] = dr.ChemDataReader

    def __init__(
        self, labeled: XYBaseDataModule, unlabeled: XYBaseDataModule, *args, **kwargs
    ):
        """
        Args:
            labeled (XYBaseDataModule): Labeled dataset module.
            unlabeled (XYBaseDataModule): Unlabeled dataset module.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        self.labeled = labeled
        self.unlabeled = unlabeled
        super().__init__(*args, **kwargs)

    @property
    def _name(self) -> str:
        """
        Returns the name of the dataset.
        """
        return f"Mixed_{self.labeled._name}_{self.unlabeled._name}"

    def dataloader(self, kind: str, **kwargs) -> DataLoader:
        """
        Returns a dataloader for the specified kind.

        Args:
            kind (str): Type of data (e.g., 'train', 'validation', 'test').
            **kwargs: Additional keyword arguments for DataLoader.

        Returns:
            DataLoader: DataLoader instance.
        """
        labeled_data = self.labeled.load_processed_data(kind)
        unlabeled_data = self.unlabeled.load_processed_data(kind)
        if self.data_limit is not None:
            labeled_data = labeled_data[: self.data_limit]
            unlabeled_data = unlabeled_data[: self.data_limit]
        return DataLoader(
            labeled_data + unlabeled_data,
            collate_fn=self.reader.collator,
            batch_size=self.batch_size,
            **kwargs,
        )

    @property
    def raw_file_names(self) -> List[str]:
        """
        Returns the list of raw file names (empty for mixed dataset).
        """
        return []

    @property
    def processed_file_names(self) -> List[str]:
        """
        Returns the list of processed file names.
        """
        return ["test.pt", "train.pt", "validation.pt"]

    def setup_processed(self):
        """
        Sets up the processed data by setting up labeled and unlabeled datasets.
        """
        self.labeled.setup()
        self.unlabeled.setup()


class PubToxAndChebiX(LabeledUnlabeledMixed):
    """
    Mixed dataset combining PubChem and ChEBI datasets.

    Inherits from LabeledUnlabeledMixed.

    Attributes:
        READER (type): Data reader type for chemical data (ChemDataReader).
        CHEBI_X (type): Specific ChEBI dataset type.
    """

    READER: Type[dr.ChemDataReader] = dr.ChemDataReader
    CHEBI_X: Type[ChEBIOverX] = ChEBIOverX

    def __init__(self, *args, **kwargs):
        """
        Args:
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            self.CHEBI_X(*args, **kwargs), PubChem(*args, **kwargs), *args, **kwargs
        )

    @property
    def _name(self) -> str:
        """
        Returns the name of the dataset.
        """
        return "PubToxU" + self.labeled._name


class PubToxAndChebi100(PubToxAndChebiX):
    """
    Mixed dataset combining PubChem and ChEBI datasets with over 100 entries.

    Inherits from PubToxAndChebiX.

    Attributes:
        CHEBI_X (type): ChEBI dataset with over 100 entries.
    """

    CHEBI_X: Type[ChEBIOver100] = ChEBIOver100


class PubToxAndChebi50(PubToxAndChebiX):
    """
    Mixed dataset combining PubChem and ChEBI datasets with over 50 entries.

    Inherits from PubToxAndChebiX.

    Attributes:
        CHEBI_X (type): ChEBI dataset with over 50 entries.
    """

    CHEBI_X: Type[ChEBIOver50] = ChEBIOver50


class PubChemDeepSMILES(PubChem):
    """
    Subset of PubChem using DeepChemDataReader for data reading.

    Inherits from PubChem.

    Attributes:
        READER (type): Data reader type for chemical data (DeepChemDataReader).
    """

    READER: Type[dr.DeepChemDataReader] = dr.DeepChemDataReader


class PubChemSELFIES(PubChem):
    """
    Subset of PubChem using SelfiesReader for data reading.

    Inherits from PubChem.

    Attributes:
        READER (type): Data reader type for chemical data (SelfiesReader).
    """

    READER: Type[dr.SelfiesReader] = dr.SelfiesReader


if __name__ == "__main__":
    dataset = PubChem(n_samples=10_000)
    dataset.prepare_data()
    dataset.setup()
