from tempfile import NamedTemporaryFile, TemporaryDirectory
from urllib import request
import csv
import gzip
import os
import random
import shutil
import zipfile
from typing import Dict, Generator, List, Optional

from rdkit import Chem
from sklearn.model_selection import (
    GroupShuffleSplit,
    train_test_split,
    StratifiedShuffleSplit,
)
import numpy as np
import pysmiles
import torch
from sklearn.preprocessing import LabelBinarizer

from chebai.preprocessing import reader as dr
from chebai.preprocessing.datasets.base import MergedDataset, XYBaseDataModule
from chebai.preprocessing.datasets.chebi import JCIExtendedTokenData
from chebai.preprocessing.datasets.pubchem import Hazardous


class ClinTox(XYBaseDataModule):
    """Data module for ClinTox MoleculeNet dataset."""

    HEADERS = [
        "FDA_APPROVED",
        "CT_TOX",
    ]

    @property
    def _name(self) -> str:
        """Returns the name of the dataset."""
        return "ClinTox"

    @property
    def label_number(self) -> int:
        """Returns the number of labels."""
        return 2

    @property
    def raw_file_names(self) -> List[str]:
        """Returns a list of raw file names."""
        return ["clintox.csv"]

    # @property
    # def processed_file_names(self) -> List[str]:
    #     """Returns a list of processed file names."""
    #     return ["test.pt", "train.pt", "validation.pt"]

    @property
    def processed_file_names_dict(self) -> dict:
        return {
            "test": "test.pt", 
            "train": "train.pt", 
            "validation": "validation.pt",
        }

    def download(self) -> None:
        """Downloads and extracts the dataset."""
        with NamedTemporaryFile("rb") as gout:
            request.urlretrieve(
                "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv.gz",
                gout.name,
            )
            with gzip.open(gout.name) as gfile:
                with open(
                    os.path.join(self.raw_dir, "clintox.csv"), "wt"
                ) as fout:
                    fout.write(gfile.read().decode())

    def setup_processed(self) -> None:
        """Processes and splits the dataset."""
        print("Create splits")
        data = list(
            self._load_data_from_file(
                os.path.join(self.raw_dir, f"clintox.csv")
            )
        )
        groups = np.array([d["group"] for d in data])
        if not all(g is None for g in groups):
            split_size = int(len(set(groups)) * (1 - self.test_split - self.validation_split))
            os.makedirs(self.processed_dir, exist_ok=True)
            splitter = GroupShuffleSplit(train_size=split_size, n_splits=1)

            train_split_index, temp_split_index = next(
                splitter.split(data, groups=groups)
            )

            split_groups = groups[temp_split_index]

            splitter = GroupShuffleSplit(
                train_size=int(len(set(split_groups)) * (1 - self.test_split - self.validation_split)), n_splits=1
            )
            test_split_index, validation_split_index = next(
                splitter.split(temp_split_index, groups=split_groups)
            )
            train_split = [data[i] for i in train_split_index]
            test_split = [
                d for d in (data[temp_split_index[i]] for i in test_split_index)
            ]
            validation_split = [
                d for d in (data[temp_split_index[i]] for i in validation_split_index)
            ]
        else:
            train_split, test_split = train_test_split(data, test_size=self.test_split, shuffle=True)
            train_split, validation_split = train_test_split(train_split, test_size=self.validation_split, shuffle=True)
        for k, split in [
            ("test", test_split),
            ("train", train_split),
            ("validation", validation_split),
        ]:
            print("transform", k)
            torch.save(
                split,
                os.path.join(self.processed_dir, f"{k}.pt"),
            )

    def setup(self, **kwargs) -> None:
        """Sets up the dataset by downloading and processing if necessary."""
        if any(
            not os.path.isfile(os.path.join(self.raw_dir, f))
            for f in self.raw_file_names
        ):
            self.download()
        if any(
            not os.path.isfile(os.path.join(self.processed_dir, f))
            for f in self.processed_file_names
        ):
            self.setup_processed()

        self._after_setup()

    def _set_processed_data_props(self):
        """
        Load processed data and extract metadata.

        Sets:
            - self._num_of_labels: Number of target labels in the dataset.
            - self._feature_vector_size: Maximum feature vector length across all data points.
        """
        pt_file_path = os.path.join(
            self.processed_dir, self.processed_file_names_dict["train"]
        )
        data_pt = torch.load(pt_file_path, weights_only=False)

        self._num_of_labels = len(data_pt[0]["labels"])
        self._feature_vector_size = max(len(d["features"]) for d in data_pt)

    def _load_dict(self, input_file_path: str) -> List[Dict]:
        """Loads data from a CSV file.

        Args:
            input_file_path (str): Path to the CSV file.

        Returns:
            List[Dict]: List of data dictionaries.
        """
        i = 0
        with open(input_file_path, "r") as input_file:
            reader = csv.DictReader(input_file)
            for row in reader:
                i += 1
                smiles = row["smiles"]
                labels = [
                    bool(int(l)) if l else None for l in (row[k] for k in self.HEADERS)
                ]
                # group = int(row["group"])
                yield dict(features=smiles, labels=labels, ident=i, 
                            # group=group
                )
                # yield dict(features=smiles, labels=labels, ident=i)
                # yield self.reader.to_data(dict(features=smiles, labels=labels, ident=i))

    def _perform_data_preparation(self, *args, **kwargs) -> None:
        pass


class BBBP(XYBaseDataModule):
    """Data module for ClinTox MoleculeNet dataset."""

    HEADERS = [
        "p_np",
    ]

    @property
    def _name(self) -> str:
        """Returns the name of the dataset."""
        return "BBBP"

    @property
    def label_number(self) -> int:
        """Returns the number of labels."""
        return 1

    @property
    def raw_file_names(self) -> List[str]:
        """Returns a list of raw file names."""
        return ["bbbp.csv"]

    # @property
    # def processed_file_names(self) -> List[str]:
    #     """Returns a list of processed file names."""
    #     return ["test.pt", "train.pt", "validation.pt"]

    @property
    def processed_file_names_dict(self) -> dict:
        return {
            "test": "test.pt", 
            "train": "train.pt", 
            "validation": "validation.pt",
        }

    def download(self) -> None:
        """Downloads and extracts the dataset."""
        with open(os.path.join(self.raw_dir, "bbbp.csv"), "ab") as dst:
            with request.urlopen(
                f"https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv",
            ) as src:
                shutil.copyfileobj(src, dst)

    def setup_processed(self) -> None:
        """Processes and splits the dataset."""
        print("Create splits")
        data = list(
            self._load_data_from_file(os.path.join(self.raw_dir, f"bbbp.csv"))
        )
        groups = np.array([d["group"] for d in data])
        if not all(g is None for g in groups):
            print("Group shuffled")
            split_size = int(len(set(groups)) * (1 - self.test_split - self.validation_split))
            os.makedirs(self.processed_dir, exist_ok=True)
            splitter = GroupShuffleSplit(train_size=split_size, n_splits=1)

            train_split_index, temp_split_index = next(
                splitter.split(data, groups=groups)
            )

            split_groups = groups[temp_split_index]

            splitter = GroupShuffleSplit(
                train_size=int(len(set(split_groups)) * (1 - self.test_split - self.validation_split)), n_splits=1
            )
            test_split_index, validation_split_index = next(
                splitter.split(temp_split_index, groups=split_groups)
            )
            train_split = [data[i] for i in train_split_index]
            test_split = [
                d
                for d in (data[temp_split_index[i]] for i in test_split_index)
                # if d["original"]
            ]
            validation_split = [
                d
                for d in (data[temp_split_index[i]] for i in validation_split_index)
                # if d["original"]
            ]
        else:
            train_split, test_split = train_test_split(data, test_size=self.test_split, shuffle=True)
            train_split, validation_split = train_test_split(train_split, test_size=self.validation_split, shuffle=True)
        for k, split in [
            ("test", test_split),
            ("train", train_split),
            ("validation", validation_split),
        ]:
            print("transform", k)
            torch.save(
                split,
                os.path.join(self.processed_dir, f"{k}.pt"),
            )

    def setup(self, **kwargs) -> None:
        """Sets up the dataset by downloading and processing if necessary."""
        if any(
            not os.path.isfile(os.path.join(self.raw_dir, f))
            for f in self.raw_file_names
        ):
            self.download()
        if any(
            not os.path.isfile(os.path.join(self.processed_dir, f))
            for f in self.processed_file_names
        ):
            self.setup_processed()
        
        self._after_setup()


    def _set_processed_data_props(self):
        """
        Load processed data and extract metadata.

        Sets:
            - self._num_of_labels: Number of target labels in the dataset.
            - self._feature_vector_size: Maximum feature vector length across all data points.
        """
        pt_file_path = os.path.join(
            self.processed_dir, self.processed_file_names_dict["train"]
        )
        data_pt = torch.load(pt_file_path, weights_only=False)

        self._num_of_labels = len(data_pt[0]["labels"])
        self._feature_vector_size = max(len(d["features"]) for d in data_pt)


    def _load_dict(self, input_file_path: str) -> List[Dict]:
        """Loads data from a CSV file.

        Args:
            input_file_path (str): Path to the CSV file.

        Returns:
            List[Dict]: List of data dictionaries.
        """
        i = 0
        with open(input_file_path, "r") as input_file:
            reader = csv.DictReader(input_file)
            for row in reader:
                i += 1
                smiles = row["smiles"]
                labels = [int(row["p_np"])]
                # group = int(row["group"])
                yield dict(features=smiles, labels=labels, ident=i
                # , group=group
                )
                # yield self.reader.to_data(dict(features=smiles, labels=labels, ident=i))

    def _perform_data_preparation(self, *args, **kwargs) -> None:
        pass


class Sider(XYBaseDataModule):
    """Data module for ClinTox MoleculeNet dataset."""

    HEADERS = [
        "Hepatobiliary disorders",
        "Metabolism and nutrition disorders",
        "Product issues",
        "Eye disorders",
        "Investigations",
        "Musculoskeletal and connective tissue disorders",
        "Gastrointestinal disorders",
        "Social circumstances",
        "Immune system disorders",
        "Reproductive system and breast disorders",
        "Neoplasms benign, malignant and unspecified (incl cysts and polyps)",
        "General disorders and administration site conditions",
        "Endocrine disorders",
        "Surgical and medical procedures",
        "Vascular disorders",
        "Blood and lymphatic system disorders",
        "Skin and subcutaneous tissue disorders",
        "Congenital, familial and genetic disorders",
        "Infections and infestations",
        "Respiratory, thoracic and mediastinal disorders",
        "Psychiatric disorders",
        "Renal and urinary disorders",
        "Pregnancy, puerperium and perinatal conditions",
        "Ear and labyrinth disorders",
        "Cardiac disorders",
        "Nervous system disorders",
        "Injury, poisoning and procedural complications",
    ]

    @property
    def _name(self) -> str:
        """Returns the name of the dataset."""
        return "Sider"

    @property
    def label_number(self) -> int:
        """Returns the number of labels."""
        return 27

    @property
    def raw_file_names(self) -> List[str]:
        """Returns a list of raw file names."""
        return ["sider.csv"]

    # @property
    # def processed_file_names(self) -> List[str]:
    #     """Returns a list of processed file names."""
    #     return ["test.pt", "train.pt", "validation.pt"]

    @property
    def processed_file_names_dict(self) -> dict:
        return {
            "test": "test.pt", 
            "train": "train.pt", 
            "validation": "validation.pt",
        }

    def download(self) -> None:
        """Downloads and extracts the dataset."""
        with NamedTemporaryFile("rb") as gout:
            request.urlretrieve(
                "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/sider.csv.gz",
                gout.name,
            )
            with gzip.open(gout.name) as gfile:
                with open(os.path.join(self.raw_dir, "sider.csv"), "wt") as fout:
                    fout.write(gfile.read().decode())

    def setup_processed(self) -> None:
        """Processes and splits the dataset."""
        print("Create splits")
        data = list(
            self._load_data_from_file(os.path.join(self.raw_dir, f"sider.csv"))
        )
        groups = np.array([d["group"] for d in data])
        if not all(g is None for g in groups):
            split_size = int(len(set(groups)) * (1 - self.test_split - self.validation_split))
            os.makedirs(self.processed_dir, exist_ok=True)
            splitter = GroupShuffleSplit(train_size=split_size, n_splits=1)

            train_split_index, temp_split_index = next(
                splitter.split(data, groups=groups)
            )

            split_groups = groups[temp_split_index]

            splitter = GroupShuffleSplit(
                train_size=int(len(set(split_groups)) * (1 - self.test_split - self.validation_split)), n_splits=1
            )
            test_split_index, validation_split_index = next(
                splitter.split(temp_split_index, groups=split_groups)
            )
            train_split = [data[i] for i in train_split_index]
            test_split = [
                d
                for d in (data[temp_split_index[i]] for i in test_split_index)
                # if d["original"]
            ]
            validation_split = [
                d
                for d in (data[temp_split_index[i]] for i in validation_split_index)
                # if d["original"]
            ]
        else:
            train_split, test_split = train_test_split(data, test_size=self.test_split, shuffle=True)
            train_split, validation_split = train_test_split(train_split, test_size=self.validation_split, shuffle=True)
        for k, split in [
            ("test", test_split),
            ("train", train_split),
            ("validation", validation_split),
        ]:
            print("transform", k)
            torch.save(
                split,
                os.path.join(self.processed_dir, f"{k}.pt"),
            )

    def setup(self, **kwargs) -> None:
        """Sets up the dataset by downloading and processing if necessary."""
        if any(
            not os.path.isfile(os.path.join(self.raw_dir, f))
            for f in self.raw_file_names
        ):
            self.download()
        if any(
            not os.path.isfile(os.path.join(self.processed_dir, f))
            for f in self.processed_file_names
        ):
            self.setup_processed()
        
        self._after_setup()

    def _set_processed_data_props(self):
        """
        Load processed data and extract metadata.

        Sets:
            - self._num_of_labels: Number of target labels in the dataset.
            - self._feature_vector_size: Maximum feature vector length across all data points.
        """
        pt_file_path = os.path.join(
            self.processed_dir, self.processed_file_names_dict["train"]
        )
        data_pt = torch.load(pt_file_path, weights_only=False)

        self._num_of_labels = len(data_pt[0]["labels"])
        self._feature_vector_size = max(len(d["features"]) for d in data_pt)

    def _load_dict(self, input_file_path: str) -> List[Dict]:
        """Loads data from a CSV file.

        Args:
            input_file_path (str): Path to the CSV file.

        Returns:
            List[Dict]: List of data dictionaries.
        """
        i = 0
        with open(input_file_path, "r") as input_file:
            reader = csv.DictReader(input_file)
            for row in reader:
                i += 1
                smiles = row["smiles"]
                labels = [
                    bool(int(l)) if l else None for l in (row[k] for k in self.HEADERS)
                ]
                # group = row["group"]
                yield dict(features=smiles, labels=labels, ident=i
                # , group=group
                )
                # yield self.reader.to_data(dict(features=smiles, labels=labels, ident=i))

    def _perform_data_preparation(self, *args, **kwargs) -> None:
        pass

class Bace(XYBaseDataModule):
    """Data module for ClinTox MoleculeNet dataset."""

    HEADERS = [
        "class",
    ]

    @property
    def _name(self) -> str:
        """Returns the name of the dataset."""
        return "Bace"

    @property
    def label_number(self) -> int:
        """Returns the number of labels."""
        return 1

    @property
    def raw_file_names(self) -> List[str]:
        """Returns a list of raw file names."""
        return ["bace.csv"]

    # @property
    # def processed_file_names(self) -> List[str]:
    #     """Returns a list of processed file names."""
    #     return ["test.pt", "train.pt", "validation.pt"]

    @property
    def processed_file_names_dict(self) -> dict:
        return {
            "test": "test.pt", 
            "train": "train.pt", 
            "validation": "validation.pt",
        }

    def download(self) -> None:
        """Downloads and extracts the dataset."""
        with open(os.path.join(self.raw_dir, "bace.csv"), "ab") as dst:
            with request.urlopen(
                f"https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv",
            ) as src:
                shutil.copyfileobj(src, dst)

    def setup_processed(self) -> None:
        """Processes and splits the dataset."""
        print("Create splits")
        data = list(self._load_data_from_file(os.path.join(self.raw_dir, f"bace.csv")))
        # groups = np.array([d.get("group") for d in data])

        # if not all(g is None for g in groups):
        #     split_size = int(len(set(groups)) * (1 - self.test_split - self.validation_split))
        #     os.makedirs(self.processed_dir, exist_ok=True)
        #     splitter = GroupShuffleSplit(train_size=split_size, n_splits=1)

        #     train_split_index, temp_split_index = next(
        #         splitter.split(data, groups=groups)
        #     )

        #     split_groups = groups[temp_split_index]

        #     splitter = GroupShuffleSplit(
        #         train_size=int(len(set(split_groups)) * (1 - self.test_split - self.validation_split)), n_splits=1
        #     )
        #     test_split_index, validation_split_index = next(
        #         splitter.split(temp_split_index, groups=split_groups)
        #     )
        #     train_split = [data[i] for i in train_split_index]
        #     test_split = [
        #         d
        #         for d in (data[temp_split_index[i]] for i in test_split_index)
        #     ]
        #     validation_split = [
        #         d
        #         for d in (data[temp_split_index[i]] for i in validation_split_index)
        #     ]
        # else:
        train_split, test_split = train_test_split(data, test_size=self.test_split, shuffle=True)
        train_split, validation_split = train_test_split(train_split, test_size=self.validation_split, shuffle=True)
        for k, split in [
            ("test", test_split),
            ("train", train_split),
            ("validation", validation_split),
        ]:
            print("transform", k)
            torch.save(
                split,
                os.path.join(self.processed_dir, f"{k}.pt"),
            )

    def setup(self, **kwargs) -> None:
        """Sets up the dataset by downloading and processing if necessary."""
        if any(
            not os.path.isfile(os.path.join(self.raw_dir, f))
            for f in self.raw_file_names
        ):
            self.download()
        if any(
            not os.path.isfile(os.path.join(self.processed_dir, f))
            for f in self.processed_file_names
        ):
            self.setup_processed()

        self._after_setup()

    def _set_processed_data_props(self):
        """
        Load processed data and extract metadata.

        Sets:
            - self._num_of_labels: Number of target labels in the dataset.
            - self._feature_vector_size: Maximum feature vector length across all data points.
        """
        pt_file_path = os.path.join(
            self.processed_dir, self.processed_file_names_dict["train"]
        )
        data_pt = torch.load(pt_file_path, weights_only=False)

        self._num_of_labels = len(data_pt[0]["labels"])
        self._feature_vector_size = max(len(d["features"]) for d in data_pt)

    def _load_dict(self, input_file_path: str) -> List[Dict]:
        """Loads data from a CSV file.

        Args:
            input_file_path (str): Path to the CSV file.

        Returns:
            List[Dict]: List of data dictionaries.
        """
        i = 0
        with open(input_file_path, "r") as input_file:
            reader = csv.DictReader(input_file)
            for row in reader:
                i += 1
                smiles = row["mol"]
                labels = [int(row["Class"])]
                # group = row["group"]
                yield dict(features=smiles, labels=labels, ident=i)  # , group=group
                # yield self.reader.to_data(dict(features=smiles, labels=labels, ident=i))

    def _perform_data_preparation(self, *args, **kwargs) -> None:
        pass


class HIV(XYBaseDataModule):
    """Data module for ClinTox MoleculeNet dataset."""

    HEADERS = [
        "HIV_active",
    ]

    @property
    def _name(self) -> str:
        """Returns the name of the dataset."""
        return "HIV"

    @property
    def label_number(self) -> int:
        """Returns the number of labels."""
        return 1

    @property
    def raw_file_names(self) -> List[str]:
        """Returns a list of raw file names."""
        return ["hiv.csv"]

    # @property
    # def processed_file_names(self) -> List[str]:
    #     """Returns a list of processed file names."""
    #     return ["test.pt", "train.pt", "validation.pt"]

    @property
    def processed_file_names_dict(self) -> dict:
        return {
            "test": "test.pt", 
            "train": "train.pt", 
            "validation": "validation.pt",
        }

    def download(self) -> None:
        """Downloads and extracts the dataset."""
        with open(os.path.join(self.raw_dir, "hiv.csv"), "ab") as dst:
            with request.urlopen(
                f"https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/HIV.csv",
            ) as src:
                shutil.copyfileobj(src, dst)

    def setup_processed(self) -> None:
        """Processes and splits the dataset."""
        print("Create splits")
        data = list(
            self._load_data_from_file(os.path.join(self.raw_dir, f"hiv.csv"))
        )
        groups = np.array([d["group"] for d in data])
        if not all(g is None for g in groups):
            print("Group shuffled")
            split_size = int(len(set(groups)) * (1 - self.test_split - self.validation_split))
            os.makedirs(self.processed_dir, exist_ok=True)
            splitter = GroupShuffleSplit(train_size=split_size, n_splits=1)

            train_split_index, temp_split_index = next(
                splitter.split(data, groups=groups)
            )

            split_groups = groups[temp_split_index]

            splitter = GroupShuffleSplit(
                train_size=int(len(set(split_groups)) * (1 - self.test_split - self.validation_split)), n_splits=1
            )
            test_split_index, validation_split_index = next(
                splitter.split(temp_split_index, groups=split_groups)
            )
            train_split = [data[i] for i in train_split_index]
            test_split = [
                d for d in (data[temp_split_index[i]] for i in test_split_index)
            ]
            validation_split = [
                d for d in (data[temp_split_index[i]] for i in validation_split_index)
            ]
        else:
            train_split, test_split = train_test_split(data, test_size=self.test_split, shuffle=True)
            train_split, validation_split = train_test_split(train_split, test_size=self.validation_split, shuffle=True)
        for k, split in [
            ("test", test_split),
            ("train", train_split),
            ("validation", validation_split),
        ]:
            print("transform", k)
            torch.save(
                split,
                os.path.join(self.processed_dir, f"{k}.pt"),
            )

    def setup(self, **kwargs) -> None:
        """Sets up the dataset by downloading and processing if necessary."""
        if any(
            not os.path.isfile(os.path.join(self.raw_dir, f))
            for f in self.raw_file_names
        ):
            self.download()
        if any(
            not os.path.isfile(os.path.join(self.processed_dir, f))
            for f in self.processed_file_names
        ):
            self.setup_processed()

        self._after_setup()

    def _set_processed_data_props(self):
        """
        Load processed data and extract metadata.

        Sets:
            - self._num_of_labels: Number of target labels in the dataset.
            - self._feature_vector_size: Maximum feature vector length across all data points.
        """
        pt_file_path = os.path.join(
            self.processed_dir, self.processed_file_names_dict["train"]
        )
        data_pt = torch.load(pt_file_path, weights_only=False)

        self._num_of_labels = len(data_pt[0]["labels"])
        self._feature_vector_size = max(len(d["features"]) for d in data_pt)

    def _load_dict(self, input_file_path: str) -> List[Dict]:
        """Loads data from a CSV file.

        Args:
            input_file_path (str): Path to the CSV file.

        Returns:
            List[Dict]: List of data dictionaries.
        """
        i = 0
        with open(input_file_path, "r") as input_file:
            reader = csv.DictReader(input_file)
            for row in reader:
                if len(row) > 1:
                    i += 1
                    smiles = row["smiles"]
                    labels = [int(row["HIV_active"])]
                    # group = int(row["group"])
                    yield dict(features=smiles, labels=labels, ident=i
                    # , group=group
                    )
                # yield self.reader.to_data(dict(features=smiles, labels=labels, ident=i))

    def _perform_data_preparation(self, *args, **kwargs) -> None:
        pass


class MUV(XYBaseDataModule):
    """Data module for ClinTox MoleculeNet dataset."""

    HEADERS = [
        "MUV-466",
        "MUV-548",
        "MUV-600",
        "MUV-644",
        "MUV-652",
        "MUV-689",
        "MUV-692",
        "MUV-712",
        "MUV-713",
        "MUV-733",
        "MUV-737",
        "MUV-810",
        "MUV-832",
        "MUV-846",
        "MUV-852",
        "MUV-858",
        "MUV-859",
    ]

    @property
    def _name(self) -> str:
        """Returns the name of the dataset."""
        return "MUV"

    @property
    def label_number(self) -> int:
        """Returns the number of labels."""
        return 17

    @property
    def raw_file_names(self) -> List[str]:
        """Returns a list of raw file names."""
        return ["muv.csv"]

    # @property
    # def processed_file_names(self) -> List[str]:
    #     """Returns a list of processed file names."""
    #     return ["test.pt", "train.pt", "validation.pt"]

    @property
    def processed_file_names_dict(self) -> dict:
        return {
            "test": "test.pt", 
            "train": "train.pt", 
            "validation": "validation.pt",
        }

    def download(self) -> None:
        """Downloads and extracts the dataset."""
        with NamedTemporaryFile("rb") as gout:
            request.urlretrieve(
                "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/muv.csv.gz",
                gout.name,
            )
            with gzip.open(gout.name) as gfile:
                with open(os.path.join(self.raw_dir, "muv.csv"), "wt") as fout:
                    fout.write(gfile.read().decode())

    def setup_processed(self) -> None:
        """Processes and splits the dataset."""
        print("Create splits")
        data = list(self._load_data_from_file(os.path.join(self.raw_dir, f"muv.csv")))
        groups = np.array([d["group"] for d in data])
        if not all(g is None for g in groups):
            split_size = int(len(set(groups)) * (1 - self.test_split - self.validation_split))
            os.makedirs(self.processed_dir, exist_ok=True)
            splitter = GroupShuffleSplit(train_size=split_size, n_splits=1)

            train_split_index, temp_split_index = next(
                splitter.split(data, groups=groups)
            )

            split_groups = groups[temp_split_index]

            splitter = GroupShuffleSplit(
                train_size=int(len(set(split_groups)) * (1 - self.test_split - self.validation_split)), n_splits=1
            )
            test_split_index, validation_split_index = next(
                splitter.split(temp_split_index, groups=split_groups)
            )
            train_split = [data[i] for i in train_split_index]
            test_split = [
                d
                for d in (data[temp_split_index[i]] for i in test_split_index)
                # if d["original"]
            ]
            validation_split = [
                d
                for d in (data[temp_split_index[i]] for i in validation_split_index)
                # if d["original"]
            ]
        else:
            train_split, test_split = train_test_split(data, test_size=self.test_split, shuffle=True)
            train_split, validation_split = train_test_split(train_split, test_size=self.validation_split, shuffle=True)
        for k, split in [
            ("test", test_split),
            ("train", train_split),
            ("validation", validation_split),
        ]:
            print("transform", k)
            torch.save(
                split,
                os.path.join(self.processed_dir, f"{k}.pt"),
            )

    def setup(self, **kwargs) -> None:
        """Sets up the dataset by downloading and processing if necessary."""
        if any(
            not os.path.isfile(os.path.join(self.raw_dir, f))
            for f in self.raw_file_names
        ):
            self.download()
        if any(
            not os.path.isfile(os.path.join(self.processed_dir, f))
            for f in self.processed_file_names
        ):
            self.setup_processed()

        self._after_setup()

    def _set_processed_data_props(self):
        """
        Load processed data and extract metadata.

        Sets:
            - self._num_of_labels: Number of target labels in the dataset.
            - self._feature_vector_size: Maximum feature vector length across all data points.
        """
        pt_file_path = os.path.join(
            self.processed_dir, self.processed_file_names_dict["train"]
        )
        data_pt = torch.load(pt_file_path, weights_only=False)

        self._num_of_labels = len(data_pt[0]["labels"])
        self._feature_vector_size = max(len(d["features"]) for d in data_pt)


    def _load_dict(self, input_file_path: str) -> List[Dict]:
        """Loads data from a CSV file.

        Args:
            input_file_path (str): Path to the CSV file.

        Returns:
            List[Dict]: List of data dictionaries.
        """
        i = 0
        with open(input_file_path, "r") as input_file:
            reader = csv.DictReader(input_file)
            for row in reader:
                i += 1
                smiles = row["smiles"]
                labels = [
                    bool(int(l)) if l else None for l in (row[k] for k in self.HEADERS)
                ]
                # group = row["group"]
                yield dict(features=smiles, labels=labels, ident=i)  # , group=group)
                # yield self.reader.to_data(dict(features=smiles, labels=labels, ident=i))

    def _perform_data_preparation(self, *args, **kwargs) -> None:
        pass


class BaceChem(Bace):
    """Chemical data reader for Tox21MolNet dataset."""

    READER = dr.ChemDataReader


class SiderChem(Sider):
    """Chemical data reader for Tox21MolNet dataset."""

    READER = dr.ChemDataReader


class BBBPChem(BBBP):
    """Chemical data reader for Tox21MolNet dataset."""

    READER = dr.ChemDataReader


class ClinToxChem(ClinTox):
    """Chemical data reader for Tox21MolNet dataset."""

    READER = dr.ChemDataReader


class HIVChem(HIV):
    """Chemical data reader for Tox21MolNet dataset."""

    READER = dr.ChemDataReader


class MUVChem(MUV):
    """Chemical data reader for Tox21MolNet dataset."""

    READER = dr.ChemDataReader
