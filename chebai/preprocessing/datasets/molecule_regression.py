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
from sklearn.model_selection import GroupShuffleSplit, train_test_split
import numpy as np
import pysmiles
import torch
from sklearn.preprocessing import LabelBinarizer

from chebai.preprocessing import reader as dr
from chebai.preprocessing.datasets.base import MergedDataset, XYBaseDataModule
from chebai.preprocessing.datasets.chebi import JCIExtendedTokenData
from chebai.preprocessing.datasets.pubchem import Hazardous


class Lipo(XYBaseDataModule):
    HEADERS = [
        "exp",
    ]

    @property
    def _name(self):
        return "Lipo"

    @property
    def label_number(self):
        return 1

    @property
    def raw_file_names(self):
        return ["Lipo.csv"]

    # @property
    # def processed_file_names(self):
    #     return ["test.pt", "train.pt", "validation.pt"]

    @property
    def processed_file_names_dict(self) -> dict:
        return {
            "test": "test.pt",
            "train": "train.pt",
            "validation": "validation.pt",
        }

    def download(self):
        # download
        with open(os.path.join(self.raw_dir, "Lipo.csv"), "ab") as dst:
            with request.urlopen(
                f"https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv",
            ) as src:
                shutil.copyfileobj(src, dst)

    def setup_processed(self):
        print("Create splits")
        data = list(self._load_data_from_file(os.path.join(self.raw_dir, f"Lipo.csv")))
        print(len(data))

        train_split, test_split = train_test_split(
            data, test_size=self.test_split, shuffle=True
        )
        train_split, validation_split = train_test_split(
            train_split, test_size=self.validation_split, shuffle=True
        )

        if False:
            train_split, test_split = train_test_split(
                data, train_size=self.train_split, shuffle=True
            )
            test_split, validation_split = train_test_split(
                test_split, train_size=0.5, shuffle=True
            )
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

    def setup(self, **kwargs):
        if any(
            not os.path.isfile(os.path.join(self.raw_dir, f))
            for f in self.raw_file_names
        ):
            self.download()
        print(
            [
                not os.path.isfile(os.path.join(self.processed_dir, f))
                for f in self.processed_file_names
            ]
        )
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
        smiles_l = []
        labels_l = []
        with open(input_file_path, "r") as input_file:
            reader = csv.DictReader(input_file)
            print(reader.fieldnames)
            for row in reader:
                smiles_l.append(row["smiles"])
                labels_l.append(float(row["exp"]))

        for i in range(0, len(smiles_l)):
            yield dict(features=smiles_l[i], labels=[labels_l[i]], ident=i)
            # yield self.reader.to_data(dict(features=smiles_l[i], labels=[labels_l[i]], ident=i))

    def _perform_data_preparation(self, *args, **kwargs) -> None:
        pass


class LipoChem(Lipo):
    """Chemical data reader for the solubility dataset."""

    READER = dr.ChemDataReader


class FreeSolv(XYBaseDataModule):
    HEADERS = [
        "expt",
    ]

    @property
    def _name(self):
        return "FreeSolv"

    @property
    def label_number(self):
        return 1

    @property
    def raw_file_names(self):
        return ["FreeSolv.csv"]

    # @property
    # def processed_file_names(self):
    #     return ["test.pt", "train.pt", "validation.pt"]

    @property
    def processed_file_names_dict(self) -> dict:
        return {
            "test": "test.pt",
            "train": "train.pt",
            "validation": "validation.pt",
        }

    def download(self):
        # download
        with open(os.path.join(self.raw_dir, "FreeSolv.csv"), "ab") as dst:
            with request.urlopen(
                f"https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/SAMPL.csv",
            ) as src:
                shutil.copyfileobj(src, dst)

    def setup_processed(self):
        print("Create splits")
        data = list(
            self._load_data_from_file(os.path.join(self.raw_dir, f"FreeSolv.csv"))
        )
        print(len(data))
        train_split, test_split = train_test_split(
            data, test_size=self.test_split, shuffle=True
        )
        train_split, validation_split = train_test_split(
            train_split, test_size=self.validation_split, shuffle=True
        )

        if False:
            train_split, test_split = train_test_split(
                data, train_size=self.train_split, shuffle=True
            )
            test_split, validation_split = train_test_split(
                test_split, train_size=0.5, shuffle=True
            )
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

    def setup(self, **kwargs):
        if any(
            not os.path.isfile(os.path.join(self.raw_dir, f))
            for f in self.raw_file_names
        ):
            self.download()
        print(
            [
                not os.path.isfile(os.path.join(self.processed_dir, f))
                for f in self.processed_file_names
            ]
        )
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
        smiles_l = []
        labels_l = []
        with open(input_file_path, "r") as input_file:
            reader = csv.DictReader(input_file)
            print(reader.fieldnames)
            for row in reader:
                smiles_l.append(row["smiles"])
                labels_l.append(float(row["expt"]))

        for i in range(0, len(smiles_l)):
            yield dict(features=smiles_l[i], labels=[labels_l[i]], ident=i)
            # yield self.reader.to_data(dict(features=smiles_l[i], labels=[labels_l[i]], ident=i))

    def _perform_data_preparation(self, *args, **kwargs) -> None:
        pass


class FreeSolvChem(FreeSolv):
    """Chemical data reader for the solubility dataset."""

    READER = dr.ChemDataReader
