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


class SolCuration(XYBaseDataModule):
    HEADERS = [
        "logS",
    ]

    @property
    def _name(self):
        return "SolCuration"

    @property
    def label_number(self):
        return 1

    @property
    def raw_file_names(self):
        return ["solCuration.csv"]

    @property
    def processed_file_names(self):
        return ["test.pt", "train.pt", "validation.pt"]

    def download(self):
        # download and combine all the available curated datasets from xxx
        db_sol = ["aqsol", "aqua", "esol", "ochem", "phys"]
        with open(os.path.join(self.raw_dir, "solCuration.csv"), "ab") as dst:
            for i, db in enumerate(db_sol):
                with request.urlopen(
                    f"https://raw.githubusercontent.com/Mengjintao/SolCuration/master/cure/{db}_cure.csv",
                ) as src:
                    if i > 0:
                        src.readline()
                    shutil.copyfileobj(src, dst)

    def setup_processed(self):
        print("Create splits")
        print(self.train_split)
        print(os.path.join(self.raw_dir, f"solCuration.csv"))
        data = list(
            self._load_data_from_file(os.path.join(self.raw_dir, f"solCuration.csv"))
        )
        print(len(data))
        # data = self._load_data_from_file(os.path.join(self.raw_dir, f"solCuration.csv"))
        if 0 == 0:
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

    def _load_data_from_file(self, input_file_path: str) -> List[Dict]:
        """Loads data from a CSV file.

        Args:
            input_file_path (str): Path to the CSV file.

        Returns:
            List[Dict]: List of data dictionaries.
        """
        print("!!!!!!!!!!!!!!!!")
        smiles_l = []
        labels_l = []
        with open(input_file_path, "r") as input_file:
            reader = csv.DictReader(input_file)
            for row in reader:
                if not row["smiles"] in smiles_l:
                    smiles_l.append(row["smiles"])
                    labels_l.append(float(row["logS"]))
        # print(len(smiles_l), len(labels_l))
        # labels_l.append(np.floor(float(row["logS"])))
        # onehotencoding
        # label_binarizer = LabelBinarizer()
        # label_binarizer.fit(labels_l)
        # onehot_label_l = label_binarizer.transform(labels_l)

        # normalise data to be between 0 and 1
        # labels_norm = [(float(label)-min(labels_l))/(max(labels_l)-min(labels_l)) for label in labels_l]
        for i in range(0, len(smiles_l)):
            yield self.reader.to_data(
                dict(features=smiles_l[i], labels=[labels_l[i]], ident=i)
            )


class SolESOL(XYBaseDataModule):
    HEADERS = [
        "logS",
    ]

    @property
    def _name(self):
        return "SolESOL"

    @property
    def label_number(self):
        return 1

    @property
    def raw_file_names(self):
        return ["solESOL.csv"]

    @property
    def processed_file_names(self):
        return ["test.pt", "train.pt", "validation.pt"]

    def download(self):
        # download
        with open(os.path.join(self.raw_dir, "solESOL.csv"), "ab") as dst:
            with request.urlopen(
                f"https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv",
            ) as src:
                shutil.copyfileobj(src, dst)

    def setup_processed(self):
        print("Create splits")
        data = list(
            self._load_data_from_file(os.path.join(self.raw_dir, f"solESOL.csv"))
        )
        print(len(data))
        # data = self._load_data_from_file(os.path.join(self.raw_dir, f"solCuration.csv"))
        if 0 == 0:
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
                labels_l.append(float(row["measured log solubility in mols per litre"]))

        for i in range(0, len(smiles_l)):
            yield dict(features=smiles_l[i], labels=[labels_l[i]], ident=i)
            # yield self.reader.to_data(dict(features=smiles_l[i], labels=[labels_l[i]], ident=i))


class SolCurationChem(SolCuration):
    """Chemical data reader for the solubility dataset."""

    READER = dr.ChemDataReader


class SolESOLChem(SolESOL):
    """Chemical data reader for the solubility dataset."""

    READER = dr.ChemDataReader
