import random

from chebai.preprocessing.datasets.chebi import JCIExtendedTokenData
from chebai.preprocessing.datasets.pubchem import Hazardous
from chebai.preprocessing.datasets.base import XYBaseDataModule, MergedDataset
from tempfile import NamedTemporaryFile, TemporaryDirectory
from urllib import request
from sklearn.model_selection import train_test_split, GroupShuffleSplit
import gzip
import os
import csv
import torch
from chebai.preprocessing import reader as dr
import pysmiles
import numpy as np
from rdkit import Chem
import zipfile
import shutil


class Tox21MolNet(XYBaseDataModule):
    HEADERS = [
        "NR-AR",
        "NR-AR-LBD",
        "NR-AhR",
        "NR-Aromatase",
        "NR-ER",
        "NR-ER-LBD",
        "NR-PPAR-gamma",
        "SR-ARE",
        "SR-ATAD5",
        "SR-HSE",
        "SR-MMP",
        "SR-p53",
    ]

    @property
    def _name(self):
        return "Tox21MN"

    @property
    def label_number(self):
        return 12

    @property
    def raw_file_names(self):
        return ["tox21.csv"]

    @property
    def processed_file_names(self):
        return ["test.pt", "train.pt", "validation.pt"]

    def download(self):
        with NamedTemporaryFile("rb") as gout:
            request.urlretrieve(
                "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz",
                gout.name,
            )
            with gzip.open(gout.name) as gfile:
                with open(os.path.join(self.raw_dir, "tox21.csv"), "wt") as fout:
                    fout.write(gfile.read().decode())

    def setup_processed(self):
        print("Create splits")
        data = self._load_data_from_file(os.path.join(self.raw_dir, f"tox21.csv"))
        groups = np.array([d["group"] for d in data])
        if not all(g is None for g in groups):
            split_size = int(len(set(groups)) * self.train_split)
            os.makedirs(self.processed_dir, exist_ok=True)
            splitter = GroupShuffleSplit(train_size=split_size, n_splits=1)

            train_split_index, temp_split_index = next(
                splitter.split(data, groups=groups)
            )

            split_groups = groups[temp_split_index]

            splitter = GroupShuffleSplit(
                train_size=int(len(set(split_groups)) * self.train_split), n_splits=1
            )
            test_split_index, validation_split_index = next(
                splitter.split(temp_split_index, groups=split_groups)
            )
            train_split = [data[i] for i in train_split_index]
            test_split = [
                d
                for d in (data[temp_split_index[i]] for i in test_split_index)
                if d["original"]
            ]
            validation_split = [
                d
                for d in (data[temp_split_index[i]] for i in validation_split_index)
                if d["original"]
            ]
        else:
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
        if any(
            not os.path.isfile(os.path.join(self.processed_dir, f))
            for f in self.processed_file_names
        ):
            self.setup_processed()

    def _load_dict(self, input_file_path):
        with open(input_file_path, "r") as input_file:
            reader = csv.DictReader(input_file)
            for row in reader:
                smiles = row["smiles"]
                labels = [
                    bool(int(l)) if l else None for l in (row[k] for k in self.HEADERS)
                ]
                yield dict(features=smiles, labels=labels, ident=row["mol_id"])


class Tox21Challenge(XYBaseDataModule):
    HEADERS = [
        "NR-AR",
        "NR-AR-LBD",
        "NR-AhR",
        "NR-Aromatase",
        "NR-ER",
        "NR-ER-LBD",
        "NR-PPAR-gamma",
        "SR-ARE",
        "SR-ATAD5",
        "SR-HSE",
        "SR-MMP",
        "SR-p53",
    ]

    @property
    def _name(self):
        return "Tox21Chal"

    @property
    def label_number(self):
        return 12

    @property
    def raw_file_names(self):
        return [
            "train.sdf",
            "validation.sdf",
            "validation.smiles",
            "test.smiles",
            "test_results.txt",
        ]

    @property
    def processed_file_names(self):
        return ["test.pt", "train.pt", "validation.pt"]

    def download(self):
        self._retrieve_file(
            "https://tripod.nih.gov/tox21/challenge/download?id=tox21_10k_data_allsdf&sec=",
            "train.sdf",
            compression="zip",
        )
        self._retrieve_file(
            "https://tripod.nih.gov/tox21/challenge/download?id=tox21_10k_challenge_testsdf&sec=",
            "validation.sdf",
            compression="zip",
        )
        self._retrieve_file(
            "https://tripod.nih.gov/tox21/challenge/download?id=tox21_10k_challenge_scoresmiles&sec=",
            "test.smiles",
        )
        self._retrieve_file(
            "https://tripod.nih.gov/tox21/challenge/download?id=tox21_10k_challenge_scoretxt&sec=",
            "test_results.txt",
        )

    def _retrieve_file(self, url, target_file, compression=None):
        target_path = os.path.join(self.raw_dir, target_file)
        if not os.path.isfile(target_path):
            with NamedTemporaryFile("rb") as gout:
                if compression is None:
                    download_path = target_path
                else:
                    download_path = gout.name
                request.urlretrieve(
                    url,
                    download_path,
                )
                if compression == "zip":
                    td = TemporaryDirectory()
                    with zipfile.ZipFile(download_path, "r") as zip_ref:
                        zip_ref.extractall(td.name)
                        files_in_zip = os.listdir(td.name)
                        f = files_in_zip[0]
                        assert len(files_in_zip) == 1
                        shutil.move(os.path.join(td.name, f), target_path)

    def _load_data_from_file(self, path):
        sdf = Chem.SDMolSupplier(path)
        data = []
        for mol in sdf:
            if mol is not None:
                d = dict(
                    labels=[
                        int(mol.GetProp(h)) if h in mol.GetPropNames() else None
                        for h in self.HEADERS
                    ],
                    ident=[
                        mol.GetProp(k)
                        for k in ("DSSTox_CID", "Compound ID")
                        if k in mol.GetPropNames()
                    ][0],
                    features=Chem.MolToSmiles(mol),
                )
                data.append(self.reader.to_data(d))
        return data

    def setup_processed(self):
        for k in ("train", "validation"):
            d = self._load_data_from_file(os.path.join(self.raw_dir, f"{k}.sdf"))
            torch.save(d, os.path.join(self.processed_dir, f"{k}.pt"))

        with open(os.path.join(self.raw_dir, f"test.smiles")) as fin:
            next(fin)
            test_smiles = dict(reversed(row.strip().split("\t")) for row in fin)
        with open(os.path.join(self.raw_dir, f"test_results.txt")) as fin:
            headers = next(fin).strip().split("\t")
            test_results = {
                k["Sample ID"]: [
                    int(k[h]) if k[h] != "x" else None for h in self.HEADERS
                ]
                for k in (
                    dict(zip(headers, row.strip().split("\t"))) for row in fin if row
                )
            }
        test_data = [
            self.reader.to_data(
                dict(features=test_smiles[k], labels=test_results[k], ident=k)
            )
            for k in test_smiles
        ]
        torch.save(test_data, os.path.join(self.processed_dir, f"test.pt"))

    def setup(self, **kwargs):
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

    def _load_dict(self, input_file_path):
        with open(input_file_path, "r") as input_file:
            reader = csv.DictReader(input_file)
            for row in reader:
                smiles = row["smiles"]
                labels = [
                    bool(int(l)) if l else None for l in (row[k] for k in self.HEADERS)
                ]
                yield dict(features=smiles, labels=labels, ident=row["mol_id"])


class Tox21ChallengeChem(Tox21Challenge):
    READER = dr.ChemDataReader


class Tox21MolNetChem(Tox21MolNet):
    READER = dr.ChemDataReader
