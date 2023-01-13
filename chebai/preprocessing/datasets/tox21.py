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
import rdkit
import zipfile
import shutil


class Tox21Base(XYBaseDataModule):
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
        return "tox21"

    @property
    def label_number(self):
        return 12

    @property
    def raw_file_names(self):
        return ["train.sdf", "validation.sdf", "validation.smiles", "test.smiles", "test_results.txt"]

    @property
    def processed_file_names(self):
        return ["test.pt", "train.pt", "validation.pt"]

    def download(self):
        self._retrieve_file("https://tripod.nih.gov/tox21/challenge/download?id=tox21_10k_data_allsdf&sec=", "train.sdf", compression="zip")
        self._retrieve_file("https://tripod.nih.gov/tox21/challenge/download?id=tox21_10k_challenge_testsdf&sec=",
                            "validation.sdf", compression="zip")
        self._retrieve_file("https://tripod.nih.gov/tox21/challenge/download?id=tox21_10k_challenge_scoresmiles&sec=",
                            "test.smiles")
        self._retrieve_file("https://tripod.nih.gov/tox21/challenge/download?id=tox21_10k_challenge_scoretxt&sec=",
                            "test_results.txt")

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
                    with zipfile.ZipFile(download_path, 'r') as zip_ref:
                        zip_ref.extractall(td.name)
                        files_in_zip = os.listdir(td.name)
                        f = files_in_zip[0]
                        assert len(files_in_zip) == 1
                        shutil.move(os.path.join(td.name, f), target_path)

    def _load_data_from_file(self, path):
        sdf = rdkit.Chem.SDMolSupplier(path)
        data = []
        for mol in sdf:
            if mol is not None:
                d = dict(
                    labels=[int(mol.GetProp(h)) if h in mol.GetPropNames() else None for h in self.HEADERS],
                    ident=[mol.GetProp(k) for k in ("DSSTox_CID", "Compound ID") if k in mol.GetPropNames() ][0],
                    features=rdkit.Chem.MolToSmiles(mol))
                data.append(self.reader.to_data(d))
        return data

    def setup_processed(self):
        for k in ("train", "validation"):
            torch.save(self._load_data_from_file(os.path.join(self.raw_dir, f"{k}.sdf")), os.path.join(self.processed_dir, f"{k}.pt"))

        with open(os.path.join(self.raw_dir, f"test.smiles")) as fin:
            headers = next(fin)
            test_smiles = dict(reversed(row.strip().split("\t")) for row in fin)
        with open(os.path.join(self.raw_dir, f"test_results.txt")) as fin:
            headers = next(fin).strip().split("\t")
            test_results = {k["Sample ID"]:[int(k[h]) if k[h] != "x" else None for h in self.HEADERS] for k in (dict(zip(headers, row.strip().split("\t"), strict=True)) for row in fin if row)}
        test_data = [self.reader.to_data(dict(features=test_smiles[k], labels=test_results[k], ident=k)) for k in test_smiles]
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


class Tox21Chem(Tox21Base):
    READER = dr.ChemDataReader


class Tox21Graph(Tox21Base):
    READER = dr.GraphReader



class Tox21ExtendedChem(MergedDataset):
    MERGED = [Tox21Chem, Hazardous, JCIExtendedTokenData]

    @property
    def limits(self):
        return [None, 5000, 5000]

    def _process_data(self, subset_id, data):
        res = dict(
            features=data["features"], labels=data["labels"], ident=data["ident"]
        )
        # Feature: non-toxic
        if subset_id == 0:
            res["labels"] = [not any(res["labels"])]
        elif subset_id == 1:
            res["labels"] = [False]
        elif subset_id == 2:
            res["labels"] = [True]
        return res

    @property
    def label_number(self):
        return 1
