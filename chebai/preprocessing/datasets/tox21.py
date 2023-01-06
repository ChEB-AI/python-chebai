import random

from chebai.preprocessing.datasets.chebi import JCIBase
from chebai.preprocessing.datasets.base import XYBaseDataModule
from tempfile import NamedTemporaryFile
from urllib import request
from sklearn.model_selection import train_test_split, GroupShuffleSplit
import gzip
import os
import csv
import torch
from chebai.preprocessing import reader as dr
import pysmiles
import numpy as np

class Tox21Base(XYBaseDataModule):
    HEADERS = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']

    @property
    def _name(self):
        return "tox21"

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
            request.urlretrieve("https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz", gout.name)
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


            train_split_index, temp_split_index = next(splitter.split(
                data, groups=groups
            ))

            split_groups = groups[temp_split_index]

            splitter = GroupShuffleSplit(train_size=int(len(set(split_groups)) * self.train_split), n_splits=1)
            test_split_index, validation_split_index = next(splitter.split(
                temp_split_index, groups=split_groups
            ))
            train_split = [data[i] for i in train_split_index]
            test_split = [data[temp_split_index[i]] for i in test_split_index]
            validation_split = [data[temp_split_index[i]] for i in validation_split_index]
        else:
            train_split, test_split = train_test_split(
                data, train_size=self.train_split, shuffle=True
            )
            test_split, validation_split = train_test_split(
                test_split, train_size=self.train_split, shuffle=True
            )
        for k, split in [("test", test_split), ("train", train_split), ("validation", validation_split)]:
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
                labels = [bool(int(l)) if l else None for l in (row[k] for k in self.HEADERS)]
                yield dict(features=smiles, labels=labels)

class Tox21Chem(Tox21Base):
    READER = dr.ChemDataReader

class Tox21Graph(Tox21Base):
    READER = dr.GraphReader


class Tox21Bloat(Tox21Base):

    @property
    def _name(self):
        return "tox21bloat"

    def _load_dict(self, input_file_path):
        with open(input_file_path, "r") as input_file:
            reader = csv.DictReader(input_file)
            for row in reader:
                smiles = row["smiles"]
                labels = [bool(int(l)) if l else None for l in (row[k] for k in self.HEADERS)]
                yield dict(features=smiles, labels=labels, group=row["mol_id"])
                try:
                    mol = pysmiles.read_smiles(smiles)
                except:
                    pass
                else:
                    for _ in range(5):
                        n = random.randint(0, len(mol.nodes)-1)
                        try:
                            alt_smiles = pysmiles.write_smiles(mol, start=n)
                        except:
                            pass
                        else:
                            yield dict(features=alt_smiles, labels=labels, group=row["mol_id"])


class Tox21BloatChem(Tox21Bloat):
    READER = dr.ChemDataReader
