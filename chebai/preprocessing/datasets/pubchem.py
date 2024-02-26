__all__ = [
    "PubchemBPE",
    "PubChemTokens",
    "SWJSelfies",
    "SWJPreChem",
    "SWJBPE",
    "SWJChem",
]

import gzip
import os
import random
import shutil
import tempfile

from sklearn.model_selection import train_test_split
import requests
import torch
import tqdm

from chebai.preprocessing import reader as dr
from chebai.preprocessing.datasets.base import DataLoader, XYBaseDataModule
from chebai.preprocessing.datasets.chebi import (
    ChEBIOverX,
)


class PubChem(XYBaseDataModule):
    SMILES_INDEX = 0
    LABEL_INDEX = 1
    FULL = 0
    UNLABELED = True

    def __init__(self, *args, k=100000, **kwargs):
        self._k = k
        self.pubchem_url = "https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Monthly/2023-11-01/Extras/CID-SMILES.gz"

        super(PubChem, self).__init__(*args, **kwargs)

    @property
    def _name(self):
        return f"Pubchem"

    @property
    def identifier(self):
        return self.reader.name(), self.split_label

    @property
    def split_label(self):
        if self._k:
            return str(self._k)
        else:
            return "full"

    @property
    def raw_dir(self):
        return os.path.join(self.base_dir, "raw", self.split_label)

    @staticmethod
    def _load_dict(input_file_path):
        with open(input_file_path, "r") as input_file:
            for row in input_file:
                ident, smiles = row.split("\t")
                yield dict(features=smiles, labels=None, ident=ident)

    def download(self):
        if self._k == PubChem.FULL:
            if not os.path.isfile(os.path.join(self.raw_dir, "smiles.txt")):
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
            full_dataset = self.__class__(k=PubChem.FULL)
            full_dataset.download()
            with open(os.path.join(full_dataset.raw_dir, "smiles.txt"), "r") as f_in:
                lines = sum(1 for _ in f_in)
                selected = frozenset(random.sample(list(range(lines)), k=self._k))
                f_in.seek(0)
                selected_lines = list(
                    filter(
                        lambda x: x[0] in selected,
                        enumerate(tqdm.tqdm(f_in, total=lines)),
                    )
                )
            with open(os.path.join(self.raw_dir, "smiles.txt"), "w") as f_out:
                f_out.writelines([l for i, l in selected_lines])

    def setup_processed(self):
        # Collect token distribution
        filename = os.path.join(self.raw_dir, self.raw_file_names[0])
        print("Load data from file", filename)
        data = self._load_data_from_file(filename)
        print("Create splits")
        train, test = train_test_split(data, train_size=self.train_split)
        del data
        test, val = train_test_split(test, train_size=self.train_split)
        torch.save(train, os.path.join(self.processed_dir, f"train.pt"))
        torch.save(test, os.path.join(self.processed_dir, f"test.pt"))
        torch.save(val, os.path.join(self.processed_dir, f"validation.pt"))

    @property
    def raw_file_names(self):
        return ["smiles.txt"]

    @property
    def processed_file_names(self):
        return ["test.pt", "train.pt", "validation.pt"]

    def prepare_data(self, *args, **kwargs):
        print("Check for raw data in", self.raw_dir)
        if any(
            not os.path.isfile(os.path.join(self.raw_dir, f))
            for f in self.raw_file_names
        ):
            print("Downloading data. This may take some time...")
            self.download()
            print("Done")


class SWJPreChem(PubChem):
    UNLABELED = True

    @property
    def _name(self):
        return f"SWJpre"

    def download(self):
        raise Exception("Required raw files not found")

    @property
    def identifier(self):
        return (self.reader.name(),)

    @property
    def raw_dir(self):
        return os.path.join("data", self._name, "raw")


class SWJSelfies(SWJPreChem):
    READER = dr.SelfiesReader


class PubchemChem(PubChem):
    READER = dr.ChemDataReader

    @property
    def label_number(self):
        return -1


class PubchemBPE(PubChem):
    READER = dr.ChemBPEReader

    @property
    def label_number(self):
        return -1


class SWJChem(SWJPreChem):
    READER = dr.ChemDataUnlabeledReader

    @property
    def label_number(self):
        return -1


class SWJBPE(SWJPreChem):
    READER = dr.ChemBPEReader

    @property
    def label_number(self):
        return -1


class PubChemTokens(PubChem):
    READER = dr.ChemDataReader


class Hazardous(SWJChem):
    READER = dr.ChemDataUnlabeledReader

    @property
    def _name(self):
        return f"hazardous"

    @staticmethod
    def _load_dict(input_file_path):
        with open(input_file_path, "r") as input_file:
            for row in input_file:
                smiles = row.strip()
                yield dict(features=smiles, labels=None)

    def download(self):
        raise Exception(
            "This dataset is not publicly available, yet. Please supply raw data manually."
        )


class SWJPreChem(PubChem):
    UNLABELED = True

    @property
    def _name(self):
        return f"SWJpre"

    def download(self):
        raise Exception("Required raw files not found")

    @property
    def identifier(self):
        return (self.reader.name(),)


class PubToxAndChebiX(XYBaseDataModule):
    READER = dr.ChemDataReader

    def __init__(self, chebi_x: ChEBIOverX, *args, **kwargs):
        self.labeled = chebi_x
        self.unlabeled = PubchemChem(*args, **kwargs)
        super().__init__(*args, **kwargs)

    @property
    def _name(self):
        return "PubToxU" + self.labeled._name

    def dataloader(self, kind, **kwargs):
        labeled_data = torch.load(
            os.path.join(self.labeled.processed_dir, f"{kind}.pt")
        )
        unlabeled_data = torch.load(
            os.path.join(self.unlabeled.processed_dir, f"{kind}.pt")
        )
        if self.data_limit is not None:
            labeled_data = labeled_data[: self.data_limit]
            unlabeled_data = unlabeled_data[: self.data_limit]
        return DataLoader(
            labeled_data + unlabeled_data,
            collate_fn=self.reader.collater,
            batch_size=self.batch_size,
            **kwargs,
        )

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ["test.pt", "train.pt", "validation.pt"]

    def setup_processed(self):
        self.labeled.setup()
        self.unlabeled.setup()


class PubChemDeepSMILES(PubChem):
    READER = dr.DeepChemDataReader


class PubChemSELFIES(PubChem):
    READER = dr.SelfiesReader
