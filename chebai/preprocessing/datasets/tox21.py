from chebai.preprocessing.datasets.chebi import JCIBase
from chebai.preprocessing.datasets.base import XYBaseDataModule
from tempfile import NamedTemporaryFile
from urllib import request
from sklearn.model_selection import train_test_split
import gzip
import os
import csv
import torch
from chebai.preprocessing import reader as dr


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
        os.makedirs(self.processed_dir, exist_ok=True)
        train_split, test_split = train_test_split(
            data, train_size=self.train_split, shuffle=True
        )
        test_split, validation_split = train_test_split(
            test_split, train_size=self.train_split, shuffle=True
        )
        for k, s in [("test", test_split), ("train", train_split), ("validation", validation_split)]:
            print("transform", k)
            torch.save(
                s,
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

    def _load_tuples(self, input_file_path):
        with open(input_file_path, "r") as input_file:
            reader = csv.DictReader(input_file)
            for row in reader:
                smiles = row["smiles"]
                labels = [bool(int(l)) if l else None for l in (row[k] for k in self.HEADERS)]
                yield smiles, labels

class Tox21Chem(Tox21Base):
    READER = dr.ChemDataReader