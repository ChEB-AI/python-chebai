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

import pandas as pd
from sklearn.model_selection import train_test_split
import requests
import torch
import time
import numpy as np
import tqdm
from datetime import datetime

from chebai.preprocessing import reader as dr
from chebai.preprocessing.datasets.base import DataLoader, XYBaseDataModule
from chebai.preprocessing.datasets.chebi import (
    ChEBIOver50,
    ChEBIOver100,
    ChEBIOverX,
)
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


class PubChem(XYBaseDataModule):
    SMILES_INDEX = 0
    LABEL_INDEX = 1
    FULL = 0
    UNLABELED = True
    READER = dr.ChemDataReader

    def __init__(self, *args, k=100000, **kwargs):
        self._k = k
        current_year = datetime.today().year
        current_month = datetime.today().month
        self.pubchem_url = f"https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Monthly/{current_year}-{current_month:02d}-01/Extras/CID-SMILES.gz"

        super(PubChem, self).__init__(*args, **kwargs)

    @property
    def _name(self):
        return f"Pubchem"

    @property
    def identifier(self):
        return self.reader.name(), self.split_label

    @property
    def split_label(self):
        if self._k and self._k != self.FULL:
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
        if not os.path.isfile(os.path.join(self.raw_dir, "smiles.txt")):
            if self._k == PubChem.FULL:
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
                with open(
                    os.path.join(full_dataset.raw_dir, "smiles.txt"), "r"
                ) as f_in:
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

        self.reader.on_finish()

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


class PubChemDissimilar(PubChem):
    """Subset of PubChem, but choosing the most dissimilar molecules (according to fingerprint)"""

    def __init__(
        self, *args, k=100000, n_random_subsets=100, random_size_factor=5, **kwargs
    ):
        """k: number of entries in this dataset,
        n_random_subsets: number of subsets of random data from which to draw
        the most dissimilar molecules,
        random_size_factor: size of random subsets (in total) in relation to k"""
        self.n_random_subsets = n_random_subsets
        self.random_size_factor = random_size_factor
        super(PubChemDissimilar, self).__init__(*args, k=k, **kwargs)

    @property
    def _name(self):
        return f"PubchemDissimilar"

    def download(self):
        if self._k == PubChem.FULL:
            super().download()
        else:
            # split random subset into n parts, from each part, select the most dissimilar entities
            random_dataset = PubChem(k=self._k * self.random_size_factor)
            random_dataset.download()

            with open(os.path.join(random_dataset.raw_dir, "smiles.txt"), "r") as f_in:
                random_smiles = [
                    [x.strip() for x in s.split("\t")] for s in f_in.readlines()
                ]
                fpgen = AllChem.GetRDKitFPGenerator()
                selected_smiles = []
                print(f"Selecting most dissimilar values from random subsets...")
                for i in tqdm.tqdm(range(self.n_random_subsets)):
                    smiles_i = random_smiles[
                        i
                        * len(random_smiles)
                        // self.n_random_subsets : (i + 1)
                        * len(random_smiles)
                        // self.n_random_subsets
                    ]
                    mols_i = [Chem.MolFromSmiles(smiles) for _, smiles in smiles_i]
                    fps = [
                        fpgen.GetFingerprint(m) if m is not None else m for m in mols_i
                    ]
                    nonnull_fps = [fp for fp in fps if fp is not None]
                    similarity = []
                    for i, fp in enumerate(fps):
                        try:
                            if fp is not None:
                                bulk = DataStructs.BulkTanimotoSimilarity(
                                    fp, nonnull_fps
                                )
                                similarity.append(sum(bulk))
                            else:
                                similarity.append(len(smiles_i))
                        except Exception as e:
                            print(i, smiles_i[i])
                            print(e.with_traceback(None))
                            similarity.append(len(smiles_i))

                    similarity = sorted(zip(smiles_i, similarity), key=lambda x: x[1])
                    selected_smiles += list(
                        list(
                            zip(*similarity[: len(smiles_i) // self.random_size_factor])
                        )[0]
                    )
            with open(os.path.join(self.raw_dir, "smiles.txt"), "w") as f_out:
                f_out.writelines(
                    "\n".join(["\t".join(smiles) for smiles in selected_smiles])
                )


class PubChemKMeans(PubChem):

    def __init__(self, *args, n_clusters=1e4, random_size=1e6, **kwargs):
        """k: number of entries in this dataset,
        n_random_subsets: number of subsets of random data from which to draw
        the most dissimilar molecules,
        random_size_factor: size of random subsets (in total) in relation to k"""
        self.n_clusters = int(n_clusters)
        super(PubChemKMeans, self).__init__(*args, k=int(random_size), **kwargs)

    @property
    def _name(self):
        return f"PubchemKMeans"

    def download(self):
        if self._k == PubChem.FULL:
            super().download()
        else:
            print(f"Loading random dataset (size: {self._k})...")
            random_dataset = PubChem(k=self._k)
            random_dataset.download()
            fingerprints_path = os.path.join(self.raw_dir, "fingerprints.pkl")
            if not os.path.exists(fingerprints_path):
                with open(
                    os.path.join(random_dataset.raw_dir, "smiles.txt"), "r"
                ) as f_in:
                    random_smiles = [s.split("\t")[1].strip() for s in f_in.readlines()]
                    fpgen = AllChem.GetRDKitFPGenerator()
                    selected_smiles = []
                    print(f"Converting SMILES to molecules...")
                    mols = [Chem.MolFromSmiles(s) for s in tqdm.tqdm(random_smiles)]
                    print(f"Generating Fingerprints...")
                    fps = [
                        fpgen.GetFingerprint(m) if m is not None else m
                        for m in tqdm.tqdm(mols)
                    ]
                    similarity = []
                    d = {"smiles": random_smiles, "fps": fps}
                    df = pd.DataFrame(d, columns=["smiles", "fps"])
                    df = df.dropna()
                    df.to_pickle(open(fingerprints_path, "wb"))
            else:
                df = pd.read_pickle(open(fingerprints_path, "rb"))
            fps = np.array([list(vec) for vec in df["fps"].tolist()])
            print(f"Starting k-means clustering...")
            start_time = time.perf_counter()
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=0, n_init="auto")
            kmeans.fit(fps)
            print(f"Finished k-means in {time.perf_counter() - start_time:.2f} seconds")
            df["label"] = kmeans.labels_
            df.to_pickle(
                open(
                    os.path.join(
                        self.raw_dir, f"fingerprints_labeled_{self.n_clusters}.pkl"
                    ),
                    "wb",
                )
            )
            cluster_df = pd.DataFrame(
                data={"centers": [center for center in kmeans.cluster_centers_]}
            )
            cluster_df.to_pickle(
                open(
                    os.path.join(
                        self.raw_dir, f"cluster_centers_{self.n_clusters}.pkl"
                    ),
                    "wb",
                )
            )


class PubChemDissimilarSMILES(PubChemDissimilar):
    READER = dr.ChemDataReader


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
        return f"PubChemHazardous"

    def setup_processed(self):
        # Collect token distribution
        filename = os.path.join(self.raw_dir, self.raw_file_names[0])
        print("Load data from file", filename)
        data = self._load_data_from_file(filename)
        torch.save(data, os.path.join(self.processed_dir, f"all.pt"))

        self.reader.on_finish()

    def processed_file_names(self):
        return ["all.pt"]

    def download(self):
        # requires the / a hazardous subset from pubchem, e.g. obtained by entering
        # "PubChem: PubChem Compound TOC: GHS Classification" in the pubchem search -> download -> csv
        csv_path = os.path.join(self.raw_dir, "pubchem_hazardous_compound_list.csv")
        compounds = pd.read_csv(csv_path)
        smiles_list = []
        for id, compound in compounds.iterrows():
            if (
                not isinstance(compound["cmpdsynonym"], str)
                or "CHEBI" not in compound["cmpdsynonym"]
            ):
                smiles_list.append(f"{compound['cid']}\t{compound['isosmiles']}")
        with open(os.path.join(self.raw_dir, "smiles.txt"), "w") as f:
            f.write("\n".join(smiles_list))


if __name__ == "__main__":
    kmeans_data = PubChemKMeans()
    kmeans_data.download()


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


class LabeledUnlabeledMixed(XYBaseDataModule):
    READER = dr.ChemDataReader

    def __init__(
        self, labeled: XYBaseDataModule, unlabeled: XYBaseDataModule, *args, **kwargs
    ):
        self.labeled = labeled
        self.unlabeled = unlabeled
        super().__init__(*args, **kwargs)

    @property
    def _name(self):
        return f"Mixed_{self.labeled._name}_{self.unlabeled._name}"

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


class PubToxAndChebiX(LabeledUnlabeledMixed):
    READER = dr.ChemDataReader
    CHEBI_X = ChEBIOverX

    def __init__(self, *args, **kwargs):
        super().__init__(
            self.CHEBI_X(*args, **kwargs), PubchemChem(*args, **kwargs), *args, **kwargs
        )

    @property
    def _name(self):
        return "PubToxU" + self.labeled._name


class PubToxAndChebi100(PubToxAndChebiX):
    CHEBI_X = ChEBIOver100


class PubToxAndChebi50(PubToxAndChebiX):
    CHEBI_X = ChEBIOver50


class PubChemDeepSMILES(PubChem):
    READER = dr.DeepChemDataReader


class PubChemSELFIES(PubChem):
    READER = dr.SelfiesReader
