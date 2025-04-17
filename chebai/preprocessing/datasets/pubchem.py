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
import time
from datetime import datetime
from typing import Generator, List, Optional, Tuple, Type

import numpy as np
import pandas as pd
import requests
import torch
import tqdm
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from scipy import spatial
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

from chebai.preprocessing import reader as dr
from chebai.preprocessing.datasets.base import DataLoader, XYBaseDataModule
from chebai.preprocessing.datasets.chebi import (
    ChEBIOver50,
    ChEBIOver100,
    ChEBIOverX,
    _ChEBIDataExtractor,
)


class PubChem(XYBaseDataModule):
    """
    Dataset module for PubChem compounds.
    """

    SMILES_INDEX = 0
    LABEL_INDEX = 1
    FULL = 0
    UNLABELED = True
    READER = dr.ChemDataReader

    def __init__(self, *args, k: Optional[int] = 100000, **kwargs):
        """
        Args:
            k (Optional[int]): Number of samples to use. Set to `PubChem.FULL` for full dataset.
            *args: Additional arguments for superclass initialization.
            **kwargs: Additional keyword arguments for superclass initialization.
        """
        self._k = k
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
        return f"Pubchem"

    @property
    def identifier(self) -> tuple:
        """
        Returns:
            tuple: Tuple containing reader name and split label.
        """
        return self.reader.name(), self.split_label

    @property
    def split_label(self) -> str:
        """
        Returns:
            str: Label indicating the split of the dataset ('full' or a specific number).
        """
        if self._k and self._k != self.FULL:
            return str(self._k)
        else:
            return "full"

    @property
    def raw_dir(self) -> str:
        """
        Returns:
            str: Directory path where raw data is stored.
        """
        return os.path.join(self.base_dir, "raw", self.split_label)

    @staticmethod
    def _load_dict(input_file_path: str) -> Generator[dict, None, None]:
        """
        Args:
            input_file_path (str): Path to the input file.

        Yields:
            dict: Dictionary containing 'features', 'labels' (None), and 'ident' fields.
        """
        with open(input_file_path, "r") as input_file:
            for row in input_file:
                ident, smiles = row.split("\t")
                yield dict(features=smiles, labels=None, ident=ident)

    def download(self):
        """
        Downloads PubChem data based on `_k` parameter.
        """
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
        """
        Prepares processed data and saves them as Torch tensors.
        """
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
    def raw_file_names(self) -> List[str]:
        """
        Returns:
            List[str]: List of raw data file names.
        """
        return ["smiles.txt"]

    @property
    def processed_file_names(self) -> List[str]:
        """
        Returns:
            List[str]: List of processed data file names.
        """
        return ["test.pt", "train.pt", "validation.pt"]

    def _perform_data_preparation(self, *args, **kwargs):
        """
        Checks for raw data and downloads if necessary.
        """
        print("Check for raw data in", self.raw_dir)
        if any(
            not os.path.isfile(os.path.join(self.raw_dir, f))
            for f in self.raw_file_names
        ):
            print("Downloading data. This may take some time...")
            self.download()
            print("Done")


class PubChemDissimilar(PubChem):
    """
    Subset of PubChem, but choosing the most dissimilar molecules (according to fingerprint)
    """

    def __init__(
        self,
        *args,
        k: Optional[int] = 100000,
        n_random_subsets: Optional[int] = 100,
        random_size_factor: Optional[int] = 5,
        **kwargs,
    ):
        """
        Args:
            k (Optional[int]): Number of entries in this dataset.
            n_random_subsets (Optional[int]): Number of subsets of random data to draw most dissimilar molecules from.
            random_size_factor (Optional[int]): Size of random subsets in relation to k.
            *args: Additional arguments for superclass initialization.
            **kwargs: Additional keyword arguments for superclass initialization.
        """
        self.n_random_subsets = n_random_subsets
        self.random_size_factor = random_size_factor
        super(PubChemDissimilar, self).__init__(*args, k=k, **kwargs)

    @property
    def _name(self) -> str:
        """
        Returns:
            str: Name of the dataset.
        """
        return f"PubchemDissimilar"

    def download(self):
        """
        Downloads the PubChemDissimilar dataset.

        If `k` is set to `PubChem.FULL`, downloads the full dataset.
        Otherwise, generates random subsets to select the most dissimilar molecules based on fingerprints.
        """
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
    """
    Dataset class representing a subset of PubChem dataset clustered using K-Means algorithm.
    The idea is to create distinct distributions where pretraining and test sets are formed from dissimilar data.
    """

    def __init__(
        self,
        *args,
        n_clusters: int = 10000,
        random_size: int = 1000000,
        exclude_data_from: _ChEBIDataExtractor = None,
        validation_size_limit: int = 4000,
        include_min_n_clusters: int = 100,
        **kwargs,
    ):
        """
        Args:
            n_clusters (int): Number of clusters to create using K-Means.
            random_size (int): Size of random dataset to download.
            exclude_data_from (_ChEBIDataExtractor): Dataset which should not overlap with selected clusters
            (remove all clusters that contain data from this dataset).
            validation_size_limit (int): Validation set will contain at most this number of instances.
            include_min_n_clusters (int): Minimum number of clusters to keep if there are not enough clusters that don't
            overlap with the `exclude_data_from` dataset.

            *args: Additional arguments for superclass initialization.
            **kwargs: Additional keyword arguments for superclass initialization.
        """
        self.n_clusters = int(n_clusters)
        self.exclude_data_from = exclude_data_from
        self.validation_size_limit = validation_size_limit
        self.include_min_n_clusters = include_min_n_clusters
        super(PubChemKMeans, self).__init__(*args, k=int(random_size), **kwargs)
        self._fingerprints = None
        self._cluster_centers = None
        self._fingerprints_clustered = None
        self._exclusion_data_clustered = None
        self._cluster_centers_superclustered = None

    @property
    def _name(self) -> str:
        """
        Returns:
            str: Name of the dataset.
        """
        return f"PubchemKMeans"

    @property
    def split_label(self) -> str:
        """
        Returns:
            str: Label describing the split based on number of clusters.
        """
        if self._k and self._k != self.FULL:
            return f"{self.n_clusters}_centers_out_of_{self._k}"
        else:
            return f"{self.n_clusters}_centers_out_of_full"

    @property
    def raw_file_names(self) -> List[str]:
        """
        Clusters generated by K-Means, sorted by size (cluster0 is the largest).
        cluster0 is the training cluster (will be split into train/val/test in processed, used for pretraining)
        Returns:
            List[str]: List of raw file names expected in the raw directory.
        """
        return ["cluster0.txt", "cluster1.txt", "cluster2.txt"]

    @property
    def fingerprints(self) -> pd.DataFrame:
        """
        Creates random dataset, sanitises, creates Mol objects, generates fingerprints (RDKit)
        Saves `fingerprints_df` to `fingerprints.pkl`

        Returns:
            pd.DataFrame: DataFrame containing SMILES and corresponding fingerprints.
        """
        if self._fingerprints is None:
            fingerprints_path = os.path.join(self.raw_dir, "fingerprints.pkl")
            if not os.path.exists(fingerprints_path):
                print(f"No fingerprints found...")
                print(f"Loading random dataset (size: {self._k})...")
                random_dataset = PubChem(k=self._k)
                random_dataset.download()
                with open(
                    os.path.join(random_dataset.raw_dir, "smiles.txt"), "r"
                ) as f_in:
                    random_smiles = [s.split("\t")[1].strip() for s in f_in.readlines()]
                    fpgen = AllChem.GetRDKitFPGenerator()
                    print(f"Converting SMILES to molecules...")
                    mols = [Chem.MolFromSmiles(s) for s in tqdm.tqdm(random_smiles)]
                    print(f"Generating Fingerprints...")
                    fps = [
                        fpgen.GetFingerprint(m) if m is not None else m
                        for m in tqdm.tqdm(mols)
                    ]
                    d = {"smiles": random_smiles, "fps": fps}
                    fingerprints_df = pd.DataFrame(d, columns=["smiles", "fps"])
                    fingerprints_df = fingerprints_df.dropna()
                    fingerprints_df.to_pickle(open(fingerprints_path, "wb"))
                    self._fingerprints = fingerprints_df
            else:
                self._fingerprints = pd.read_pickle(open(fingerprints_path, "rb"))
        return self._fingerprints

    def _build_clusters(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Performs K-Means clustering on fingerprints and saves cluster information.

        Returns:
            tuple: Tuple containing cluster centers DataFrame and clustered fingerprints DataFrame.
        """
        fingerprints_clustered_path = os.path.join(
            self.raw_dir, "fingerprints_clustered.pkl"
        )
        cluster_centers_path = os.path.join(self.raw_dir, f"cluster_centers.pkl")
        print(f"Starting k-means clustering...")
        start_time = time.perf_counter()
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=0, n_init="auto")
        fps = np.array([list(vec) for vec in self.fingerprints["fps"].tolist()])
        kmeans.fit(fps)
        print(f"Finished k-means in {time.perf_counter() - start_time:.2f} seconds")
        fingerprints_df = self.fingerprints
        fingerprints_df["label"] = kmeans.labels_
        fingerprints_df.to_pickle(
            open(
                fingerprints_clustered_path,
                "wb",
            )
        )
        cluster_df = pd.DataFrame(
            data={"centers": [center for center in kmeans.cluster_centers_]}
        )
        cluster_df.to_pickle(
            open(
                cluster_centers_path,
                "wb",
            )
        )

        return cluster_df, fingerprints_df

    def _exclude_clusters(self, cluster_centers: pd.DataFrame) -> pd.DataFrame:
        """
        Excludes clusters based on data from an exclusion dataset (in a training setup, this is the labeled dataset,
        usually ChEBI). The goal is to avoid having similar data in the labeled training and the PubChem evaluation.

         Loads data from `exclude_data_from` dataset, generates mols, fingerprints, finds closest cluster centre for
         each fingerprint, saves data to `exclusion_data_clustered.pkl`, returns all clusters with no instances from the
         exclusion data (or the n clusters with the lowest number of instances if there are less than n clusters with no
         instances, n being the minimum number of clusters to include)

        Args:
            cluster_centers (pd.DataFrame): DataFrame of cluster centers.

        Returns:
            pd.DataFrame: DataFrame of filtered cluster centers.
        """
        exclusion_data_path = os.path.join(self.raw_dir, "exclusion_data_clustered.pkl")
        cluster_centers_np = np.array(
            [
                [cci for cci in cluster_center]
                for cluster_center in cluster_centers["centers"]
            ]
        )
        if self.exclude_data_from is not None:
            if not os.path.exists(exclusion_data_path):
                print(f"Loading data for exclusion of clusters...")
                raw_chebi = []
                for filename in self.exclude_data_from.raw_file_names:
                    raw_chebi.append(
                        pd.read_pickle(
                            open(
                                os.path.join(self.exclude_data_from.raw_dir, filename),
                                "rb",
                            )
                        )
                    )
                raw_chebi = pd.concat(raw_chebi)
                raw_chebi_smiles = np.array(raw_chebi["SMILES"])
                fpgen = AllChem.GetRDKitFPGenerator()
                print(f"Converting SMILES to molecules...")
                mols = [Chem.MolFromSmiles(s) for s in tqdm.tqdm(raw_chebi_smiles)]
                print(f"Generating Fingerprints...")
                chebi_fps = [
                    fpgen.GetFingerprint(m) if m is not None else m
                    for m in tqdm.tqdm(mols)
                ]
                print(f"Finding cluster for each instance from exclusion-data")
                chebi_fps = np.array([list(fp) for fp in chebi_fps if fp is not None])
                tree = spatial.KDTree(cluster_centers_np)
                chebi_clusters = [tree.query(fp)[1] for fp in chebi_fps]
                chebi_clusters_df = pd.DataFrame(
                    {"fp": [fp for fp in chebi_fps], "center_id": chebi_clusters},
                    columns=["fp", "center_id"],
                )
                chebi_clusters_df.to_pickle(open(exclusion_data_path, "wb"))
            else:
                chebi_clusters_df = pd.read_pickle(open(exclusion_data_path, "rb"))
            # filter pubchem clusters and remove all that contain data from the exclusion set
            print(f"Removing clusters with data from exclusion-set")
            counts = chebi_clusters_df["center_id"].value_counts()
            cluster_centers["n_chebi_instances"] = counts
            cluster_centers["n_chebi_instances"].fillna(0, inplace=True)
            cluster_centers.sort_values(
                by="n_chebi_instances", ascending=False, inplace=True
            )
            zero_centers = cluster_centers[cluster_centers["n_chebi_instances"] == 0]
            if len(zero_centers) > self.include_min_n_clusters:
                cluster_centers = zero_centers
            else:
                cluster_centers = cluster_centers[-self.include_min_n_clusters :]
        return cluster_centers

    @property
    def cluster_centers(self) -> pd.DataFrame:
        """
        Loads cluster centers from file if possible, otherwise calls `self._build_clusters()`.
        Returns:
            pd.DataFrame: DataFrame of cluster centers.
        """
        cluster_centers_path = os.path.join(self.raw_dir, f"cluster_centers.pkl")
        if self._cluster_centers is None:
            if os.path.exists(cluster_centers_path):
                self._cluster_centers = pd.read_pickle(open(cluster_centers_path, "rb"))
            else:
                self._cluster_centers = self._build_clusters()[0]
        return self._cluster_centers

    @property
    def fingerprints_clustered(self) -> pd.DataFrame:
        """
        Loads fingerprints with assigned clusters from file if possible, otherwise calls `self._build_clusters()`.
        Returns:
            pd.DataFrame: DataFrame of clustered fingerprints.
        """
        fingerprints_path = os.path.join(self.raw_dir, f"fingerprints_clustered.pkl")
        if self._fingerprints_clustered is None:
            if os.path.exists(fingerprints_path):
                self._fingerprints_clustered = pd.read_pickle(
                    open(fingerprints_path, "rb")
                )
            else:
                self._fingerprints_clustered = self._build_clusters()[1]
        return self._fingerprints_clustered

    @property
    def cluster_centers_superclustered(self) -> pd.DataFrame:
        """
        Calls `_exclude_clusters()` which removes all clusters that contain data from the exclusion set (usually the
        ChEBI, i.e., the labeled dataset).
        Runs KMeans with 3 clusters on remaining data, saves cluster centres with assigned supercluster-labels to
        `cluster_centers_superclustered.pkl`
        Returns:
            pd.DataFrame: DataFrame of superclustered cluster centers.
        """
        cluster_centers_path = os.path.join(
            self.raw_dir, f"cluster_centers_superclustered.pkl"
        )
        if self._cluster_centers_superclustered is None:
            if not os.path.exists(cluster_centers_path):
                clusters_filtered = self._exclude_clusters(self.cluster_centers)
                print(f"Superclustering PubChem clusters")
                kmeans = KMeans(n_clusters=3, random_state=0, n_init="auto")
                clusters_np = np.array(
                    [[cci for cci in center] for center in clusters_filtered["centers"]]
                )
                kmeans.fit(clusters_np)
                clusters_filtered["label"] = kmeans.labels_
                clusters_filtered.to_pickle(
                    open(
                        os.path.join(
                            self.raw_dir, "cluster_centers_superclustered.pkl"
                        ),
                        "wb",
                    )
                )
                self._cluster_centers_superclustered = clusters_filtered
            else:
                self._cluster_centers_superclustered = pd.read_pickle(
                    open(
                        os.path.join(
                            self.raw_dir, f"cluster_centers_superclustered.pkl"
                        ),
                        "rb",
                    )
                )
        return self._cluster_centers_superclustered

    def download(self):
        """
        Downloads the PubChemKMeans dataset. This function creates the complete dataset (including train, test, and
        validation splits). Most of the steps are hidden in properties (e.g., `self.fingerprints_clustered` triggers
        the download of a random dataset, the calculation of fingerprints for it and the KMeans clustering)
        The final splits are created by assigning all fingerprints that belong to a cluster of a certain supercluster
        to a dataset. This creates 3 datasets (for each of the 3 superclusters), the datasets are saved as validation,
        test and train based on their size. The validation set is limited to `self.validation_size_limit` entries.
        """
        if self._k == PubChem.FULL:
            super().download()
        else:
            if not all(
                os.path.exists(os.path.join(self.raw_dir, file))
                for file in self.raw_file_names
            ):
                fingerprints = self.fingerprints_clustered
                fingerprints["big_cluster_assignment"] = fingerprints["label"].apply(
                    lambda l: (
                        -1
                        if l not in self.cluster_centers_superclustered.index
                        else self.cluster_centers_superclustered.loc[int(l), "label"]
                    )
                )
                fp_grouped = fingerprints.groupby("big_cluster_assignment")
                splits = [fp_grouped.get_group(g) for g in fp_grouped.groups if g != -1]
                splits[0] = splits[0][: self.validation_size_limit]
                splits.sort(key=lambda x: len(x))
                for i, name in enumerate(["cluster2", "cluster1", "cluster0"]):
                    if not os.path.exists(os.path.join(self.raw_dir, f"{name}.txt")):
                        open(os.path.join(self.raw_dir, f"{name}.txt"), "x").close()
                    with open(os.path.join(self.raw_dir, f"{name}.txt"), "w") as f:
                        for id, row in splits[i].iterrows():
                            f.writelines(f"{id}\t{row['smiles']}\n")


class PubChemDissimilarSMILES(PubChemDissimilar):
    """
    Subset of PubChem, selecting most dissimilar molecules based on fingerprints.

    Inherits from PubChemDissimilar.

    Attributes:
        READER (type): Data reader type for chemical data.
    """

    READER: Type[dr.ChemDataReader] = dr.ChemDataReader


class SWJPreChem(PubChem):
    """
    Subset of PubChem with unlabeled data, specific to SWJPre.

    Inherits from PubChem.

    Attributes:
        UNLABELED (bool): Indicates if the data is unlabeled.
        _name (str): Name of the dataset.
    """

    UNLABELED: bool = True

    @property
    def _name(self) -> str:
        """
        Returns the name of the dataset.
        """
        return f"SWJpre"

    def download(self):
        """
        Raises an exception since required raw files are not found.
        """
        raise Exception("Required raw files not found")

    @property
    def identifier(self) -> Tuple[str]:
        """
        Returns the identifier for the dataset.
        """
        return (self.reader.name(),)

    @property
    def raw_dir(self) -> str:
        """
        Returns the directory path for raw data.
        """
        return os.path.join("data", self._name, "raw")


class SWJSelfies(SWJPreChem):
    """
    Subset of SWJPreChem using SelfiesReader for data reading.

    Inherits from SWJPreChem.

    Attributes:
        READER (type): Data reader type for chemical data (SelfiesReader).
    """

    READER: Type[dr.SelfiesReader] = dr.SelfiesReader


class PubchemChem(PubChem):
    """
    Subset of PubChem using ChemDataReader for data reading.

    Inherits from PubChem.

    Attributes:
        READER (type): Data reader type for chemical data (ChemDataReader).
    """

    READER: Type[dr.ChemDataReader] = dr.ChemDataReader


class PubchemBPE(PubChem):
    """
    Subset of PubChem using ChemBPEReader for data reading.

    Inherits from PubChem.

    Attributes:
        READER (type): Data reader type for chemical data (ChemBPEReader).
    """

    READER: Type[dr.ChemBPEReader] = dr.ChemBPEReader


class SWJChem(SWJPreChem):
    """
    Subset of SWJPreChem using ChemDataUnlabeledReader for data reading.

    Inherits from SWJPreChem.

    Attributes:
        READER (type): Data reader type for chemical data (ChemDataUnlabeledReader).
    """

    READER: Type[dr.ChemDataUnlabeledReader] = dr.ChemDataUnlabeledReader


class SWJBPE(SWJPreChem):
    """
    Subset of SWJPreChem using ChemBPEReader for data reading.

    Inherits from SWJPreChem.

    Attributes:
        READER (type): Data reader type for chemical data (ChemBPEReader).
    """

    READER: Type[dr.ChemBPEReader] = dr.ChemBPEReader


class PubChemTokens(PubChem):
    """
    Subset of PubChem using ChemDataReader for data reading.

    Inherits from PubChem.

    Attributes:
        READER (type): Data reader type for chemical data (ChemDataReader).
    """

    READER: Type[dr.ChemDataReader] = dr.ChemDataReader


class Hazardous(SWJChem):
    """
    Subset of SWJChem for hazardous compounds.

    Inherits from SWJChem.

    Attributes:
        READER (type): Data reader type for chemical data (ChemDataUnlabeledReader).
    """

    READER: Type[dr.ChemDataUnlabeledReader] = dr.ChemDataUnlabeledReader

    @property
    def _name(self) -> str:
        """
        Returns the name of the dataset.
        """
        return f"PubChemHazardous"

    def setup_processed(self):
        """
        Sets up the processed data.
        """
        filename = os.path.join(self.raw_dir, self.raw_file_names[0])
        print("Load data from file", filename)
        data = self._load_data_from_file(filename)
        torch.save(data, os.path.join(self.processed_dir, f"all.pt"))

        self.reader.on_finish()

    def processed_file_names(self) -> List[str]:
        """
        Returns the list of processed file names.
        """
        return ["all.pt"]

    def download(self):
        """
        Downloads hazardous compound data from PubChem.
        """
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


class SWJPreChem(PubChem):
    """
    Subset of PubChem specific to SWJpre with unlabeled data.

    Inherits from PubChem.

    Attributes:
        UNLABELED (bool): Indicates if the data is unlabeled.
        _name (str): Name of the dataset.
    """

    UNLABELED: bool = True

    @property
    def _name(self) -> str:
        """
        Returns the name of the dataset.

        Returns:
            str: Name of the dataset.
        """
        return f"SWJpre"

    def download(self) -> None:
        """
        Raises an exception since required raw files are not found.

        Raises:
            Exception: If required raw files are not found.
        """
        raise Exception("Required raw files not found")

    @property
    def identifier(self) -> Tuple[str]:
        """
        Returns the identifier for the dataset.

        Returns:
            tuple: A tuple containing the name of the reader.
        """
        return (self.reader.name(),)


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
        labeled_data = torch.load(
            os.path.join(self.labeled.processed_dir, f"{kind}.pt"), weights_only=False
        )
        unlabeled_data = torch.load(
            os.path.join(self.unlabeled.processed_dir, f"{kind}.pt"), weights_only=False
        )
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
            self.CHEBI_X(*args, **kwargs), PubchemChem(*args, **kwargs), *args, **kwargs
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
