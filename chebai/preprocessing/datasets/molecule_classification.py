import gzip
import os
import shutil
from abc import ABC
from tempfile import NamedTemporaryFile
from typing import Any, Generator, List
from urllib import request

import numpy as np
import pandas as pd

from chebai.preprocessing import reader as dr
from chebai.preprocessing.datasets.base import _DynamicDataset
from chebai.preprocessing.splitters import GeneralSplitter, GroupSplitter


class MoleculeNetDataExtractor(_DynamicDataset, ABC):
    READER = dr.ChemDataReader

    LABLES_COLUMNS = []
    FEATURE_COLUMN_NAME = "smiles"
    ID_COLUMN_NAME = None

    @property
    def _name(self) -> str:
        """Returns the name of the dataset."""
        return str(self.__class__.__name__)

    def _preprocess_data_into_dataframe(self, raw_data_path: str) -> pd.DataFrame:
        """
        Preprocesses the raw data into a DataFrame.

        Args:
            raw_data_path (str): Path to the raw data.

        Returns:
            pd.DataFrame: The preprocessed data as a DataFrame.
        """
        return pd.read_csv(raw_data_path, header=0)

    def _load_dict(self, input_file_path: str) -> Generator[dict[str, Any], None, None]:
        """Loads data from a CSV file.

        Args:
            input_file_path (str): Path to the CSV file.

        Returns:
            List[Dict]: List of data dictionaries.
        """
        with open(input_file_path, "rb") as input_file:
            df = pd.read_pickle(input_file)

        features = df[self.FEATURE_COLUMN_NAME].to_numpy()
        if self.ID_COLUMN_NAME is not None and self.ID_COLUMN_NAME in df.columns:
            idents = df[self.ID_COLUMN_NAME].to_numpy()
        else:
            idents = np.arange(len(df))
        labels = df[self.LABLES_COLUMNS].to_numpy()

        for feat, labels, ident in zip(features, labels, idents):
            yield dict(features=feat, labels=labels, ident=ident)

    @property
    def base_dir(self) -> str:
        """
        Return the base directory path for data.

        Returns:
            str: The base directory path for data.
        """
        return os.path.join("data", "MoleculeNetClassification")


class ClinTox(MoleculeNetDataExtractor, GroupSplitter):
    """Data module for ClinTox MoleculeNet dataset."""

    LABLES_COLUMNS = [
        "FDA_APPROVED",
        "CT_TOX",
    ]

    @property
    def raw_file_names_dict(self) -> dict:
        """Returns a dictionary of raw file names."""
        return {"clintox": "clintox.csv"}

    def _download_required_data(self) -> str:
        """Downloads and extracts the dataset."""
        with NamedTemporaryFile("rb") as gout:
            request.urlretrieve(
                "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv.gz",
                gout.name,
            )
            with gzip.open(gout.name) as gfile:
                with open(
                    os.path.join(self.raw_dir, self.raw_file_names_dict["clintox"]),
                    "wt",
                ) as fout:
                    fout.write(gfile.read().decode())
        return os.path.join(self.raw_dir, self.raw_file_names_dict["clintox"])


class BBBP(MoleculeNetDataExtractor, GroupSplitter):
    """Data module for ClinTox MoleculeNet dataset."""

    LABLES_COLUMNS = [
        "p_np",
    ]

    @property
    def raw_file_names_dict(self) -> dict:
        """Returns a dictionary of raw file names."""
        return {"bbbp": "bbbp.csv"}

    def _download_required_data(self) -> str:
        """Downloads and extracts the dataset."""
        with open(
            os.path.join(self.raw_dir, self.raw_file_names_dict["bbbp"]), "ab"
        ) as dst:
            with request.urlopen(
                "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv",
            ) as src:
                shutil.copyfileobj(src, dst)
        return os.path.join(self.raw_dir, self.raw_file_names_dict["bbbp"])


class Sider(MoleculeNetDataExtractor, GroupSplitter):
    """Data module for ClinTox MoleculeNet dataset."""

    LABLES_COLUMNS = [
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
    def raw_file_names_dict(self) -> dict:
        """Returns a dictionary of raw file names."""
        return {"sider": "sider.csv"}

    def _download_required_data(self) -> str:
        """Downloads and extracts the dataset."""
        with NamedTemporaryFile("rb") as gout:
            request.urlretrieve(
                "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/sider.csv.gz",
                gout.name,
            )
            with gzip.open(gout.name) as gfile:
                with open(
                    os.path.join(self.raw_dir, self.raw_file_names_dict["sider"]), "wt"
                ) as fout:
                    fout.write(gfile.read().decode())
        return os.path.join(self.raw_dir, self.raw_file_names_dict["sider"])


class Bace(MoleculeNetDataExtractor, GeneralSplitter):
    """Data module for ClinTox MoleculeNet dataset."""

    LABELS_COLUMNS = [
        "class",
    ]

    @property
    def raw_file_names_dict(self) -> dict:
        """Returns a dictionary of raw file names."""
        return {"bace": "bace.csv"}

    def _download_required_data(self) -> str:
        """Downloads and extracts the dataset."""
        with open(
            os.path.join(self.raw_dir, self.raw_file_names_dict["bace"]), "ab"
        ) as dst:
            with request.urlopen(
                "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv",
            ) as src:
                shutil.copyfileobj(src, dst)
        return os.path.join(self.raw_dir, self.raw_file_names_dict["bace"])


class HIV(MoleculeNetDataExtractor, GroupSplitter):
    """Data module for ClinTox MoleculeNet dataset."""

    LABELS_COLUMNS = [
        "HIV_active",
    ]

    @property
    def raw_file_names_dict(self) -> dict:
        """Returns a dictionary of raw file names."""
        return {"hiv": "hiv.csv"}

    def _download_required_data(self) -> str:
        """Downloads and extracts the dataset."""
        with open(
            os.path.join(self.raw_dir, self.raw_file_names_dict["hiv"]), "ab"
        ) as dst:
            with request.urlopen(
                "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/HIV.csv",
            ) as src:
                shutil.copyfileobj(src, dst)
        return os.path.join(self.raw_dir, self.raw_file_names_dict["hiv"])


class MUV(MoleculeNetDataExtractor, GroupSplitter):
    """Data module for ClinTox MoleculeNet dataset."""

    LABELS_COLUMNS = [
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
    def raw_file_names_dict(self) -> dict:
        """Returns a dictionary of raw file names."""
        return {"muv": "muv.csv"}

    def _download_required_data(self) -> str:
        """Downloads and extracts the dataset."""
        with NamedTemporaryFile("rb") as gout:
            request.urlretrieve(
                "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/muv.csv.gz",
                gout.name,
            )
            with gzip.open(gout.name) as gfile:
                with open(
                    os.path.join(self.raw_dir, self.raw_file_names_dict["muv"]), "wt"
                ) as fout:
                    fout.write(gfile.read().decode())

        return os.path.join(self.raw_dir, self.raw_file_names_dict["muv"])


if __name__ == "__main__":
    # Example usage
    dataset = ClinTox()
    dataset.prepare_data()
    dataset.setup()
