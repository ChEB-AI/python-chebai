import gzip
import os
import shutil
from abc import ABC, abstractmethod
from tempfile import NamedTemporaryFile
from typing import Any, Generator
from urllib import request

import deepchem as dc
import pandas as pd

from chebai.preprocessing import reader as dr
from chebai.preprocessing.datasets.base import _DynamicDataset
from chebai.preprocessing.splitters import GeneralSplitter, GroupSplitter


class MoleculeNetDataExtractor(_DynamicDataset, ABC):
    READER = dr.ChemDataReader

    @property
    def _name(self) -> str:
        """Returns the name of the dataset."""
        return str(self.__class__.__name__)

    def _preprocess_data_into_dataframe(self, raw_data_path: str) -> None:
        pass

    def _download_required_data(self) -> None:
        pass

    def save_processed(self, data: pd.DataFrame, filename: str) -> None:
        """
        Save the processed dataset to a pickle file.

        Args:
            data (pd.DataFrame): The processed dataset to be saved.
            filename (str): The filename for the pickle file.
        """
        if data is not None:
            data.to_pickle(open(os.path.join(self.processed_dir_main, filename), "wb"))

    def _get_data_size(self, input_file_path: str) -> None:
        pass

    @abstractmethod
    def _load_dict(
        self,
        input_file_path: str,
    ) -> Generator[dict[str, Any], None, None]:
        """Loads data from a CSV file.

        Args:
            input_file_path (str): Path to the CSV file.

        Returns:
            List[Dict]: List of data dictionaries.
        """
        pass

    def _get_data_splits(self) -> None:
        pass

    @property
    def base_dir(self) -> str:
        """
        Return the base directory path for data.

        Returns:
            str: The base directory path for data.
        """
        return os.path.join("data", f"{self._name}:MNClassification")

    @property
    def raw_file_names_dict(self) -> None:
        """Returns a dictionary of raw file names."""
        pass


class ClinTox(MoleculeNetDataExtractor, GroupSplitter):
    """Data module for ClinTox MoleculeNet dataset."""

    # Total: 1484, FDA_APPROVE is 1: 1390; CT_TOX is 1: 112
    # Multilabel splits?, stratified splits?
    LABLES_COLUMNS = [
        "FDA_APPROVED",
        "CT_TOX",
    ]

    @property
    def raw_file_names_dict(self) -> dict:
        """Returns a dictionary of raw file names."""
        return {"clintox": "clintox.csv"}


class BBBP(MoleculeNetDataExtractor, GroupSplitter):
    """Data module for BBBP MoleculeNet dataset."""

    def _load_dict(self, input_file_path: str) -> Generator[dict[str, Any], None, None]:
        """Loads data from a CSV file.

        Args:
            input_file_path (str): Path to the CSV file.

        Returns:
            List[Dict]: List of data dictionaries.
        """
        splits = []
        tasks, datasets, transformers = dc.molnet.load_bbbp(
            featurizer="Raw", splitter="scaffold"
        )
        train: dc.data.DiskDataset = datasets[0]
        valid: dc.data.DiskDataset = datasets[1]
        test: dc.data.DiskDataset = datasets[2]
        for split_name, data in [
            ("train", train),
            ("valid", valid),
            ("test", test),
        ]:
            for idx, (mol, labels, wi, smiles) in enumerate(data.itersamples()):
                yield dict(
                    features=smiles,
                    labels=labels,
                    ident=idx,
                )
                splits.append(
                    {
                        "id": idx,
                        "split": split_name,
                    }
                )
        splits_df = pd.DataFrame(splits)
        splits_df.to_csv(
            os.path.join(self.processed_dir_main, "splits.csv"), index=False
        )


class Sider(MoleculeNetDataExtractor, GroupSplitter):
    """Data module for Sider MoleculeNet dataset."""

    # Total 1427, multilabel splits, stratified splits?
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
    """Data module for Bace MoleculeNet dataset."""

    # Scaffold?
    # TODO: Train, val, test split already marked in data, which to use?
    # BINARY CLASSIFICATION task, total 1513, Class 1: 691
    # what are other columns?
    ID_COLUMN_NAME = "CID"
    FEATURE_COLUMN_NAME = "mol"
    LABELS_COLUMNS = [
        "Class",
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
    """Data module for HIV MoleculeNet dataset."""

    # Scaffold?
    # HIV: 82255, HIV_active is 1: 1443
    # Why activity not used CI: 39684, CM:1039, CA:404 columns? What are they?
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
    """Data module for MUV MoleculeNet dataset."""

    # remove row where all nan, or zeros ?
    # To much Nan values, Total: 186175
    # 27.0
    # 29.0
    # 30.0
    # 30.0
    # 29.0
    # 29.0
    # 30.0
    # 28.0
    # 29.0
    # 28.0
    # 29.0
    # 29.0
    # 30.0
    # 30.0
    # 29.0
    # 29.0
    # 24.0
    # Multilabel splits
    ID_COLUMN_NAME = "mol_id"
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
    dataset = BBBP()
    dataset.prepare_data()
    dataset.setup()
    # TODO: add TOX 21, TOX CAST, PCBA, ToxChallenge
    # import deepchem as dc

    # # https://deepchem.readthedocs.io/en/latest/api_reference/moleculenet.html
    # tasks, datasets, transformers = dc.molnet.load_bbbp(
    #     featurizer="Raw", splitter="scaffold"
    # )
    # # train, valid, test = datasets
    # train: dc.data.DiskDataset = datasets[0]
    # valid: dc.data.DiskDataset = datasets[1]
    # test: dc.data.DiskDataset = datasets[2]
    # for data in [train, valid, test]:
    #     for xi, yi, wi, idi in data.itersamples():
    #         print(
    #             f"features={xi}",  # SMILES
    #             f"labels={yi}",  # label (0/1)
    #             f"ident={idi}",  # molecule ID
    #         )

    # dc.molnet.load_hiv()
    # dc.molnet.load_bace_classification()
    # dc.molnet.load_tox21()
    # dc.molnet.load_toxcast()
    # dc.molnet.load_sider()
    # dc.molnet.load_clintox()
    # dc.molnet.load_muv()
