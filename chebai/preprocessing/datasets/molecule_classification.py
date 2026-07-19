import os
from abc import ABC, abstractmethod
from typing import Any, Generator

import deepchem as dc
import pandas as pd
from deepchem.data import DiskDataset

from chebai.preprocessing import reader as dr
from chebai.preprocessing.datasets.base import _DynamicDataset


class MoleculeNetDataExtractor(_DynamicDataset, ABC):
    """
    Base class for MoleculeNet dataset extraction and preprocessing.

    Reference:
        - https://deepchem.readthedocs.io/en/latest/api_reference/moleculenet.html
        - Zhenqin Wu, Bharath Ramsundar, Evan N. Feinberg, Joseph Gomes, Caleb Geniesse,
            Aneesh S. Pappu, Karl Leswing, Vijay Pande; MoleculeNet: a benchmark for molecular
            machine learning. Chem. Sci. 2018; 9 (2): 513–530. https://doi.org/10.1039/c7sc02664a
    """

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

    def _load_dict(self, input_file_path: str) -> Generator[dict[str, Any], None, None]:
        """Loads data from a CSV file.

        Args:
            input_file_path (str): Path to the CSV file.

        Returns:
            List[Dict]: List of data dictionaries.
        """
        splits = []
        train, valid, test = self._deep_chem_data_loader_api()
        for split_name, data in [
            ("train", train),
            ("valid", valid),
            ("test", test),
        ]:
            for idx, (mol, labels, wi, smiles) in enumerate(data.itersamples()):
                yield dict(
                    features=mol,
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

    @abstractmethod
    def _deep_chem_data_loader_api(
        self,
    ) -> tuple[DiskDataset, DiskDataset, DiskDataset]:
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


class ClinTox(MoleculeNetDataExtractor):
    """Data module for ClinTox MoleculeNet dataset."""

    def _deep_chem_data_loader_api(
        self,
    ) -> tuple[DiskDataset, DiskDataset, DiskDataset]:
        # Random splitting is recommended for this dataset.
        tasks, datasets, transformers = dc.molnet.load_clintox(
            featurizer="Raw",
            splitter="random",
            data_dir=self.raw_dir,
            save_dir=self.processed_dir_main,
        )
        return datasets


class BBBP(MoleculeNetDataExtractor):
    """Data module for BBBP MoleculeNet dataset."""

    def _deep_chem_data_loader_api(
        self,
    ) -> tuple[DiskDataset, DiskDataset, DiskDataset]:
        # Scaffold splitting is recommended for this dataset.
        tasks, datasets, transformers = dc.molnet.load_bbbp(
            featurizer="Raw",
            splitter="scaffold",
            data_dir=self.raw_dir,
            save_dir=self.processed_dir_main,
        )
        return datasets


class Sider(MoleculeNetDataExtractor):
    """Data module for Sider MoleculeNet dataset."""

    def _deep_chem_data_loader_api(
        self,
    ) -> tuple[DiskDataset, DiskDataset, DiskDataset]:
        # Random splitting is recommended for this dataset.
        tasks, datasets, transformers = dc.molnet.load_sider(
            featurizer="Raw",
            splitter="random",
            data_dir=self.raw_dir,
            save_dir=self.processed_dir_main,
        )
        return datasets


class Bace(MoleculeNetDataExtractor):
    """Data module for Bace MoleculeNet dataset."""

    def _deep_chem_data_loader_api(
        self,
    ) -> tuple[DiskDataset, DiskDataset, DiskDataset]:
        # Scaffold splitting is recommended for this dataset.
        tasks, datasets, transformers = dc.molnet.load_bace_classification(
            featurizer="Raw",
            splitter="scaffold",
            data_dir=self.raw_dir,
            save_dir=self.processed_dir_main,
        )
        return datasets


class HIV(MoleculeNetDataExtractor):
    """Data module for HIV MoleculeNet dataset."""

    def _deep_chem_data_loader_api(
        self,
    ) -> tuple[DiskDataset, DiskDataset, DiskDataset]:
        # Scaffold splitting is recommended for this dataset.
        tasks, datasets, transformers = dc.molnet.load_hiv(
            featurizer="Raw",
            splitter="scaffold",
            data_dir=self.raw_dir,
            save_dir=self.processed_dir_main,
        )
        return datasets


class MUV(MoleculeNetDataExtractor):
    """Data module for MUV MoleculeNet dataset."""

    def _deep_chem_data_loader_api(
        self,
    ) -> tuple[DiskDataset, DiskDataset, DiskDataset]:
        # Scaffold splitting is recommended for this dataset.
        tasks, datasets, transformers = dc.molnet.load_muv(
            featurizer="Raw",
            splitter="scaffold",
            data_dir=self.raw_dir,
            save_dir=self.processed_dir_main,
        )
        return datasets


class Tox21MolNet(MoleculeNetDataExtractor):
    """Data module for Tox21MolNet dataset."""

    def _deep_chem_data_loader_api(
        self,
    ) -> tuple[DiskDataset, DiskDataset, DiskDataset]:
        # Random splitting is recommended for this dataset.
        tasks, datasets, transformers = dc.molnet.load_tox21(
            featurizer="Raw",
            splitter="random",
            data_dir=self.raw_dir,
            save_dir=self.processed_dir_main,
        )
        return datasets


class ToxCast(MoleculeNetDataExtractor):
    """Data module for ToxCast MoleculeNet dataset."""

    def _deep_chem_data_loader_api(
        self,
    ) -> tuple[DiskDataset, DiskDataset, DiskDataset]:
        # Random splitting is recommended for this dataset.
        tasks, datasets, transformers = dc.molnet.load_toxcast(
            featurizer="Raw",
            splitter="random",
            data_dir=self.raw_dir,
            save_dir=self.processed_dir_main,
        )
        return datasets


class PCBA(MoleculeNetDataExtractor):
    """Data module for PCBA MoleculeNet dataset."""

    def _deep_chem_data_loader_api(
        self,
    ) -> tuple[DiskDataset, DiskDataset, DiskDataset]:
        # Random splitting is recommended for this dataset.
        tasks, datasets, transformers = dc.molnet.load_pcba(
            featurizer="Raw",
            splitter="random",
            data_dir=self.raw_dir,
            save_dir=self.processed_dir_main,
        )
        return datasets


if __name__ == "__main__":
    # Example usage
    dataset = BBBP()
    dataset.prepare_data()
    dataset.setup()
