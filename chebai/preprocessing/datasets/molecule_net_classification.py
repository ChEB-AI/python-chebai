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

    def __init__(
        self,
        test_split: float | None = None,
        validation_split: float | None = None,
        **kwargs,
    ):
        if test_split is not None or validation_split is not None:
            raise ValueError(
                "Custom splits are not supported for MoleculeNet datasets. "
                "Please use the predefined splits provided by the deepchem community"
                "by using `--splits_file_path=<path/to/splits.csv>`"
            )
        super().__init__(
            test_split=test_split, validation_split=validation_split, **kwargs
        )

    READER = dr.ChemDataReader

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
            data.to_pickle(os.path.join(self.processed_dir_main, filename))

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
        idx = 0
        for split_name, data in [
            ("train", train),
            ("validation", valid),
            ("test", test),
        ]:
            for mol, labels, wi, smiles in data.itersamples():
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
                idx += 1
        splits_file_path = os.path.join(self.processed_dir_main, "splits.csv")
        if not os.path.exists(splits_file_path):
            splits_df = pd.DataFrame(splits)
            splits_df.to_csv(splits_file_path, index=False)

    @abstractmethod
    def _deep_chem_data_loader_api(
        self,
    ) -> tuple[DiskDataset, DiskDataset, DiskDataset]:
        pass

    def _get_data_splits(self) -> None:
        pass

    def _generate_dynamic_splits(self) -> None:
        raise ValueError(
            "Custom splits are not supported for MoleculeNet datasets. "
            "Please use the predefined splits provided by the deepchem community"
            "by using `--splits_file_path=<path/to/splits.csv>`"
        )

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

    @property
    def data_type(self) -> str:
        return "clin_tox"

    @property
    def _name(self) -> str:
        return "ClinTox"


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

    @property
    def data_type(self) -> str:
        return "bbbp"

    @property
    def _name(self) -> str:
        return "BBBP"


class SIDER(MoleculeNetDataExtractor):
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

    @property
    def data_type(self) -> str:
        return "sider"

    @property
    def _name(self) -> str:
        return "SIDER"


class BACE(MoleculeNetDataExtractor):
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

    @property
    def data_type(self) -> str:
        return "bace"

    @property
    def _name(self) -> str:
        return "BACE"


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

    @property
    def data_type(self) -> str:
        return "hiv"

    @property
    def _name(self) -> str:
        return "HIV"


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

    @property
    def data_type(self) -> str:
        return "muv"

    @property
    def _name(self) -> str:
        return "MUV"


class Tox21(MoleculeNetDataExtractor):
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

    @property
    def data_type(self) -> str:
        return "tox21"

    @property
    def _name(self) -> str:
        return "Tox21"


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

    @property
    def data_type(self) -> str:
        return "toxcast"

    @property
    def _name(self) -> str:
        return "ToxCast"


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

    @property
    def data_type(self) -> str:
        return "pcba"

    @property
    def _name(self) -> str:
        return "PCBA"


if __name__ == "__main__":
    # Example usage
    dataset = BACE()
    dataset.prepare_data()
    dataset.setup()
