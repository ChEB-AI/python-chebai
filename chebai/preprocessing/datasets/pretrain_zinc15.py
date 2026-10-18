import os
from typing import Any, Generator

import deepchem as dc
import numpy as np
import pandas as pd
from deepchem.data import DiskDataset
from torch.utils.data import DataLoader

from chebai.preprocessing import reader as dr
from chebai.preprocessing.datasets.base import _DynamicDataset


class Zinc15Pretrain(_DynamicDataset):
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
        data = self.dc_loader_api()

        for mol, labels, w, idx in data.itersamples():
            # https://deepchem.readthedocs.io/en/latest/api_reference/moleculenet.html
            # Note that the “w” matrix represents the weight of each sample. Some assays may have missing values, in which case the weight is 0.
            # Otherwise, the weight is 1. This is when `transformers` are set to `[]` in the deepchem data loader API.
            # By default `transformers` is set to ['balancing'], which doesn't hand you the raw 0/1 weight matrix.
            # Instead meaning DeepChem automatically applies a BalancingTransformer before returning the dataset.
            # That transformer reweights the observed labels per task so positive and negative examples end up with equal total weight — it upweights the rarer class.
            # Currently, transformers are set to `[]` in the deepchem data loader API, so we can get the raw 0/1 weight matrix.
            labels = [
                bool(label) if int(weight) != 0 else None
                for label, weight in zip(labels, w)
            ]
            yield dict(features=mol, labels=labels, ident=idx)
            splits.append({"id": idx, "split": "train"})
        splits_file_path = os.path.join(self.processed_dir_main, "splits.csv")
        if not os.path.exists(splits_file_path):
            splits_df = pd.DataFrame(splits)
            splits_df.to_csv(splits_file_path, index=False)

    def dc_loader_api(self) -> DiskDataset:
        """
        Load the dataset using DeepChem's MoleculeNet API.

        Returns:
            DiskDataset: The loaded dataset.
        """
        old_state = np.random.get_state()
        np.random.seed(self.dynamic_data_split_seed)
        try:
            return self._deep_chem_data_loader_api()
        finally:
            np.random.set_state(old_state)

    def _get_data_splits(self) -> None:
        pass

    def _generate_dynamic_splits(self) -> None:
        raise ValueError(
            "Custom splits are not supported for MoleculeNet datasets. "
            "Please use the predefined splits provided by the deepchem community"
            "by using `--splits_file_path=<path/to/splits.csv>`"
        )

    def _check_retrieved_splits(self) -> None:
        assert len(self._dynamic_df_train) > 0, (
            "No training data found after applying splits"
        )

    def val_dataloader(self, *args, **kwargs) -> None:
        return None

    @property
    def base_dir(self) -> str:
        """
        Return the base directory path for data.

        Returns:
            str: The base directory path for data.
        """
        return os.path.join("data", f"{self._name}:Pretrain")

    @property
    def raw_file_names_dict(self) -> None:
        """Returns a dictionary of raw file names."""
        pass

    def _deep_chem_data_loader_api(
        self,
    ) -> DiskDataset:
        # Random splitting is recommended for this dataset.
        tasks, datasets, transformers = dc.molnet.load_zinc15(  # type: ignore
            featurizer="Raw",
            splitter=None,
            data_dir=self.raw_dir,
            save_dir=self.processed_dir_main,
            dataset_size="250K",
            transformers=[],
        )
        return datasets[0]

    @property
    def tokens_directory(self) -> str:
        return "_shared"

    @property
    def _name(self) -> str:
        return "Zinc15"


if __name__ == "__main__":
    dataset = Zinc15Pretrain()
    dataset.prepare_data()
    dataset.setup()
