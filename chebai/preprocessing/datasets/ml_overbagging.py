import os
from typing import Any

import pandas as pd
import tqdm

from chebai.preprocessing.datasets.base import _DynamicDataset


class _ResampledDynamicDataset(_DynamicDataset):
    """
    A dataset class that extends _DynamicDataset with an additional resampled data file.

    Args:
        **kwargs: Additional keyword arguments passed to :class:`_DynamicDataset`.
    """

    _RESAMPLED_PKL_FILENAME: str = "data_resampled.pkl"

    def __init__(self, **kwargs):
        # splits_file_path has to be provided
        if "splits_file_path" not in kwargs:
            raise ValueError(
                "`splits_file_path` must be provided for resampled datasets. To generate a new dataset, use the regular dataset classes"
            )
        super().__init__(**kwargs)

    # ------------------------------ Phase: Prepare data -----------------------------------
    def _perform_data_preparation(self, *args: Any, **kwargs: Any) -> None:
        """
        Prepares both the standard and resampled data files.

        First runs the regular data preparation pipeline (producing ``data_standard.pkl``),
        then generates ``data_resampled.pkl`` by applying :meth:`_resample_data` to the
        standard data.
        """

        resampled_path = os.path.join(
            self.processed_dir_main, self._RESAMPLED_PKL_FILENAME
        )
        if not os.path.isfile(resampled_path):
            print(
                f"Missing resampled data file (`{self._RESAMPLED_PKL_FILENAME}`). Generating..."
            )
            standard_pkl_path = os.path.join(
                self.processed_dir_main, self.processed_main_file_names_dict["data"]
            )
            if standard_pkl_path is None:
                raise FileNotFoundError(
                    f"Standard data file `{self._STANDARD_PKL_FILENAME}` not found "
                    f"in {self.processed_dir_main}"
                )
            standard_df = pd.read_pickle(standard_pkl_path)
            splits_df = pd.read_csv(self.splits_file_path)
            splits_df["id"] = splits_df["id"].astype(str)
            train_ids = splits_df[splits_df["split"] == "train"]["id"].values

            resampled_df = self._resample_data(standard_df, train_ids)
            self.save_processed(resampled_df, self._RESAMPLED_PKL_FILENAME)

    def scumble(self, label_imbalance_ratios):
        if len(label_imbalance_ratios) == 0:
            return None
        geometric_mean_ir = label_imbalance_ratios.prod() ** (
            1 / len(label_imbalance_ratios)
        )
        arithmetic_mean_ir = label_imbalance_ratios.mean()
        scumble_score = 1 - geometric_mean_ir / arithmetic_mean_ir
        return scumble_score

    def _resample_data(
        self, data: pd.DataFrame, train_instances: list[str]
    ) -> pd.DataFrame:
        """
        Resample the standard ChEBI dataset with REMEDIAL.

        Args:
            data (pd.DataFrame): The standard dataset as produced by the regular
                data preparation pipeline.

        Returns:
            pd.DataFrame: The resampled dataset.
        """
        print("Resampling with REMEDIAL...")
        print(data.head())
        labels = data.columns[2:]
        print(f"Number of labels: {len(labels)}, first 10 labels: {labels[:10]}")
        label_frequencies = data[labels].sum()
        print("Label frequencies before resampling:")
        print(len(label_frequencies), label_frequencies[:10])
        max_freq = label_frequencies.max()
        print(f"Maximum label frequency: {max_freq}")
        irlbl = max_freq / label_frequencies
        print("Imbalance ratio per label:")
        print(len(irlbl), irlbl[:10])
        meanir = irlbl.mean()
        print(f"Mean imbalance ratio: {meanir}")
        with open(
            os.path.join(self.processed_dir_main, "label_imbalance_ratios.csv"), "w"
        ) as f:
            for label, ir in irlbl.items():
                f.write(f"{label},{ir}\n")

        train_data = data[data["chebi_id"].isin(train_instances)]
        if os.path.isfile(os.path.join(self.processed_dir_main, "data_scumble.csv")):
            print("Scumble scores already calculated, loading from file...")
            scumble_df = pd.read_csv(
                os.path.join(self.processed_dir_main, "data_scumble.csv")
            )
            scumble_df["chebi_id"] = scumble_df["chebi_id"].astype(str)
            scumble_dict = dict(zip(scumble_df["chebi_id"], scumble_df["scumble"]))
            train_data["scumble"] = train_data["chebi_id"].map(scumble_dict)
        else:
            for row in tqdm.tqdm(
                train_data.itertuples(),
                total=len(train_data),
                desc="Calculating scumble scores",
            ):
                label_values = row[3:]
                label_imbalance_ratios = irlbl[[v == 1 for v in label_values]]
                scumble_score = self.scumble(label_imbalance_ratios)
                train_data.at[row[0], "scumble"] = scumble_score
            with open(
                os.path.join(self.processed_dir_main, "data_scumble.csv"), "w"
            ) as f:
                f.write("chebi_id,scumble\n")
                for row in train_data.itertuples():
                    f.write(f"{row.chebi_id},{row.scumble}\n")
        scumble_mean = train_data["scumble"].mean()
        print(f"Mean scumble score: {scumble_mean}")

        # split labels into majority labels (irlbl > meanir) and minority labels (irlbl <= meanir)
        minority_labels = irlbl[irlbl > meanir].index
        majority_labels = irlbl[irlbl <= meanir].index
        print(
            f"Majority labels: {len(majority_labels)}, first 10: {majority_labels[:10]}"
        )
        print(
            f"Minority labels: {len(minority_labels)}, first 10: {minority_labels[:10]}"
        )

        # split instances where scumble > mean into two copies, one with only majority labels and one with only minority labels
        # Drop train instances with NaN scumble (no labels)
        nan_scumble_idx = train_data.index[train_data["scumble"].isna()]
        # Identify train instances to split
        high_scumble = train_data[train_data["scumble"] > scumble_mean]

        # Build majority and minority copies of high-scumble rows with zeroed-out labels
        majority_rows = high_scumble[data.columns].copy()
        majority_rows[minority_labels] = 0

        minority_rows = high_scumble[data.columns].copy()
        minority_rows[majority_labels] = 0

        # Indices to remove from the original data: NaN-scumble rows + rows that were split
        indices_to_drop = nan_scumble_idx.union(high_scumble.index)

        resampled_data = pd.concat(
            [
                data.drop(index=indices_to_drop.intersection(data.index)),
                majority_rows,
                minority_rows,
            ],
            ignore_index=True,
        )
        print(
            "Data resampling completed, dataset size after resampling:",
            len(resampled_data),
        )
        print(resampled_data.head())
        return resampled_data

    # ------------------------------ Properties -----------------------------------
    @property
    def processed_main_file_names_dict(self) -> dict:
        """
        Returns a dictionary of all main processed file names, including both the
        standard and resampled pickle files.
        """
        d = super().processed_main_file_names_dict
        d["data_resampled"] = self._RESAMPLED_PKL_FILENAME
        return d

    @property
    def processed_file_names_dict(self) -> dict:
        return {
            "data": "data_resampled.pt",
        }

    def setup_processed(self) -> None:
        """
        Instead of data.pkl, use resampled data as basis for processing

        Returns:
            None
        """
        os.makedirs(self.processed_dir, exist_ok=True)
        transformed_file_name = self.processed_file_names_dict["data"]
        print(
            f"Missing transformed data (`{transformed_file_name}` file). Transforming data.... "
        )
        import torch

        torch.save(
            self._load_data_from_file(
                os.path.join(
                    self.processed_dir_main,
                    self.processed_main_file_names_dict["data_resampled"],
                )
            ),
            os.path.join(self.processed_dir, transformed_file_name),
        )
