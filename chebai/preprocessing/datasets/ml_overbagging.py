import os
import random
from typing import Any

import pandas as pd
import tqdm

from chebai.preprocessing.datasets.base import _DynamicDataset


class _ResampledDynamicDataset(_DynamicDataset):
    """
    A dataset class that extends _DynamicDataset with an additional resampled data file (using the REMEDIAL algorithm).

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

        First runs the regular data preparation pipeline (producing ``data.pkl``),
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
        labels = data.columns[3:]
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
        majority_rows[minority_labels] = None

        minority_rows = high_scumble[data.columns].copy()
        minority_rows[majority_labels] = None

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
        for col in resampled_data.columns[3:]:
            resampled_data[col] = resampled_data[col].astype(bool)

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


def bootstrap_data(data: pd.DataFrame, train_instances: list[str]) -> pd.DataFrame:
    """
    Bootstrap the training instances in the dataset.

    Args:
        data (pd.DataFrame): The standard dataset as produced by the regular
            data preparation pipeline.

    Returns:
        pd.DataFrame: The bootstrapped dataset.
    """
    print("Bootstrapping data...")
    train_data = data[data["chebi_id"].isin(train_instances)]
    bootstrapped_data = train_data.sample(
        n=len(train_data), replace=True, random_state=42
    )
    # Add non-train instances back to the bootstrapped data
    non_train_data = data[~data["chebi_id"].isin(train_instances)]
    bootstrapped_data = pd.concat(
        [bootstrapped_data, non_train_data], ignore_index=True
    )
    return bootstrapped_data


class _BootstrapDynamicDataset(_DynamicDataset):
    """
    A dataset class that extends _DynamicDataset by bootstrapping the base dataset.

    Args:
        **kwargs: Additional keyword arguments passed to :class:`_DynamicDataset`.
    """

    def __init__(self, bag_name: str, input_data_file: str, **kwargs):
        # splits_file_path has to be provided
        if "splits_file_path" not in kwargs:
            raise ValueError(
                "`splits_file_path` must be provided for bootstrapping datasets. To generate a new dataset, use the regular dataset classes"
            )
        super().__init__(**kwargs)
        self.bag_name = bag_name
        self.input_data_file = input_data_file  # filename in processed_dir_main to use as input for bootstrapping

    # ------------------------------ Phase: Prepare data -----------------------------------
    def _perform_data_preparation(self, *args: Any, **kwargs: Any) -> None:
        """
        Prepares both the base data file and a bag.

        First runs the regular data preparation pipeline,
        then generates bags` by applying :meth:`_bootstrap_data` to the
        standard data.
        """

        bag_path = os.path.join(
            self.processed_dir_main, self.processed_main_file_names_dict["data"]
        )
        if not os.path.isfile(bag_path):
            print(
                f"Missing bag file (`{self.processed_main_file_names_dict['data']}`). Generating..."
            )
            standard_pkl_path = os.path.join(
                self.processed_dir_main, self.input_data_file
            )
            if standard_pkl_path is None:
                raise FileNotFoundError(
                    f"Standard data file `{standard_pkl_path}` not found "
                )
            standard_df = pd.read_pickle(standard_pkl_path)
            splits_df = pd.read_csv(self.splits_file_path)
            splits_df["id"] = splits_df["id"].astype(str)
            train_ids = splits_df[splits_df["split"] == "train"]["id"].values

            bag_df = bootstrap_data(standard_df, train_ids)
            self.save_processed(bag_df, self.processed_main_file_names_dict["data"])

    @property
    def processed_main_file_names_dict(self) -> dict:
        """
        Returns a dictionary of all main processed file names.
        """
        d = {"data": f"data_{self.bag_name}.pkl"}
        return d

    @property
    def processed_file_names_dict(self) -> dict:
        return {
            "data": f"data_{self.bag_name}.pt",
        }


def oversample(
    data: pd.DataFrame, train_instances: list[str], sampling_rate: float = 0.1
) -> pd.DataFrame:
    """
    Oversample the training instances in the dataset using ML-ROS.

    Args:
        data (pd.DataFrame): The standard dataset as produced by the regular dataset classes.
        train_instances (list[str]): A list of instance IDs to oversample.
        sampling_rate (float): The rate at which to oversample the training instances.

    Returns:
        pd.DataFrame: The oversampled dataset.
    """
    data = data.reset_index(drop=True)
    # Implementation for oversampling logic
    samples_to_add = sampling_rate * len(train_instances)
    print(f"Need to add {samples_to_add} samples to data")
    # calculate label imbalance ratios
    labels = data.columns[2:]
    label_frequencies = data[labels].sum()
    max_freq = label_frequencies.max()
    irlbl = max_freq / label_frequencies
    meanir = irlbl.mean()
    print(f"Mean imbalance ratio: {meanir:.2f}")
    # get bags for all labels where irlbl > meanir
    minority_labels = irlbl[irlbl > meanir].index
    print(f"Oversampling {len(minority_labels)} minority labels")
    minority_bags = dict()
    for label in minority_labels:
        minority_bags[label] = list(data[data[label] == 1].index)
    new_samples = []
    round_idx = 1
    while samples_to_add > 0:
        minority_bags_next_round = dict()
        for label, bag in minority_bags.items():
            new_sample = bag[random.randint(0, len(bag) - 1)]
            bag.append(new_sample)
            new_samples.append(new_sample)
            samples_to_add -= 1
            irlbl_bag = max_freq / len(bag)
            if irlbl_bag > meanir:
                minority_bags_next_round[label] = bag
        minority_bags = minority_bags_next_round
        if round_idx % 5 == 0:
            print(
                f"Round {round_idx} finished, {samples_to_add} samples to go, {len(minority_bags)} minority bags left"
            )
        round_idx += 1

    new_samples_df = data.iloc[new_samples]
    print(f"Adding {len(new_samples_df)} samples to data")
    return new_samples_df


class _MLROSDynamicDataset(_DynamicDataset):
    """
    A dataset class that extends _DynamicDataset by applying ML-ROS to the base dataset.
    Takes a dataset from which to oversample and a dataset to which to add the oversampled data as inputs
    (might be the same or different, e.g. sample from REMEDIAL dataset, add data to bags).

    Args:
        **kwargs: Additional keyword arguments passed to :class:`_DynamicDataset`.
    """

    def __init__(
        self,
        take_from_file: str,
        add_to_file: str,
        sampling_rate: float = 0.1,
        **kwargs,
    ):
        # splits_file_path has to be provided
        if "splits_file_path" not in kwargs:
            raise ValueError(
                "`splits_file_path` must be provided for ML-ROS datasets. To generate a new dataset, use the regular dataset classes"
            )
        super().__init__(**kwargs)
        self.take_from_file = take_from_file
        self.add_to_file = add_to_file
        self.sampling_rate = sampling_rate

    def _perform_data_preparation(self, *args: Any, **kwargs: Any) -> None:
        """
        Prepares the oversampled dataset.
        """

        oversampled_path = os.path.join(
            self.processed_dir_main, self.processed_main_file_names_dict["data"]
        )
        if not os.path.isfile(oversampled_path):
            print(
                f"Missing oversampled file (`{self.processed_main_file_names_dict['data']}`). Generating..."
            )
            take_from_pkl_path = os.path.join(
                self.processed_dir_main, self.take_from_file
            )
            add_to_pkl_path = os.path.join(self.processed_dir_main, self.add_to_file)
            if take_from_pkl_path is None:
                raise FileNotFoundError(f"File `{take_from_pkl_path}` not found ")
            if add_to_pkl_path is None:
                raise FileNotFoundError(f"File `{add_to_pkl_path}` not found ")
            take_from_df = pd.read_pickle(take_from_pkl_path)
            add_to_df = pd.read_pickle(add_to_pkl_path)
            splits_df = pd.read_csv(self.splits_file_path)
            splits_df["id"] = splits_df["id"].astype(str)
            train_ids = splits_df[splits_df["split"] == "train"]["id"].values
            extra_samples = oversample(take_from_df, train_ids, self.sampling_rate)
            add_to_df = pd.concat([add_to_df, extra_samples], ignore_index=True)

            self.save_processed(add_to_df, self.processed_main_file_names_dict["data"])

    @property
    def processed_main_file_names_dict(self) -> dict:
        """
        Returns a dictionary of all main processed file names.
        """
        d = {
            "data": f"{self.add_to_file[:-4]}_oversampled_with_{self.sampling_rate:.1f}_from_{self.take_from_file[:-4]}.pkl"
        }
        return d

    @property
    def processed_file_names_dict(self) -> dict:
        return {
            "data": f"{self.add_to_file[:-4]}_oversampled_with_{self.sampling_rate:.1f}_from_{self.take_from_file[:-4]}.pt",
        }
