import argparse
import os
import shutil
from typing import Dict, List, Tuple, Type

import pandas as pd
import torch

from chebai.preprocessing.datasets.chebi import _ChEBIDataExtractor


class ChebiDataMigration:
    """
    A class to handle migration of ChEBI dataset to a new structure.

    Attributes:
        __MODULE_PATH (str): The path to the module containing ChEBI classes.
        __DATA_ROOT_DIR (str): The root directory for data.
        _chebi_cls (_ChEBIDataExtractor): The ChEBI class instance.
        _chebi_version (int): The version of the ChEBI dataset.
        _single_class (int, optional): The ID of a single class to predict.
        _class_name (str): The name of the ChEBI class.
    """

    __MODULE_PATH: str = "chebai.preprocessing.datasets.chebi"
    __DATA_ROOT_DIR: str = "data"

    def __init__(self, class_name: str, chebi_version: int, single_class: int = None):
        """
        Initialize the ChebiDataMigration class.

        Args:
            class_name (str): The name of the ChEBI class.
            chebi_version (int): The version of the ChEBI dataset.
            single_class (int, optional): The ID of the single class to predict.
        """
        self._chebi_cls: Type[_ChEBIDataExtractor] = self._dynamic_import_chebi_cls(
            class_name, chebi_version, single_class
        )
        self._chebi_version: int = chebi_version
        self._single_class: int = single_class
        self._class_name: str = class_name

    @classmethod
    def _dynamic_import_chebi_cls(
        cls, class_name: str, chebi_version: int, single_class: int
    ) -> Type[_ChEBIDataExtractor]:
        """
        Dynamically import the ChEBI class.

        Args:
            class_name (str): The name of the ChEBI class.
            chebi_version (int): The version of the ChEBI dataset.
            single_class (int): The ID of the single class to predict.

        Returns:
            _ChEBIDataExtractor: An instance of the dynamically imported class.
        """
        class_name = class_name.strip()
        module = __import__(cls.__MODULE_PATH, fromlist=[class_name])
        _class = getattr(module, class_name)
        return _class(**{"chebi_version": chebi_version, "single_class": single_class})

    def migrate(self) -> None:
        """
        Start the migration process for the ChEBI dataset.
        """
        os.makedirs(self._chebi_cls.base_dir, exist_ok=True)
        print("Migration started.....")
        self._migrate_old_raw_data()

        # Either we can combine `.pt` split files to form `data.pt` file
        # self._migrate_old_processed_data()
        # OR
        # we can transform `data.pkl` to `data.pt` file (this seems efficient along with less code)
        self._chebi_cls.setup_processed()
        print("Migration completed.....")

    def _migrate_old_raw_data(self) -> None:
        """
        Migrate old raw data files to the new data folder structure.
        """
        print("-" * 50)
        print("Migrating old raw Data....")

        self._copy_file(self._old_raw_dir, self._chebi_cls.raw_dir, "chebi.obo")
        self._copy_file(
            self._old_raw_dir, self._chebi_cls.processed_dir_main, "classes.txt"
        )

        old_splits_file_names = {
            "train": "train.pkl",
            "validation": "validation.pkl",
            "test": "test.pkl",
        }
        data_file_path = os.path.join(self._chebi_cls.processed_dir_main, "data.pkl")
        if os.path.isfile(data_file_path):
            print(f"File {data_file_path} already exists in new data-folder structure")
            return

        data_df, split_ass_df = self._combine_pkl_splits(
            self._old_raw_dir, old_splits_file_names
        )

        self._chebi_cls.save_processed(data_df, "data.pkl")
        print(f"File {data_file_path} saved to new data-folder structure")

        split_file = os.path.join(self._chebi_cls.processed_dir_main, "splits.csv")
        split_ass_df.to_csv(split_file)  # overwrites the files with same name
        print(f"File {split_file} saved to new data-folder structure")

    def _migrate_old_processed_data(self) -> None:
        """
        Migrate old processed data files to the new data folder structure.
        """
        print("-" * 50)
        print("Migrating old processed data.....")

        data_file_path = os.path.join(self._chebi_cls.processed_dir, "data.pt")
        if os.path.isfile(data_file_path):
            print(f"File {data_file_path} already exists in new data-folder structure")
            return

        old_splits_file_names = {
            "train": "train.pt",
            "validation": "validation.pt",
            "test": "test.pt",
        }

        data_df = self._combine_pt_splits(
            self._old_processed_dir, old_splits_file_names
        )

        torch.save(data_df, data_file_path)
        print(f"File {data_file_path} saved to new data-folder structure")

    def _combine_pt_splits(
        self, old_dir: str, old_splits_file_names: Dict[str, str]
    ) -> pd.DataFrame:
        """
        Combine old `.pt` split files into a single DataFrame.

        Args:
            old_dir (str): The directory containing the old split files.
            old_splits_file_names (Dict[str, str]): A dictionary of split names and file names.

        Returns:
            pd.DataFrame: The combined DataFrame.
        """
        self._check_if_old_splits_exists(old_dir, old_splits_file_names)

        print("Combining `.pt` splits...")
        df_list: List[pd.DataFrame] = []
        for split, file_name in old_splits_file_names.items():
            file_path = os.path.join(old_dir, file_name)
            file_df = pd.DataFrame(torch.load(file_path))
            df_list.append(file_df)

        return pd.concat(df_list, ignore_index=True)

    def _combine_pkl_splits(
        self, old_dir: str, old_splits_file_names: Dict[str, str]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Combine old `.pkl` split files into a single DataFrame and create split assignments.

        Args:
            old_dir (str): The directory containing the old split files.
            old_splits_file_names (Dict[str, str]): A dictionary of split names and file names.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: The combined DataFrame and split assignments DataFrame.
        """
        self._check_if_old_splits_exists(old_dir, old_splits_file_names)

        df_list: List[pd.DataFrame] = []
        split_assignment_list: List[pd.DataFrame] = []

        print("Combining `.pkl` splits...")
        for split, file_name in old_splits_file_names.items():
            file_path = os.path.join(old_dir, file_name)
            file_df = pd.read_pickle(file_path)
            df_list.append(file_df)

            # Create split assignment for the current DataFrame
            split_assignment = pd.DataFrame({"id": file_df["id"], "split": split})
            split_assignment_list.append(split_assignment)

        # Concatenate all dataframes and split assignments
        combined_df = pd.concat(df_list, ignore_index=True)
        combined_split_assignment = pd.concat(split_assignment_list, ignore_index=True)

        return combined_df, combined_split_assignment

    @staticmethod
    def _check_if_old_splits_exists(
        old_dir: str, old_splits_file_names: Dict[str, str]
    ) -> None:
        """
        Check if the old split files exist in the specified directory.

        Args:
            old_dir (str): The directory containing the old split files.
            old_splits_file_names (Dict[str, str]): A dictionary of split names and file names.

        Raises:
            FileNotFoundError: If any of the split files do not exist.
        """
        if any(
            not os.path.isfile(os.path.join(old_dir, file))
            for file in old_splits_file_names.values()
        ):
            raise FileNotFoundError(
                f"One of the split {old_splits_file_names.values()} doesn't exist "
                f"in old data-folder structure: {old_dir}"
            )

    @staticmethod
    def _copy_file(old_file_dir: str, new_file_dir: str, file_name: str) -> None:
        """
        Copy a file from the old directory to the new directory.

        Args:
            old_file_dir (str): The directory containing the old file.
            new_file_dir (str): The directory to copy the file to.
            file_name (str): The name of the file to copy.

        Raises:
            FileNotFoundError: If the file does not exist in the old directory.
        """
        os.makedirs(new_file_dir, exist_ok=True)
        new_file_path = os.path.join(new_file_dir, file_name)
        if os.path.isfile(new_file_path):
            print(f"File {new_file_path} already exists in new data-folder structure")
            return

        old_file_path = os.path.join(old_file_dir, file_name)
        if not os.path.isfile(old_file_path):
            raise FileNotFoundError(
                f"File {old_file_path} doesn't exist in old data-folder structure"
            )

        shutil.copy2(os.path.abspath(old_file_path), os.path.abspath(new_file_path))
        print(f"Copied from {old_file_path} to {new_file_path}")

    @property
    def _old_base_dir(self) -> str:
        """
        Get the base directory for the old data structure.

        Returns:
            str: The base directory for the old data.
        """
        return os.path.join(
            self.__DATA_ROOT_DIR,
            self._chebi_cls._name,
            f"chebi_v{self._chebi_cls.chebi_version}",
        )

    @property
    def _old_processed_dir(self) -> str:
        """
        Get the processed directory for the old data structure.

        Returns:
            str: The processed directory for the old data.
        """
        res = os.path.join(
            self._old_base_dir,
            "processed",
            *self._chebi_cls.identifier,
        )
        if self._chebi_cls.single_class is None:
            return res
        else:
            return os.path.join(res, f"single_{self._chebi_cls.single_class}")

    @property
    def _old_raw_dir(self) -> str:
        """
        Get the raw directory for the old data structure.

        Returns:
            str: The raw directory for the old data.
        """
        return os.path.join(self._old_base_dir, "raw")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Migrate ChEBI dataset to new structure and handle splits."
    )
    parser.add_argument(
        "--chebi_class",
        type=str,
        required=True,
        help="Chebi class name from the `chebai/preprocessing/datasets/chebi.py`",
    )
    parser.add_argument(
        "--chebi_version", type=int, required=True, help="Chebi data version"
    )
    parser.add_argument(
        "--single_class",
        type=int,
        help="The ID of the single class to predict",
        default=None,
    )
    args = parser.parse_args()

    ChebiDataMigration(
        class_name=args.chebi_class,
        chebi_version=args.chebi_version,
        single_class=args.single_class,
    ).migrate()
