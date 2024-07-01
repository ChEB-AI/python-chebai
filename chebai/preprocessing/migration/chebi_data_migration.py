import argparse
import os
import shutil
from typing import Dict, List, Tuple, Type

import pandas as pd
import torch

from chebai.preprocessing.datasets.chebi import _ChEBIDataExtractor


class ChebiDataMigration:
    __MODULE_PATH: str = "chebai.preprocessing.datasets.chebi"
    __DATA_ROOT_DIR: str = "data"

    def __init__(self, chebi_version, class_name: str):
        self._chebi_version: int = chebi_version
        # Chebi class instance according to new data structure
        self._chebi_cls: Type[_ChEBIDataExtractor] = self._dynamic_import_chebi_cls(
            class_name, chebi_version
        )
        self._class_path: str = class_name

    def _get_old_dir_structure(self):
        base_dir = os.path.join(
            self.__DATA_ROOT_DIR,
            self._chebi_cls._name,
            f"chebi_v{self._chebi_cls.chebi_version}",
        )

    @classmethod
    def _dynamic_import_chebi_cls(cls, class_name: str, chebi_version: int):
        class_name = class_name.strip()
        module = __import__(cls.__MODULE_PATH, fromlist=[class_name])
        _class = getattr(module, class_name)
        return _class({"chebi_version": chebi_version})

    def migrate(self):
        os.makedirs(self._chebi_cls.base_dir, exist_ok=True)
        self._migrate_old_processed_data()

    def _migrate_old_raw_data(self):
        self._copy_file(self._old_raw_dir, self._chebi_cls.raw_dir, "chebi.obo")
        self._copy_file(
            self._old_raw_dir, self._chebi_cls.processed_dir_main, "classes.txt"
        )
        old_splits_file_names = {
            "train": "train.pkl",
            "validation": "validation.pkl",
            "test": "test.pkl",
        }
        data_df, split_ass_df = self._combine_splits(
            self._old_raw_dir, old_splits_file_names
        )
        data_df.to_pickle(os.path.join(self._chebi_cls.processed_dir_main, "data.pkl"))
        split_ass_df.to_csv(
            os.path.join(self._chebi_cls.processed_dir_main, "splits.csv")
        )

    def _migrate_old_processed_data(self):
        old_splits_file_names = {
            "train": "train.pt",
            "validation": "validation.pt",
            "test": "test.pt",
        }

        data_df = self._combine_pt_splits(
            self._old_processed_dir, old_splits_file_names
        )
        torch.save(data_df, self._chebi_cls.processed_dir)

    def _combine_pt_splits(
        self, old_dir: str, old_splits_file_names: Dict[str, str]
    ) -> pd.DataFrame:
        self._check_if_old_splits_exists(old_dir, old_splits_file_names)

        df_list: List[pd.DataFrame] = []
        for split, file_name in old_splits_file_names.items():
            file_path = os.path.join(old_dir, file_name)
            file_df = pd.DataFrame(torch.load(file_path))
            df_list.append(file_df)

        return pd.concat(df_list, ignore_index=True)

    def _combine_pkl_splits(
        self, old_dir: str, old_splits_file_names: Dict[str, str]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        self._check_if_old_splits_exists(old_dir, old_splits_file_names)

        df_list: List[pd.DataFrame] = []
        split_assignment_list: List[pd.DataFrame] = []

        for split, file_name in old_splits_file_names.items():
            file_path = os.path.join(old_dir, file_name)
            file_df = pd.DataFrame(self._chebi_cls._load_data_from_file(file_path))
            file_df["split"] = split  # Assign the split label to the DataFrame
            df_list.append(file_df)

            # Create split assignment for the current DataFrame
            split_assignment = pd.DataFrame({"id": file_df["id"], "split": split})
            split_assignment_list.append(split_assignment)

        # Concatenate all dataframes and split assignments
        combined_df = pd.concat(df_list, ignore_index=True)
        combined_split_assignment = pd.concat(split_assignment_list, ignore_index=True)

        return combined_df, combined_split_assignment

    @staticmethod
    def _check_if_old_splits_exists(old_dir, old_splits_file_names):
        if any(
            not os.path.isfile(os.path.join(old_dir, file))
            for file in old_splits_file_names.values()
        ):
            raise FileNotFoundError(
                f"One of the split {old_splits_file_names.values()} doesn't exists "
                f"in old data-folder structure: {old_dir}"
            )

    @staticmethod
    def _copy_file(old_file_dir, new_file_dir, file_name):
        os.makedirs(new_file_dir, exist_ok=True)
        new_file_path = os.path.join(new_file_dir, file_name)
        if os.path.isfile(new_file_path):
            print(f"File {new_file_path} already exists in new data-folder structure")
            return

        old_file_path = os.path.join(old_file_dir, file_name)
        if not os.path.isfile(old_file_path):
            raise FileNotFoundError(
                f"File {old_file_path} doesn't exists in old data-folder structure"
            )

        shutil.copy2(os.path.abspath(old_file_path), os.path.abspath(new_file_path))
        print(f"Copied from {old_file_path} to {new_file_path}")

    @property
    def _old_base_dir(self):
        return os.path.join(
            "data", self._chebi_cls._name, f"chebi_v{self._chebi_cls.chebi_version}"
        )

    @property
    def _old_processed_dir(self):
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
    def _old_raw_dir(self):
        """name of dir where the raw data is stored"""
        return os.path.join(self._old_base_dir, "raw")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Migrate ChEBI dataset to new structure and handle splits."
    )
    parser.add_argument(
        "old_directory", type=str, help="Path to the old directory structure"
    )
    parser.add_argument(
        "new_directory", type=str, help="Path to the new directory structure"
    )
    parser.add_argument(
        "--split_file_path",
        type=str,
        help="Path to the CSV file with split configuration",
        default=None,
    )
    parser.add_argument("chebi_version", type=int, help="Data Version related to chebi")
    args = parser.parse_args()

    # main(args.old_directory, args.new_directory, args.split_file_path)

# python migration_script.py path/to/old_directory path/to/new_directory --split_file_path path/to/split_configuration.csv
# python migration_script.py path/to/old_directory path/to/new_directory
