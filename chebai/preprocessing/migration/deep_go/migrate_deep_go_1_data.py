import os
from collections import OrderedDict
from typing import List, Literal, Optional, Tuple

import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from jsonargparse import CLI

from chebai.preprocessing.datasets.deepGO.go_uniprot import DeepGO1MigratedData


class DeepGo1DataMigration:
    """
    A class to handle data migration and processing for the DeepGO project.
    It migrates the DeepGO data to our data structure followed for GO-UniProt data.

    This class handles migration of data from the DeepGO paper below:
        Maxat Kulmanov, Mohammed Asif Khan, Robert Hoehndorf,
        DeepGO: predicting protein functions from sequence and interactions using a deep ontology-aware classifier,
        Bioinformatics, Volume 34, Issue 4, February 2018, Pages 660–668
        (https://doi.org/10.1093/bioinformatics/btx624).
    """

    # Max sequence length as per DeepGO1
    _MAXLEN = 1002
    _LABELS_START_IDX = DeepGO1MigratedData._LABELS_START_IDX

    def __init__(self, data_dir: str, go_branch: Literal["cc", "mf", "bp"]):
        """
        Initializes the data migration object with a data directory and GO branch.

        Args:
            data_dir (str): Directory containing the data files.
            go_branch (Literal["cc", "mf", "bp"]): GO branch to use.
        """
        valid_go_branches = list(DeepGO1MigratedData.GO_BRANCH_MAPPING.keys())
        if go_branch not in valid_go_branches:
            raise ValueError(f"go_branch must be one of {valid_go_branches}")
        self._go_branch = go_branch

        self._data_dir: str = rf"{data_dir}"
        self._train_df: Optional[pd.DataFrame] = None
        self._test_df: Optional[pd.DataFrame] = None
        self._validation_df: Optional[pd.DataFrame] = None
        self._terms_df: Optional[pd.DataFrame] = None
        self._classes: Optional[List[str]] = None

    def migrate(self) -> None:
        """
        Executes the data migration by loading, processing, and saving the data.
        """
        print("Starting the migration process...")
        self._load_data()
        if not all(
            df is not None
            for df in [
                self._train_df,
                self._validation_df,
                self._test_df,
                self._terms_df,
            ]
        ):
            raise Exception(
                "Data splits or terms data is not available in instance variables."
            )
        splits_df = self._record_splits()
        data_with_labels_df = self._extract_required_data_from_splits()

        if not all(
            var is not None for var in [data_with_labels_df, splits_df, self._classes]
        ):
            raise Exception(
                "Data splits or terms data is not available in instance variables."
            )

        self.save_migrated_data(data_with_labels_df, splits_df)

    def _load_data(self) -> None:
        """
        Loads the test, train, validation, and terms data from the pickled files
        in the data directory.
        """
        try:
            print(f"Loading data files from directory: {self._data_dir}")
            self._test_df = pd.DataFrame(
                pd.read_pickle(
                    os.path.join(self._data_dir, f"test-{self._go_branch}.pkl")
                )
            )

            # DeepGO 1 lacks a validation split, so we will create one by further splitting the training set.
            # Although this reduces the training data slightly compared to the original DeepGO setup,
            # given the data size, the impact should be minimal.
            train_df = pd.DataFrame(
                pd.read_pickle(
                    os.path.join(self._data_dir, f"train-{self._go_branch}.pkl")
                )
            )

            self._train_df, self._validation_df = self._get_train_val_split(train_df)

            self._terms_df = pd.DataFrame(
                pd.read_pickle(os.path.join(self._data_dir, f"{self._go_branch}.pkl"))
            )

        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Data file not found in directory: {e}. "
                "Please ensure all required files are available in the specified directory."
            )

    @staticmethod
    def _get_train_val_split(
        train_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits the training data into a smaller training set and a validation set.

        Args:
            train_df (pd.DataFrame): Original training DataFrame.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Training and validation DataFrames.
        """
        labels_list_train = train_df["labels"].tolist()
        train_split = 0.85
        test_size = ((1 - train_split) ** 2) / train_split

        splitter = MultilabelStratifiedShuffleSplit(
            n_splits=1, test_size=test_size, random_state=42
        )

        train_indices, validation_indices = next(
            splitter.split(labels_list_train, labels_list_train)
        )

        df_validation = train_df.iloc[validation_indices]
        df_train = train_df.iloc[train_indices]
        return df_train, df_validation

    def _record_splits(self) -> pd.DataFrame:
        """
        Creates a DataFrame that stores the IDs and their corresponding data splits.

        Returns:
            pd.DataFrame: A combined DataFrame containing split assignments.
        """
        print("Recording data splits for train, validation, and test sets.")
        split_assignment_list: List[pd.DataFrame] = [
            pd.DataFrame({"id": self._train_df["proteins"], "split": "train"}),
            pd.DataFrame(
                {"id": self._validation_df["proteins"], "split": "validation"}
            ),
            pd.DataFrame({"id": self._test_df["proteins"], "split": "test"}),
        ]

        combined_split_assignment = pd.concat(split_assignment_list, ignore_index=True)
        return combined_split_assignment

    def _extract_required_data_from_splits(self) -> pd.DataFrame:
        """
        Extracts required columns from the combined data splits.

        Returns:
            pd.DataFrame: A DataFrame containing the essential columns for processing.
        """
        print("Combining data splits into a single DataFrame with required columns.")
        required_columns = [
            "proteins",
            "accessions",
            "sequences",
            "gos",
            "labels",
        ]

        new_df = pd.concat(
            [
                self._train_df[required_columns],
                self._validation_df[required_columns],
                self._test_df[required_columns],
            ],
            ignore_index=True,
        )
        new_df["go_ids"] = new_df.apply(
            lambda row: self.extract_go_id(row["gos"]), axis=1
        )

        labels_df = self._get_labels_columns(new_df)

        data_df = pd.DataFrame(
            OrderedDict(
                swiss_id=new_df["proteins"],
                accession=new_df["accessions"],
                go_ids=new_df["go_ids"],
                sequence=new_df["sequences"],
            )
        )

        df = pd.concat([data_df, labels_df], axis=1)

        return df

    @staticmethod
    def extract_go_id(go_list: List[str]) -> List[int]:
        """
        Extracts and parses GO IDs from a list of GO annotations.

        Args:
            go_list (List[str]): List of GO annotation strings.

        Returns:
            List[int]: List of parsed GO IDs.
        """
        return [DeepGO1MigratedData._parse_go_id(go_id_str) for go_id_str in go_list]

    def _get_labels_columns(self, data_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates columns for labels based on provided selected terms.

        Args:
            data_df (pd.DataFrame): DataFrame with GO annotations and labels.

        Returns:
            pd.DataFrame: DataFrame with label columns.
        """
        print("Generating label columns from provided selected terms.")
        parsed_go_ids: pd.Series = self._terms_df["functions"].apply(
            lambda gos: DeepGO1MigratedData._parse_go_id(gos)
        )
        all_go_ids_list = parsed_go_ids.values.tolist()
        self._classes = all_go_ids_list

        new_label_columns = pd.DataFrame(
            data_df["labels"].tolist(), index=data_df.index, columns=all_go_ids_list
        )

        return new_label_columns

    def save_migrated_data(
        self, data_df: pd.DataFrame, splits_df: pd.DataFrame
    ) -> None:
        """
        Saves the processed data and split information.

        Args:
            data_df (pd.DataFrame): Data with GO labels.
            splits_df (pd.DataFrame): Split assignment DataFrame.
        """
        print("Saving transformed data files.")

        deepgo_migr_inst: DeepGO1MigratedData = DeepGO1MigratedData(
            go_branch=DeepGO1MigratedData.GO_BRANCH_MAPPING[self._go_branch],
            max_sequence_length=self._MAXLEN,
        )

        # Save data file
        deepgo_migr_inst.save_processed(
            data_df, deepgo_migr_inst.processed_main_file_names_dict["data"]
        )
        print(
            f"{deepgo_migr_inst.processed_main_file_names_dict['data']} saved to {deepgo_migr_inst.processed_dir_main}"
        )

        # Save splits file
        splits_df.to_csv(
            os.path.join(deepgo_migr_inst.processed_dir_main, "splits_deep_go1.csv"),
            index=False,
        )
        print(f"splits_deep_go1.csv saved to {deepgo_migr_inst.processed_dir_main}")

        # Save classes file
        classes = sorted(self._classes)
        with open(
            os.path.join(deepgo_migr_inst.processed_dir_main, "classes_deep_go1.txt"),
            "wt",
        ) as fout:
            fout.writelines(str(node) + "\n" for node in classes)
        print(f"classes_deep_go1.txt saved to {deepgo_migr_inst.processed_dir_main}")

        print("Migration process completed!")


class Main:
    """
    Main class to handle the migration process for DeepGo1DataMigration.

    Methods:
        migrate(data_dir: str, go_branch: Literal["cc", "mf", "bp"]):
            Initiates the migration process for the specified data directory and GO branch.
    """

    @staticmethod
    def migrate(data_dir: str, go_branch: Literal["cc", "mf", "bp"]) -> None:
        """
        Initiates the migration process by creating a DeepGoDataMigration instance
        and invoking its migrate method.

        Args:
            data_dir (str): Directory containing the data files.
            go_branch (Literal["cc", "mf", "bp"]): GO branch to use
                                                  ("cc" for cellular_component,
                                                   "mf" for molecular_function,
                                                   or "bp" for biological_process).
        """
        DeepGo1DataMigration(data_dir, go_branch).migrate()


if __name__ == "__main__":
    # Example:  python script_name.py migrate --data_dir="data/deep_go1" --go_branch="mf"
    # --data_dir specifies the directory containing the data files.
    # --go_branch specifies the GO branch (cc, mf, or bp) you want to use for the migration.
    CLI(
        Main,
        description="DeepGo1DataMigration CLI tool to handle migration of GO data for specified branches (cc, mf, bp).",
        as_positional=False,
    )
