import os
from collections import OrderedDict
from random import randint
from typing import List, Literal

import pandas as pd
from jsonargparse import CLI

from chebai.preprocessing.datasets.go_uniprot import (
    GOUniProtOver50,
    GOUniProtOver250,
    _GOUniProtDataExtractor,
)


class DeepGoDataMigration:
    """
    A class to handle data migration and processing for the DeepGO project. It migrates the data from the DeepGO-SE
    data structure to our data structure followed for GO-UniProt data.

    Attributes:
        _CORRESPONDING_GO_CLASSES (dict): Mapping of GO branches to specific data extractor classes.
        _MAXLEN (int): Maximum sequence length for sequences.
        _LABELS_START_IDX (int): Starting index for labels in the dataset.

    Methods:
        __init__(data_dir, go_branch): Initializes the data directory and GO branch.
        _load_data(): Loads train, validation, test, and terms data from the specified directory.
        _record_splits(): Creates a DataFrame with IDs and their corresponding split.
        migrate(): Executes the migration process including data loading, processing, and saving.
        _extract_required_data_from_splits(): Extracts required columns from the splits data.
        _generate_labels(data_df): Generates label columns for the data based on GO terms.
        extract_go_id(go_list): Extracts GO IDs from a list.
        save_migrated_data(data_df, splits_df): Saves the processed data and splits.
    """

    # Link for the namespaces convention used for GO branch
    # https://github.com/bio-ontology-research-group/deepgo2/blob/main/deepgo/utils.py#L18-L22
    _CORRESPONDING_GO_CLASSES = {
        "cc": GOUniProtOver50,
        "mf": GOUniProtOver50,
        "bp": GOUniProtOver250,
    }

    # https://github.com/bio-ontology-research-group/deepgo2/blob/main/deepgo/aminoacids.py#L11
    _MAXLEN = 1000
    _LABELS_START_IDX = _GOUniProtDataExtractor._LABELS_START_IDX

    def __init__(self, data_dir: str, go_branch: Literal["cc", "mf", "bp"]):
        """
        Initializes the data migration object with a data directory and GO branch.

        Args:
            data_dir (str): Directory containing the data files.
            go_branch (Literal["cc", "mf", "bp"]): GO branch to use (cellular_component, molecular_function, or biological_process).
        """
        valid_go_branches = list(self._CORRESPONDING_GO_CLASSES.keys())
        if go_branch not in valid_go_branches:
            raise ValueError(f"go_branch must be one of {valid_go_branches}")
        self._go_branch = go_branch

        self._data_dir = os.path.join(data_dir, go_branch)
        self._train_df: pd.DataFrame = None
        self._test_df: pd.DataFrame = None
        self._validation_df: pd.DataFrame = None
        self._terms_df: pd.DataFrame = None
        self._classes: List[str] = None

    def _load_data(self) -> None:
        """
        Loads the test, train, validation, and terms data from the pickled files
        in the data directory.
        """
        try:
            print(f"Loading data from {self._data_dir}......")
            self._test_df = pd.DataFrame(
                pd.read_pickle(os.path.join(self._data_dir, "test_data.pkl"))
            )
            self._train_df = pd.DataFrame(
                pd.read_pickle(os.path.join(self._data_dir, "train_data.pkl"))
            )
            self._validation_df = pd.DataFrame(
                pd.read_pickle(os.path.join(self._data_dir, "valid_data.pkl"))
            )
            self._terms_df = pd.DataFrame(
                pd.read_pickle(os.path.join(self._data_dir, "terms.pkl"))
            )
        except FileNotFoundError as e:
            print(f"Error loading data: {e}")

    def _record_splits(self) -> pd.DataFrame:
        """
        Creates a DataFrame that stores the IDs and their corresponding data splits.

        Returns:
            pd.DataFrame: A combined DataFrame containing split assignments.
        """
        print("Recording splits...")
        split_assignment_list: List[pd.DataFrame] = [
            pd.DataFrame({"id": self._train_df["proteins"], "split": "train"}),
            pd.DataFrame(
                {"id": self._validation_df["proteins"], "split": "validation"}
            ),
            pd.DataFrame({"id": self._test_df["proteins"], "split": "test"}),
        ]

        combined_split_assignment = pd.concat(split_assignment_list, ignore_index=True)
        return combined_split_assignment

    def migrate(self) -> None:
        """
        Executes the data migration by loading, processing, and saving the data.
        """
        print("Migration started......")
        self._load_data()
        if not all(
            [self._train_df, self._validation_df, self._test_df, self._terms_df]
        ):
            raise Exception(
                "Data splits or terms data is not available in instance variables."
            )
        splits_df = self._record_splits()

        data_df = self._extract_required_data_from_splits()
        data_with_labels_df = self._generate_labels(data_df)

        if not all([data_with_labels_df, splits_df, self._classes]):
            raise Exception(
                "Data splits or terms data is not available in instance variables."
            )

        self.save_migrated_data(data_df, splits_df)

    def _extract_required_data_from_splits(self) -> pd.DataFrame:
        """
        Extracts required columns from the combined data splits.

        Returns:
            pd.DataFrame: A DataFrame containing the essential columns for processing.
        """
        print("Combining the data splits with required data..... ")
        required_columns = [
            "proteins",
            "accessions",
            "sequences",
            # https://github.com/bio-ontology-research-group/deepgo2/blob/main/gendata/uni2pandas.py#L45-L58
            "exp_annotations",  # Directly associated GO ids
            # https://github.com/bio-ontology-research-group/deepgo2/blob/main/gendata/uni2pandas.py#L60-L69
            "prop_annotations",  # Transitively associated GO ids
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
            lambda row: self.extract_go_id(row["exp_annotations"])
            + self.extract_go_id(row["prop_annotations"]),
            axis=1,
        )

        data_df = pd.DataFrame(
            OrderedDict(
                swiss_id=new_df["proteins"],
                accession=new_df["accessions"],
                go_ids=new_df["go_ids"],
                sequence=new_df["sequences"],
            )
        )
        return data_df

    def _generate_labels(self, data_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates label columns for each GO term in the dataset.

        Args:
            data_df (pd.DataFrame): DataFrame containing data with GO IDs.

        Returns:
            pd.DataFrame: DataFrame with new label columns.
        """
        print("Generating labels based on terms.pkl file.......")
        parsed_go_ids: pd.Series = self._terms_df.apply(
            lambda row: self.extract_go_id(row["gos"])
        )
        all_go_ids_list = parsed_go_ids.values.tolist()
        self._classes = all_go_ids_list
        new_label_columns = pd.DataFrame(
            False, index=data_df.index, columns=all_go_ids_list
        )
        data_df = pd.concat([data_df, new_label_columns], axis=1)

        for index, row in data_df.iterrows():
            for go_id in row["go_ids"]:
                if go_id in data_df.columns:
                    data_df.at[index, go_id] = True

        data_df = data_df[data_df.iloc[:, self._LABELS_START_IDX :].any(axis=1)]
        return data_df

    @staticmethod
    def extract_go_id(go_list: List[str]) -> List[str]:
        """
        Extracts and parses GO IDs from a list of GO annotations.

        Args:
            go_list (List[str]): List of GO annotation strings.

        Returns:
            List[str]: List of parsed GO IDs.
        """
        return [
            _GOUniProtDataExtractor._parse_go_id(go_id_str) for go_id_str in go_list
        ]

    def save_migrated_data(
        self, data_df: pd.DataFrame, splits_df: pd.DataFrame
    ) -> None:
        """
        Saves the processed data and split information.

        Args:
            data_df (pd.DataFrame): Data with GO labels.
            splits_df (pd.DataFrame): Split assignment DataFrame.
        """
        print("Saving transformed data......")
        go_class_instance: _GOUniProtDataExtractor = self._CORRESPONDING_GO_CLASSES[
            self._go_branch
        ](go_branch=self._go_branch, max_sequence_length=self._MAXLEN)

        go_class_instance.save_processed(
            data_df, go_class_instance.processed_file_names_dict["data"]
        )
        print(
            f"{go_class_instance.processed_file_names_dict['data']} saved to {go_class_instance.processed_dir_main}"
        )

        splits_df.to_csv(
            os.path.join(go_class_instance.processed_dir_main, "splits.csv"),
            index=False,
        )
        print(f"splits.csv saved to {go_class_instance.processed_dir_main}")

        classes = sorted(self._classes)
        with open(
            os.path.join(go_class_instance.processed_dir_main, "classes.txt"), "wt"
        ) as fout:
            fout.writelines(str(node) + "\n" for node in classes)
        print(f"classes.txt saved to {go_class_instance.processed_dir_main}")
        print("Migration completed!")


class Main:
    """
    Main class to handle the migration process for DeepGoDataMigration.

    Methods:
        migrate(data_dir: str, go_branch: Literal["cc", "mf", "bp"]):
            Initiates the migration process for the specified data directory and GO branch.
    """

    def migrate(self, data_dir: str, go_branch: str) -> None:
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
        DeepGoDataMigration(data_dir, go_branch).migrate()


class Main1:
    def __init__(self, max_prize: int = 100):
        """
        Args:
            max_prize: Maximum prize that can be awarded.
        """
        self.max_prize = max_prize

    def person(self, name: str, additional_prize: int = 0):
        """
        Args:
            name: Name of the winner.
            additional_prize: Additional prize that can be added to the prize amount.
        """
        prize = randint(0, self.max_prize) + additional_prize
        return f"{name} won {prize}€!"


if __name__ == "__main__":
    # Example:  python script_name.py migrate data_dir="data/deep_go_se_training_data" go_branch="bp"
    # --data_dir specifies the directory containing the data files.
    # --go_branch specifies the GO branch (cc, mf, or bp) you want to use for the migration.
    CLI(
        Main1,
        description="DeepGoDataMigration CLI tool to handle migration of GO data for specified branches (cc, mf, bp).",
    )
