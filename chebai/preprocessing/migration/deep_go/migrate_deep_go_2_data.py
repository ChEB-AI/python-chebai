import os
import re
from collections import OrderedDict
from typing import List, Literal, Optional

import pandas as pd
from jsonargparse import CLI

from chebai.preprocessing.datasets.deepGO.go_uniprot import DeepGO2MigratedData
from chebai.preprocessing.reader import ProteinDataReader


class DeepGo2DataMigration:
    """
    A class to handle data migration and processing for the DeepGO project. It migrates the data from the DeepGO-SE
    data structure to our data structure followed for GO-UniProt data.

    This class handles migration of data from the DeepGO paper below:
        Maxat Kulmanov, Mohammed Asif Khan, Robert Hoehndorf,
        DeepGO: predicting protein functions from sequence and interactions using a deep ontology-aware classifier,
        Bioinformatics, Volume 34, Issue 4, February 2018, Pages 660–668
        (https://doi.org/10.1093/bioinformatics/btx624)
    """

    _LABELS_START_IDX = DeepGO2MigratedData._LABELS_START_IDX

    def __init__(
        self, data_dir: str, go_branch: Literal["cc", "mf", "bp"], max_len: int = 1000
    ):
        """
        Initializes the data migration object with a data directory and GO branch.

        Args:
            data_dir (str): Directory containing the data files.
            go_branch (Literal["cc", "mf", "bp"]): GO branch to use.
            max_len (int): Used to truncate the sequence to this length. Default is 1000.
                # https://github.com/bio-ontology-research-group/deepgo2/blob/main/deepgo/aminoacids.py#L11
        """
        valid_go_branches = list(DeepGO2MigratedData.GO_BRANCH_MAPPING.keys())
        if go_branch not in valid_go_branches:
            raise ValueError(f"go_branch must be one of {valid_go_branches}")
        self._go_branch = go_branch

        self._data_dir: str = os.path.join(rf"{data_dir}", go_branch)
        self._max_len: int = max_len

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

        data_df = self._extract_required_data_from_splits()
        data_with_labels_df = self._generate_labels(data_df)

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
            print(f"Loading data from directory: {self._data_dir}......")

            print(
                "Pre-processing the data before loading them into instance variables\n"
                f"2-Steps preprocessing: \n"
                f"\t 1: Truncating every sequence to {self._max_len}\n"
                f"\t 2: Replacing every amino acid which is not in {ProteinDataReader.AA_LETTER}"
            )

            self._test_df = self._pre_process_data(
                pd.DataFrame(
                    pd.read_pickle(os.path.join(self._data_dir, "test_data.pkl"))
                )
            )
            self._train_df = self._pre_process_data(
                pd.DataFrame(
                    pd.read_pickle(os.path.join(self._data_dir, "train_data.pkl"))
                )
            )
            self._validation_df = self._pre_process_data(
                pd.DataFrame(
                    pd.read_pickle(os.path.join(self._data_dir, "valid_data.pkl"))
                )
            )

            self._terms_df = pd.DataFrame(
                pd.read_pickle(os.path.join(self._data_dir, "terms.pkl"))
            )

        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Data file not found in directory: {e}. "
                "Please ensure all required files are available in the specified directory."
            )

    def _pre_process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Pre-processes the input dataframe by truncating sequences to the maximum
        length and replacing invalid amino acids with 'X'.

        Args:
            df (pd.DataFrame): The dataframe to preprocess.

        Returns:
            pd.DataFrame: The processed dataframe.
        """
        df = self._truncate_sequences(df)
        df = self._replace_invalid_amino_acids(df)
        return df

    def _truncate_sequences(
        self, df: pd.DataFrame, column: str = "sequences"
    ) -> pd.DataFrame:
        """
        Truncate sequences in a specified column of a dataframe to the maximum length.

        https://github.com/bio-ontology-research-group/deepgo2/blob/main/train_cnn.py#L206-L217

        Args:
            df (pd.DataFrame): The input dataframe containing the data to be processed.
            column (str, optional): The column containing sequences to truncate.
                Defaults to "sequences".

        Returns:
            pd.DataFrame: The dataframe with sequences truncated to `self._max_len`.
        """
        df[column] = df[column].apply(lambda x: x[: self._max_len])
        return df

    @staticmethod
    def _replace_invalid_amino_acids(
        df: pd.DataFrame, column: str = "sequences"
    ) -> pd.DataFrame:
        """
        Replaces invalid amino acids in a sequence with 'X' using regex.

        https://github.com/bio-ontology-research-group/deepgo2/blob/main/deepgo/aminoacids.py#L26-L33
        https://github.com/ChEB-AI/python-chebai/pull/64#issuecomment-2517067073

        Args:
            df (pd.DataFrame): The dataframe containing the sequences to be processed.
            column (str, optional): The column containing the sequences. Defaults to "sequences".

        Returns:
            pd.DataFrame: The dataframe with invalid amino acids replaced by 'X'.
        """
        valid_amino_acids = "".join(ProteinDataReader.AA_LETTER)
        # Replace any character not in the valid set with 'X'
        df[column] = df[column].apply(
            lambda x: re.sub(f"[^{valid_amino_acids}]", "X", x)
        )
        return df

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
        print("Combining the data splits with required data..... ")
        required_columns = [
            "proteins",
            "accessions",
            "sequences",
            # https://github.com/bio-ontology-research-group/deepgo2/blob/main/gendata/uni2pandas.py#L60-L69
            "prop_annotations",  # Direct and Transitively associated GO ids
            "esm2",
        ]

        new_df = pd.concat(
            [
                self._train_df[required_columns],
                self._validation_df[required_columns],
                self._test_df[required_columns],
            ],
            ignore_index=True,
        )
        new_df["go_ids"] = new_df["prop_annotations"].apply(
            lambda x: self.extract_go_id(x)
        )

        data_df = pd.DataFrame(
            OrderedDict(
                swiss_id=new_df["proteins"],
                accession=new_df["accessions"],
                go_ids=new_df["go_ids"],
                sequence=new_df["sequences"],
                esm2_embeddings=new_df["esm2"],
            )
        )
        return data_df

    @staticmethod
    def extract_go_id(go_list: List[str]) -> List[int]:
        """
        Extracts and parses GO IDs from a list of GO annotations.

        Args:
            go_list (List[str]): List of GO annotation strings.

        Returns:
            List[str]: List of parsed GO IDs.
        """
        return [DeepGO2MigratedData._parse_go_id(go_id_str) for go_id_str in go_list]

    def _generate_labels(self, data_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates label columns for each GO term in the dataset.

        Args:
            data_df (pd.DataFrame): DataFrame containing data with GO IDs.

        Returns:
            pd.DataFrame: DataFrame with new label columns.
        """
        print("Generating labels based on terms.pkl file.......")
        parsed_go_ids: pd.Series = self._terms_df["gos"].apply(
            DeepGO2MigratedData._parse_go_id
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
        deepgo_migr_inst: DeepGO2MigratedData = DeepGO2MigratedData(
            go_branch=DeepGO2MigratedData.GO_BRANCH_MAPPING[self._go_branch],
            max_sequence_length=self._max_len,
        )

        # Save data file
        deepgo_migr_inst.save_processed(
            data_df, deepgo_migr_inst.processed_main_file_names_dict["data"]
        )
        print(
            f"{deepgo_migr_inst.processed_main_file_names_dict['data']} saved to {deepgo_migr_inst.processed_dir_main}"
        )

        # Save split file
        splits_df.to_csv(
            os.path.join(deepgo_migr_inst.processed_dir_main, "splits_deep_go2.csv"),
            index=False,
        )
        print(f"splits_deep_go2.csv saved to {deepgo_migr_inst.processed_dir_main}")

        # Save classes.txt file
        classes = sorted(self._classes)
        with open(
            os.path.join(deepgo_migr_inst.processed_dir_main, "classes_deep_go2.txt"),
            "wt",
        ) as fout:
            fout.writelines(str(node) + "\n" for node in classes)
        print(f"classes_deep_go2.txt saved to {deepgo_migr_inst.processed_dir_main}")

        print("Migration completed!")


class Main:
    """
    Main class to handle the migration process for DeepGoDataMigration.

    Methods:
        migrate(data_dir: str, go_branch: Literal["cc", "mf", "bp"]):
            Initiates the migration process for the specified data directory and GO branch.
    """

    @staticmethod
    def migrate(
        data_dir: str, go_branch: Literal["cc", "mf", "bp"], max_len: int = 1000
    ) -> None:
        """
        Initiates the migration process by creating a DeepGoDataMigration instance
        and invoking its migrate method.

        Args:
            data_dir (str): Directory containing the data files.
            go_branch (Literal["cc", "mf", "bp"]): GO branch to use
                                                  ("cc" for cellular_component,
                                                   "mf" for molecular_function,
                                                   or "bp" for biological_process).
            max_len (int): Used to truncate the sequence to this length. Default is 1000.
                # https://github.com/bio-ontology-research-group/deepgo2/blob/main/deepgo/aminoacids.py#L11
        """
        DeepGo2DataMigration(data_dir, go_branch, max_len).migrate()


if __name__ == "__main__":
    # Example:  python script_name.py migrate --data_dir="data/deep_go_se_training_data" --go_branch="bp"
    # --data_dir specifies the directory containing the data files.
    # --go_branch specifies the GO branch (cc, mf, or bp) you want to use for the migration.
    CLI(
        Main,
        description="DeepGoDataMigration CLI tool to handle migration of GO data for specified branches (cc, mf, bp).",
        as_positional=False,
    )
