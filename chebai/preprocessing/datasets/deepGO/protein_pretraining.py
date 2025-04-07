__all__ = ["SwissProteinPretrain"]

import os
from abc import ABC
from collections import OrderedDict
from typing import Any, Dict, Generator, List, Tuple

import networkx as nx
import pandas as pd
import torch
from Bio import SwissProt
from sklearn.model_selection import train_test_split

from chebai.preprocessing.datasets.base import _DynamicDataset
from chebai.preprocessing.datasets.deepGO.go_uniprot import (
    AMBIGUOUS_AMINO_ACIDS,
    EXPERIMENTAL_EVIDENCE_CODES,
    GOUniProtOver250,
)
from chebai.preprocessing.reader import ProteinDataReader


class _ProteinPretrainingData(_DynamicDataset, ABC):
    """
    Data module for pretraining protein sequences, specifically designed for Swiss-UniProt data. It includes methods for
    data preparation, loading, and dynamic splitting of protein sequences.
    The data is parsed and filtered to only select proteins with no associated `valid` Gene Ontology (GO) labels.
    A valid GO label is the one which has one of evidence codes defined in `EXPERIMENTAL_EVIDENCE_CODES`.
    """

    _ID_IDX: int = 0
    _DATA_REPRESENTATION_IDX: int = 1  # Index of `sequence` column

    def __init__(self, **kwargs):
        """
        Initializes the data module with any GOUniProt extractor class object.

        Args:
            **kwargs: Additional arguments for the superclass initialization.
        """
        self._go_uniprot_extractor = GOUniProtOver250()
        assert self._go_uniprot_extractor.go_branch == GOUniProtOver250._ALL_GO_BRANCHES

        self.max_sequence_length: int = int(kwargs.get("max_sequence_length", 1002))
        assert (
            self.max_sequence_length >= 1
        ), "Max sequence length should be greater than or equal to 1."

        super(_ProteinPretrainingData, self).__init__(**kwargs)

        if self.reader.n_gram is not None:
            assert self.max_sequence_length >= self.reader.n_gram, (
                f"max_sequence_length ({self.max_sequence_length}) must be greater than "
                f"or equal to n_gram ({self.reader.n_gram})."
            )

    # ------------------------------ Phase: Prepare data -----------------------------------
    def prepare_data(self, *args: Any, **kwargs: Any) -> None:
        """
        Prepares the data by downloading and parsing Swiss-Prot data if not already available. Saves the processed data
        for further use.

        Args:
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        processed_name = self.processed_main_file_names_dict["data"]
        if not os.path.isfile(os.path.join(self.processed_dir_main, processed_name)):
            print("Missing processed data file (`data.pkl` file)")
            os.makedirs(self.processed_dir_main, exist_ok=True)
            self._download_required_data()
            protein_df = self._parse_protein_data_for_pretraining()
            self.save_processed(protein_df, processed_name)

    def _extract_class_hierarchy(self, data_path: str) -> nx.DiGraph:
        # method not required as no Swiss-UniProt has no ontological data
        pass

    def _graph_to_raw_dataset(self, graph: nx.DiGraph) -> pd.DataFrame:
        # method not required as no Swiss-UniProt has no ontological data
        pass

    def select_classes(self, g: nx.DiGraph, *args, **kwargs) -> List:
        # method not required as no Swiss-UniProt has no ontological data
        pass

    def _download_required_data(self) -> str:
        """
        Downloads the required Swiss-Prot data using the GOUniProt extractor class.

        Returns:
            str: Path to the downloaded data.
        """
        return self._go_uniprot_extractor._download_swiss_uni_prot_data()

    def _parse_protein_data_for_pretraining(self) -> pd.DataFrame:
        """
        Parses the Swiss-Prot data and returns a DataFrame containing Swiss-Prot proteins which does not have any valid
        Gene Ontology(GO) label. A valid GO label is the one which has one of the following evidence codes, as specified in
        `EXPERIMENTAL_EVIDENCE_CODES`.

        The DataFrame includes the following columns:
            - "swiss_id": The unique identifier for each Swiss-Prot record.
            - "sequence": The protein sequence.

        Note:
            We ignore proteins with ambiguous amino acid specified in `AMBIGUOUS_AMINO_ACIDS` in their sequence.`

        Returns:
            pd.DataFrame: A DataFrame where each row corresponds to a Swiss-Prot record with not associated valid GO.
        """
        print("Parsing swiss uniprot raw data....")

        swiss_ids, sequences = [], []

        swiss_data = SwissProt.parse(
            open(
                os.path.join(
                    self._go_uniprot_extractor.raw_dir,
                    self._go_uniprot_extractor.raw_file_names_dict["SwissUniProt"],
                ),
                "r",
            )
        )

        for record in swiss_data:
            if record.data_class != "Reviewed":
                # To consider only manually-annotated swiss data
                continue

            if not record.sequence:
                # Consider protein with only sequence representation
                continue

            if len(record.sequence) > self.max_sequence_length:
                # Consider protein with only sequence length not greater than max seq. length
                continue

            if any(aa in AMBIGUOUS_AMINO_ACIDS for aa in record.sequence):
                # Skip proteins with ambiguous amino acid codes
                continue

            has_valid_associated_go_label = False
            for cross_ref in record.cross_references:
                if cross_ref[0] == self._go_uniprot_extractor._GO_DATA_INIT:

                    if len(cross_ref) <= 3:
                        # No evidence code
                        continue

                    # https://github.com/bio-ontology-research-group/deepgo/blob/master/get_functions.py#L63-L66
                    evidence_code = cross_ref[3].split(":")[0]
                    if evidence_code in EXPERIMENTAL_EVIDENCE_CODES:
                        has_valid_associated_go_label = True
                        break

            if has_valid_associated_go_label:
                # Skip proteins which has at least one associated go label
                continue

            swiss_ids.append(record.entry_name)
            sequences.append(record.sequence)

        data_dict = OrderedDict(
            swiss_id=swiss_ids,  # swiss_id column at index 0
            sequence=sequences,  # Sequence column at index 1
        )

        return pd.DataFrame(data_dict)

    # ------------------------------ Phase: Setup data -----------------------------------
    def _load_dict(self, input_file_path: str) -> Generator[Dict[str, Any], None, None]:
        """
        Loads data from a pickled file and yields individual dictionaries for each row.

        The pickled file is expected to contain rows with the following structure:
            - Data at row index `self._ID_IDX`: ID of go data instance
            - Data at row index `self._DATA_REPRESENTATION_IDX`: Sequence representation of protein

        This method is used by `_load_data_from_file` to generate dictionaries that are then
        processed and converted into a list of dictionaries containing the features and labels.

        Args:
            input_file_path (str): The path to the pickled input file.

        Yields:
            Dict[str, Any]: A dictionary containing:
                - `features` (str): The sequence data from the file.
                - `ident` (Any): The identifier from row index 0.
                - `labels`: Set to None
        """
        with open(input_file_path, "rb") as input_file:
            df = pd.read_pickle(input_file)
            for row in df.values:
                yield dict(
                    features=row[self._DATA_REPRESENTATION_IDX],
                    ident=row[self._ID_IDX],
                    labels=None,
                )

    # ------------------------------ Phase: Dynamic Splits -----------------------------------
    def _get_data_splits(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Loads encoded data and generates training, validation, and test splits.

        This method attempts to load encoded data from a file named `data.pt`. It then splits this data into
        training, validation, and test sets.

        Raises:
            FileNotFoundError: If the `data.pt` file does not exist. Ensure that `prepare_data` and/or
            `setup` methods are called to generate the necessary dataset files.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple containing three DataFrames:
                - Training set
                - Validation set
                - Test set
        """
        try:
            filename = self.processed_file_names_dict["data"]
            data_go = torch.load(
                os.path.join(self.processed_dir, filename), weights_only=False
            )
        except FileNotFoundError:
            raise FileNotFoundError(
                f"File data.pt doesn't exists. "
                f"Please call 'prepare_data' and/or 'setup' methods to generate the dataset files"
            )

        df_go_data = pd.DataFrame(data_go)
        train_df_go, df_test = train_test_split(
            df_go_data,
            train_size=self.train_split,
            random_state=self.dynamic_data_split_seed,
        )

        # Get all splits
        df_train, df_val = train_test_split(
            train_df_go,
            train_size=self.train_split,
            random_state=self.dynamic_data_split_seed,
        )

        return df_train, df_val, df_test

    # ------------------------------ Phase: Raw Properties -----------------------------------
    @property
    def base_dir(self) -> str:
        """
        str: The base directory for pretraining data storage.
        """
        return os.path.join(self._go_uniprot_extractor.base_dir, "Pretraining")

    @property
    def raw_dir(self) -> str:
        """Name of the directory where the raw data is stored."""
        return self._go_uniprot_extractor.raw_dir


class SwissProteinPretrain(_ProteinPretrainingData):
    """
    Data module for Swiss-Prot protein pretraining, inheriting from `_ProteinPretrainingData`.
    This class is specifically designed to handle data processing and loading for Swiss-Prot-based protein datasets.

    Attributes:
        READER (Type): The data reader class used to load and process protein pretraining data.
    """

    READER = ProteinDataReader

    @property
    def _name(self) -> str:
        """
        The name identifier for this data module.

        Returns:
            str: A string identifier, "SwissProteinPretrain", representing the name of this data module.
        """
        return f"Swiss_{self.max_sequence_length}"
