# References for this file :
# Reference 1:
# Maxat Kulmanov, Mohammed Asif Khan, Robert Hoehndorf;
# DeepGO: Predicting protein functions from sequence and interactions
# using a deep ontology-aware classifier, Bioinformatics, 2017.
# https://doi.org/10.1093/bioinformatics/btx624
# Github: https://github.com/bio-ontology-research-group/deepgo

# Reference 2:
# https://www.ebi.ac.uk/GOA/downloads
# https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/docs/keywlist.txt
# https://www.uniprot.org/uniprotkb

# Reference 3:
# Kulmanov, M., Guzmán-Vega, F.J., Duek Roggli,
# P. et al. Protein function prediction as approximate semantic entailment. Nat Mach Intell 6, 220–228 (2024).
# https://doi.org/10.1038/s42256-024-00795-w
# https://github.com/bio-ontology-research-group/deepgo2

__all__ = [
    "GOUniProtOver250",
    "GOUniProtOver50",
    "EXPERIMENTAL_EVIDENCE_CODES",
    "AMBIGUOUS_AMINO_ACIDS",
    "DeepGO1MigratedData",
    "DeepGO2MigratedData",
]

import gzip
import itertools
import os
import shutil
from abc import ABC, abstractmethod
from collections import OrderedDict
from tempfile import NamedTemporaryFile
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import fastobo
import networkx as nx
import pandas as pd
import requests
import torch
import tqdm
from Bio import SwissProt

from chebai.preprocessing import reader as dr
from chebai.preprocessing.datasets.base import _DynamicDataset

# https://github.com/bio-ontology-research-group/deepgo/blob/master/utils.py#L15
EXPERIMENTAL_EVIDENCE_CODES = {
    "EXP",
    "IDA",
    "IPI",
    "IMP",
    "IGI",
    "IEP",
    "TAS",
    "IC",
    # New evidence codes added in latest paper year 2024 Reference number 3
    # https://github.com/bio-ontology-research-group/deepgo2/blob/main/deepgo/utils.py#L24-L26
    "HTP",
    "HDA",
    "HMP",
    "HGI",
    "HEP",
}

# https://github.com/bio-ontology-research-group/deepgo/blob/d97447a05c108127fee97982fd2c57929b2cf7eb/aaindex.py#L8
# https://github.com/bio-ontology-research-group/deepgo2/blob/main/deepgo/aminoacids.py#L10
# `X` is now considered as valid amino acid, as per latest paper year 2024 Refernce number 3
AMBIGUOUS_AMINO_ACIDS = {"B", "O", "J", "U", "Z", "*"}


class _GOUniProtDataExtractor(_DynamicDataset, ABC):
    """
    A class for extracting and processing data from the Gene Ontology (GO) dataset and the Swiss UniProt dataset.

    Args:
        dynamic_data_split_seed (int, optional): The seed for random data splitting. Defaults to 42.
        splits_file_path (str, optional): Path to the splits CSV file. Defaults to None.
        max_sequence_length (int, optional): Specifies the maximum allowed sequence length for a protein, with a
        default of 1002. During data preprocessing, any proteins exceeding this length will be excluded from further
        processing.
        **kwargs: Additional keyword arguments passed to DynamicDataset and  XYBaseDataModule.

    Attributes:
        dynamic_data_split_seed (int): The seed for random data splitting, default is 42.
        max_sequence_length (int, optional): Specifies the maximum allowed sequence length for a protein, with a
        default of 1002. During data preprocessing, any proteins exceeding this length will be excluded from further
        processing.
        splits_file_path (Optional[str]): Path to the CSV file containing split assignments.
    """

    _GO_DATA_INIT = "GO"
    _SWISS_DATA_INIT = "SWISS"

    # -- Index for columns of processed `data.pkl` (derived from `_get_swiss_to_go_mapping` & `_graph_to_raw_dataset`
    # "swiss_id" at           row index 0
    # "accession" at          row index 1
    # "go_ids" at             row index 2
    # "sequence" at           row index 3
    # labels starting from    row index 4
    _ID_IDX: int = 0
    _DATA_REPRESENTATION_IDX: int = 3  # here `sequence` column
    _LABELS_START_IDX: int = 4

    _GO_DATA_URL: str = "https://purl.obolibrary.org/obo/go/go-basic.obo"
    _SWISS_DATA_URL: str = (
        "https://ftp.uniprot.org/pub/databases/uniprot/knowledgebase/complete/uniprot_sprot.dat.gz"
    )

    # Gene Ontology (GO) has three major branches, one for biological processes (BP), molecular functions (MF) and
    # cellular components (CC). The value "all" will take data related to all three branches into account.
    _ALL_GO_BRANCHES: str = "all"
    _GO_BRANCH_NAMESPACE: Dict[str, str] = {
        "BP": "biological_process",
        "MF": "molecular_function",
        "CC": "cellular_component",
    }

    def __init__(self, **kwargs):
        self.go_branch: str = self._get_go_branch(**kwargs)

        self.max_sequence_length: int = int(kwargs.get("max_sequence_length", 1002))
        assert (
            self.max_sequence_length >= 1
        ), "Max sequence length should be greater than or equal to 1."

        super(_GOUniProtDataExtractor, self).__init__(**kwargs)

        if self.reader.n_gram is not None:
            assert self.max_sequence_length >= self.reader.n_gram, (
                f"max_sequence_length ({self.max_sequence_length}) must be greater than "
                f"or equal to n_gram ({self.reader.n_gram})."
            )

    @classmethod
    def _get_go_branch(cls, **kwargs) -> str:
        """
        Retrieves the Gene Ontology (GO) branch based on provided keyword arguments.
        This method checks if a valid GO branch value is provided in the keyword arguments.

        Args:
            **kwargs: Arbitrary keyword arguments. Specifically looks for:
                - "go_branch" (str): The desired GO branch.
        Returns:
            str: The GO branch value. This will be one of the allowed values.

        Raises:
            ValueError: If the provided 'go_branch' value is not in the allowed list of values.
        """

        go_branch_value = kwargs.get("go_branch", cls._ALL_GO_BRANCHES)
        allowed_values = list(cls._GO_BRANCH_NAMESPACE.keys()) + [cls._ALL_GO_BRANCHES]
        if go_branch_value not in allowed_values:
            raise ValueError(
                f"Invalid value for go_branch: {go_branch_value}, Allowed values: {allowed_values}"
            )
        return go_branch_value

    # ------------------------------ Phase: Prepare data -----------------------------------
    def _download_required_data(self) -> str:
        """
        Downloads the required raw data related to Gene Ontology (GO) and Swiss-UniProt dataset.

        Returns:
            str: Path to the downloaded data.
        """
        self._download_swiss_uni_prot_data()
        return self._download_gene_ontology_data()

    def _download_gene_ontology_data(self) -> str:
        """
        Download the Gene Ontology data `.obo` file.

        Note:
            Quote from : https://geneontology.org/docs/download-ontology/
            Three versions of the ontology are available, the one use in this method is described below:
            https://purl.obolibrary.org/obo/go/go-basic.obo
            The basic version of the GO, filtered such that the graph is guaranteed to be acyclic and annotations
            can be propagated up the graph. The relations included are `is a, part of, regulates, negatively`
            `regulates` and `positively regulates`. This version excludes relationships that cross the 3 GO
            hierarchies. This version should be used with most GO-based annotation tools.

        Returns:
            str: The file path of the loaded Gene Ontology data.
        """
        go_path = os.path.join(self.raw_dir, self.raw_file_names_dict["GO"])
        os.makedirs(os.path.dirname(go_path), exist_ok=True)

        if not os.path.isfile(go_path):
            print("Missing Gene Ontology raw data")
            print(f"Downloading Gene Ontology data....")
            r = requests.get(self._GO_DATA_URL, allow_redirects=True)
            r.raise_for_status()  # Check if the request was successful
            open(go_path, "wb").write(r.content)
        return go_path

    def _download_swiss_uni_prot_data(self) -> Optional[str]:
        """
        Download the Swiss-Prot data file from UniProt Knowledgebase.

        Note:
            UniProt Knowledgebase is collection of functional information on proteins, with accurate, consistent
                and rich annotation.

            Swiss-Prot contains manually-annotated records with information extracted from literature and
                curator-evaluated computational analysis.

        Returns:
            str: The file path of the loaded Swiss-Prot data file.
        """
        uni_prot_file_path = os.path.join(
            self.raw_dir, self.raw_file_names_dict["SwissUniProt"]
        )
        os.makedirs(os.path.dirname(uni_prot_file_path), exist_ok=True)

        if not os.path.isfile(uni_prot_file_path):
            print(f"Downloading Swiss UniProt data....")

            # Create a temporary file
            with NamedTemporaryFile(delete=False) as tf:
                temp_filename = tf.name
                print(f"Downloading to temporary file {temp_filename}")

                # Download the file
                response = requests.get(self._SWISS_DATA_URL, stream=True)
                with open(temp_filename, "wb") as temp_file:
                    shutil.copyfileobj(response.raw, temp_file)

                print(f"Downloaded to {temp_filename}")

            # Unpack the gzipped file
            try:
                print(f"Unzipping the file....")
                with gzip.open(temp_filename, "rb") as f_in:
                    output_file_path = uni_prot_file_path
                    with open(output_file_path, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)
                print(f"Unpacked and saved to {output_file_path}")

            except Exception as e:
                print(f"Failed to unpack the file: {e}")
            finally:
                # Clean up the temporary file
                os.remove(temp_filename)
                print(f"Removed temporary file {temp_filename}")

        return uni_prot_file_path

    def _extract_class_hierarchy(self, data_path: str) -> nx.DiGraph:
        """
        Extracts the class hierarchy from the GO ontology.
        Constructs a directed graph (DiGraph) using NetworkX, where nodes are annotated with GO term data.

        Args:
            data_path (str): The path to the GO ontology.

        Returns:
            nx.DiGraph: A directed graph representing the class hierarchy, where nodes are GO terms and edges
                represent parent-child relationships.
        """
        print("Extracting class hierarchy...")
        elements = []
        for term in fastobo.load(data_path):
            if isinstance(term, fastobo.typedef.TypedefFrame):
                # ---- To avoid term frame of the below format/structure ----
                # [Typedef]
                # id: part_of
                # name: part of
                # namespace: external
                # xref: BFO:0000050
                # is_transitive: true
                continue

            if (
                term
                and isinstance(term.id, fastobo.id.PrefixedIdent)
                and term.id.prefix == self._GO_DATA_INIT
            ):
                # Consider only terms with id in following format - GO:2001271
                term_dict = self.term_callback(term)
                if term_dict:
                    elements.append(term_dict)

        g = nx.DiGraph()

        # Add GO term nodes to the graph and their hierarchical ontology
        for n in elements:
            g.add_node(n["go_id"], **n)
        g.add_edges_from(
            [
                (parent_id, node_id)
                for node_id in g.nodes
                for parent_id in g.nodes[node_id]["parents"]
                if parent_id in g.nodes
            ]
        )

        print("Compute transitive closure")
        return nx.transitive_closure_dag(g)

    def term_callback(self, term: fastobo.term.TermFrame) -> Union[Dict, bool]:
        """
        Extracts information from a Gene Ontology (GO) term document.

        Args:
            term: A Gene Ontology term Frame document.

        Returns:
            Optional[Dict]: A dictionary containing the extracted information if the term is not obsolete,
                            otherwise None. The dictionary includes:
                            - "id" (str): The ID of the GO term.
                            - "parents" (List[str]): A list of parent term IDs.
                            - "name" (str): The name of the GO term.
        """
        parents = []
        name = None

        for clause in term:
            if isinstance(clause, fastobo.term.NamespaceClause):
                if (
                    self.go_branch != self._ALL_GO_BRANCHES
                    and clause.namespace.escaped
                    != self._GO_BRANCH_NAMESPACE[self.go_branch]
                ):
                    # if the term document is not related to given go branch (except `all`), skip this document.
                    return False

            if isinstance(clause, fastobo.term.IsObsoleteClause):
                if clause.obsolete:
                    # if the term document contains clause as obsolete as true, skips this document.
                    return False

            if isinstance(clause, fastobo.term.IsAClause):
                parents.append(self._parse_go_id(clause.term))
            elif isinstance(clause, fastobo.term.NameClause):
                name = clause.name

        return {
            "go_id": self._parse_go_id(term.id),
            "parents": parents,
            "name": name,
        }

    @staticmethod
    def _parse_go_id(go_id: str) -> int:
        """
        Helper function to parse and normalize GO term IDs.

        Args:
            go_id: The raw GO term ID string.

        Returns:
            str: The parsed and normalized GO term ID.
        """
        # `is_a` clause has GO id in the following formats:
        # GO:0009968 ! negative regulation of signal transduction
        # GO:0046780
        return int(str(go_id).split(":")[1].split("!")[0].strip())

    def _graph_to_raw_dataset(self, g: nx.DiGraph) -> pd.DataFrame:
        """
        Processes a directed acyclic graph (DAG) to create a raw dataset in DataFrame format. The dataset includes
        Swiss-Prot protein data and their associations with Gene Ontology (GO) terms.

        Note:
            - GO classes are used as labels in the dataset. Each GO term is represented as a column, and its value
                indicates whether a Swiss-Prot protein is associated with that GO term.
            - Swiss-Prot proteins serve as samples. There is no 1-to-1 correspondence between Swiss-Prot proteins
                and GO terms.

        Data Format: pd.DataFrame
            - Column 0 : swiss_id (Identifier for SwissProt protein)
            - Column 1 : Accession of the protein
            - Column 2 : GO IDs (associated GO terms)
            - Column 3 : Sequence of the protein
            - Column 4 to Column "n": Each column corresponding to a class with value True/False indicating whether the
                protein is associated with this GO term.

        Args:
            g (nx.DiGraph): The class hierarchy graph.

        Returns:
            pd.DataFrame: The raw dataset created from the graph.
        """
        print(f"Processing graph")

        data_df = self._get_swiss_to_go_mapping()
        # add ancestors to go ids
        data_df["go_ids"] = data_df["go_ids"].apply(
            lambda go_ids: sorted(
                set(
                    itertools.chain.from_iterable(
                        [
                            [go_id] + list(g.predecessors(go_id))
                            for go_id in go_ids
                            if go_id in g.nodes
                        ]
                    )
                )
            )
        )
        # Initialize the GO term labels/columns to False
        selected_classes = self.select_classes(g, data_df=data_df)
        new_label_columns = pd.DataFrame(
            False, index=data_df.index, columns=selected_classes
        )
        data_df = pd.concat([data_df, new_label_columns], axis=1)

        # Set True for the corresponding GO IDs in the DataFrame go labels/columns
        for index, row in data_df.iterrows():
            for go_id in row["go_ids"]:
                if go_id in data_df.columns:
                    data_df.at[index, go_id] = True

        # This filters the DataFrame to include only the rows where at least one value in the row from 5th column
        # onwards is True/non-zero.
        # Quote from DeepGo Paper: `For training and testing, we use proteins which have been annotated with at least
        # one GO term from the set of the GO terms for the model`
        data_df = data_df[data_df.iloc[:, self._LABELS_START_IDX :].any(axis=1)]
        return data_df

    def _get_swiss_to_go_mapping(self) -> pd.DataFrame:
        """
        Parses the Swiss-Prot data and returns a DataFrame mapping Swiss-Prot records to Gene Ontology (GO) data.

        The DataFrame includes the following columns:
            - "swiss_id": The unique identifier for each Swiss-Prot record.
            - "sequence": The protein sequence.
            - "accessions": Comma-separated list of accession numbers.
            - "go_ids": List of GO IDs associated with the Swiss-Prot record.

        Note:
            This mapping is necessary because the GO data does not include the protein sequence representation.
            We select proteins with annotations having experimental evidence codes, as specified in
            `EXPERIMENTAL_EVIDENCE_CODES` and filter the proteins by a maximum length of 1002, ignoring proteins with
            ambiguous amino acid codes specified in `AMBIGUOUS_AMINO_ACIDS` in their sequence.

            Check the link below for keyword details:
            https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/docs/keywlist.txt

        Returns:
            pd.DataFrame: A DataFrame where each row corresponds to a Swiss-Prot record with its associated GO data.
        """

        print("Parsing swiss uniprot raw data....")

        swiss_ids, sequences, accessions, go_ids_list = [], [], [], []

        swiss_data = SwissProt.parse(
            open(
                os.path.join(self.raw_dir, self.raw_file_names_dict["SwissUniProt"]),
                "r",
            )
        )

        for record in swiss_data:
            if record.data_class != "Reviewed":
                # To consider only manually-annotated swiss data
                continue

            if not record.sequence or len(record.sequence) > self.max_sequence_length:
                # Consider protein with only sequence representation and seq. length not greater than max seq. length
                continue

            if any(aa in AMBIGUOUS_AMINO_ACIDS for aa in record.sequence):
                # Skip proteins with ambiguous amino acid codes
                continue

            go_ids = []

            for cross_ref in record.cross_references:
                if cross_ref[0] == self._GO_DATA_INIT:
                    # One swiss data protein can correspond to many GO data instances

                    if len(cross_ref) <= 3:
                        # No evidence code
                        continue

                    # https://github.com/bio-ontology-research-group/deepgo/blob/master/get_functions.py#L63-L66
                    evidence_code = cross_ref[3].split(":")[0]
                    if evidence_code not in EXPERIMENTAL_EVIDENCE_CODES:
                        # Skip GO id  without the required experimental evidence codes
                        continue

                    go_ids.append(self._parse_go_id(cross_ref[1]))

            if not go_ids:
                # Skip Swiss proteins without mapping to GO data
                continue

            swiss_ids.append(record.entry_name)
            sequences.append(record.sequence)
            accessions.append(",".join(record.accessions))
            go_ids.sort()
            go_ids_list.append(go_ids)

        data_dict = OrderedDict(
            swiss_id=swiss_ids,  # swiss_id column at index 0
            accession=accessions,  # Accession column at index 1
            go_ids=go_ids_list,  # Go_ids (data representation) column at index 2
            sequence=sequences,  # Sequence column at index 3
        )

        return pd.DataFrame(data_dict)

    # ------------------------------ Phase: Setup data -----------------------------------
    def _load_dict(self, input_file_path: str) -> Generator[Dict[str, Any], None, None]:
        """
        Loads data from a pickled file and yields individual dictionaries for each row.

        The pickled file is expected to contain rows with the following structure:
            - Data at row index `self._ID_IDX`: ID of go data instance
            - Data at row index `self._DATA_REPRESENTATION_IDX`: Sequence representation of protein
            - Data from row index `self._LABELS_START_IDX` onwards: Labels

        This method is used by `_load_data_from_file` to generate dictionaries that are then
        processed and converted into a list of dictionaries containing the features and labels.

        Args:
            input_file_path (str): The path to the pickled input file.

        Yields:
            Dict[str, Any]: A dictionary containing:
                - `features` (str): The sequence data from the file.
                - `labels` (np.ndarray): A boolean array of labels starting from row index 4.
                - `ident` (Any): The identifier from row index 0.
        """
        with open(input_file_path, "rb") as input_file:
            df = pd.read_pickle(input_file)
            for row in df.values:
                labels = row[self._LABELS_START_IDX :].astype(bool)
                # chebai.preprocessing.reader.DataReader only needs features, labels, ident, group
                # "group" set to None, by default as no such entity for this data
                yield dict(
                    features=row[self._DATA_REPRESENTATION_IDX],
                    labels=labels,
                    ident=row[self._ID_IDX],
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
        train_df_go, df_test = self.get_test_split(
            df_go_data, seed=self.dynamic_data_split_seed
        )

        # Get all splits
        df_train, df_val = self.get_train_val_splits_given_test(
            train_df_go,
            df_test,
            seed=self.dynamic_data_split_seed,
        )

        return df_train, df_val, df_test

    # ------------------------------ Phase: Raw Properties -----------------------------------
    @property
    def base_dir(self) -> str:
        """
        Returns the base directory path for storing GO-Uniprot data.

        Returns:
            str: The path to the base directory, which is "data/GO_UniProt".
        """
        return os.path.join("data", f"GO_UniProt")

    @property
    def raw_file_names_dict(self) -> dict:
        """
        Returns a dictionary of raw file names used in data processing.

        Returns:
            dict: A dictionary mapping dataset names to their respective file names.
                  For example, {"GO": "go-basic.obo", "SwissUniProt": "uniprot_sprot.dat"}.
        """
        return {"GO": "go-basic.obo", "SwissUniProt": "uniprot_sprot.dat"}


class _GOUniProtOverX(_GOUniProtDataExtractor, ABC):
    """
    A class for extracting data from the Gene Ontology (GO) dataset with a threshold for selecting classes based on
    the number of subclasses.

    This class is designed to filter GO classes based on a specified threshold, selecting only those classes
    which have a certain number of subclasses in the hierarchy.

    Attributes:
        READER (dr.ProteinDataReader): The reader used for reading the dataset.
        THRESHOLD (int): The threshold for selecting classes based on the number of subclasses.
    """

    READER: dr.ProteinDataReader = dr.ProteinDataReader
    THRESHOLD: int = None

    @property
    def _name(self) -> str:
        """
        Returns the name of the dataset.

        'max_sequence_length' in the name indicates that proteins with sequence lengths exceeding  are ignored
        in the dataset.

        Returns:
            str: The dataset name, formatted with the current threshold value and/or given go_branch.
        """
        if self.go_branch != self._ALL_GO_BRANCHES:
            return f"GO{self.THRESHOLD}_{self.go_branch}_{self.max_sequence_length}"

        return f"GO{self.THRESHOLD}_{self.max_sequence_length}"

    def select_classes(
        self, g: nx.DiGraph, *args: Any, **kwargs: Dict[str, Any]
    ) -> List[int]:
        """
        Selects classes (GO terms) from the Gene Ontology (GO) dataset based on the number of annotations meeting a
        specified threshold.

        The selection process is based on the annotations of the GO terms with its ancestors across the dataset.

        Annotations are calculated by counting how many times each GO term, along with its ancestral hierarchy,
        is annotated per protein across the dataset.
        This means that for each protein, the GO terms associated with it are considered, and the entire hierarchical
        structure (ancestors) of each GO term is taken into account. The total count for each GO term and its ancestors
        reflects how frequently these terms are annotated across all proteins in the dataset.

        Args:
            g (nx.DiGraph): The directed acyclic graph representing the GO dataset, where each node corresponds to a GO term.
            *args: Additional positional arguments (not used).
            **kwargs: Additional keyword arguments, including:
                - data_df (pd.DataFrame): A DataFrame containing the GO annotations for various proteins.
                                          It should include a 'go_ids' column with the GO terms associated with each protein.

        Returns:
            List[int]: A sorted list of selected GO term IDs that meet the annotation threshold criteria.

        Side Effects:
            - Writes the list of selected GO term IDs to a file named "classes.txt" in the specified processed directory.

        Raises:
            AttributeError: If the 'data_df' argument is not provided in kwargs.

        Notes:
            - The `THRESHOLD` attribute, which defines the minimum number of annotations required to select a GO term, should be defined in the subclass.
        """
        # Retrieve the DataFrame containing GO annotations per protein from the keyword arguments
        data_df = kwargs.get("data_df", None)
        if data_df is None or not isinstance(data_df, pd.DataFrame) or data_df.empty:
            raise AttributeError(
                "The 'data_df' argument must be provided and must be a non-empty pandas DataFrame."
            )

        print(f"Selecting GO terms based on given threshold: {self.THRESHOLD} ...")

        # https://github.com/bio-ontology-research-group/deepgo/blob/master/get_functions.py#L59-L77
        go_term_annot: Dict[int, int] = {}
        for idx, row in data_df.iterrows():
            # Count the annotations for each go_id **`per protein`**
            for go_id in row["go_ids"]:
                if go_id not in go_term_annot:
                    go_term_annot[go_id] = 0
                go_term_annot[go_id] += 1

        # Select GO terms that meet or exceed the threshold of annotations
        selected_nodes: List[int] = [
            go_id
            for go_id in g.nodes
            if go_id in go_term_annot and go_term_annot[go_id] >= self.THRESHOLD
        ]

        # Sort the selected nodes (optional but often useful for consistent output)
        selected_nodes.sort()

        # Write the selected node IDs/classes to the file
        filename = "classes.txt"
        with open(os.path.join(self.processed_dir_main, filename), "wt") as fout:
            fout.writelines(str(node) + "\n" for node in selected_nodes)

        return selected_nodes


class GOUniProtOver250(_GOUniProtOverX):
    """
    A class for extracting data from the Gene Ontology (GO) dataset with a threshold of 250 for selecting classes.

    Inherits from `_GOUniProtOverX` and sets the threshold for selecting classes to 250.

    Attributes:
        THRESHOLD (int): The threshold for selecting classes (250).
    """

    THRESHOLD: int = 250


class GOUniProtOver50(_GOUniProtOverX):
    """
    A class for extracting data from the Gene Ontology (GO) dataset with a threshold of 50 for selecting classes.

    Inherits from `_GOUniProtOverX` and sets the threshold for selecting classes to 50.

    Attributes:
        THRESHOLD (int): The threshold for selecting classes (50).
    """

    THRESHOLD: int = 50


class _DeepGOMigratedData(_GOUniProtDataExtractor, ABC):
    """
    Base class for use of the migrated DeepGO data with common properties, name formatting, and file paths.

    Attributes:
        READER (dr.ProteinDataReader): Protein data reader class.
        THRESHOLD (Optional[int]): Threshold value for GO class selection,
            determined by the GO branch type in derived classes.
    """

    READER: dr.ProteinDataReader = dr.ProteinDataReader
    THRESHOLD: Optional[int] = None

    # Mapping from GO branch conventions used in DeepGO to our conventions
    GO_BRANCH_MAPPING: dict = {
        "cc": "CC",
        "mf": "MF",
        "bp": "BP",
    }

    @property
    def _name(self) -> str:
        """
        Generates a unique identifier for the migrated data based on the GO
        branch and max sequence length, optionally including a threshold.

        Returns:
            str: A formatted name string for the data.
        """
        threshold_part = f"GO{self.THRESHOLD}_" if self.THRESHOLD is not None else "GO_"

        if self.go_branch != self._ALL_GO_BRANCHES:
            return f"{threshold_part}{self.go_branch}_{self.max_sequence_length}"

        return f"{threshold_part}{self.max_sequence_length}"

    # ------------------------------ Phase: Prepare data -----------------------------------
    def prepare_data(self, *args: Any, **kwargs: Any) -> None:
        """
        Checks for the existence of migrated DeepGO data in the specified directory.
        Raises an error if the required data file is not found, prompting
        migration from DeepGO to this data structure.

        Args:
            *args (Any): Additional positional arguments.
            **kwargs (Any): Additional keyword arguments.

        Raises:
            FileNotFoundError: If the processed data file does not exist.
        """
        print("Checking for processed data in", self.processed_dir_main)

        processed_name = self.processed_main_file_names_dict["data"]
        if not os.path.isfile(os.path.join(self.processed_dir_main, processed_name)):
            raise FileNotFoundError(
                f"File {processed_name} not found.\n"
                f"You must run the appropriate DeepGO migration script "
                f"(chebai/preprocessing/migration/deep_go) before executing this configuration "
                f"to migrate data from DeepGO to this data structure."
            )

    def select_classes(self, g: nx.DiGraph, *args, **kwargs) -> List:
        # Selection of GO classes not needed for migrated data
        pass

    # ------------------------------ Phase: Raw Properties -----------------------------------
    @property
    @abstractmethod
    def processed_main_file_names_dict(self) -> Dict[str, str]:
        """
        Abstract property for defining main processed file names.
        These files are stored in the same directory as the generated data files
        but have distinct names to differentiate them during training.

        Returns:
            dict: A dictionary with key-value pairs for main processed file names.
        """
        pass

    @property
    @abstractmethod
    def processed_file_names_dict(self) -> Dict[str, str]:
        """
        Abstract property for defining additional processed file names.
        These files are stored in the same directory as the generated data files
        but have distinct names to differentiate them during training.

        Returns:
            dict: A dictionary with key-value pairs for processed file names.
        """
        pass


class DeepGO1MigratedData(_DeepGOMigratedData):
    """
    Migrated data class specific to DeepGO1. Sets threshold values according
    to the research paper based on the GO branch.

    Note:
        Refer reference number 1 at the top of this file for the corresponding research paper.

    Args:
        **kwargs: Arbitrary keyword arguments passed to the superclass.

    Raises:
        ValueError: If an unsupported GO branch is provided.
    """

    def __init__(self, **kwargs):
        # https://github.com/bio-ontology-research-group/deepgo2/blob/main/deepgo/aminoacids.py#L11
        assert int(kwargs.get("max_sequence_length")) == 1002

        # Set threshold based on GO branch, as per DeepGO1 paper and its data.
        if kwargs.get("go_branch") in ["CC", "MF"]:
            self.THRESHOLD = 50
        elif kwargs.get("go_branch") == "BP":
            self.THRESHOLD = 250
        else:
            raise ValueError(
                f"DeepGO1 paper has no defined threshold for branch {self.go_branch}"
            )

        super(_DeepGOMigratedData, self).__init__(**kwargs)

    @property
    def processed_main_file_names_dict(self) -> Dict[str, str]:
        """
        Returns main processed file names specific to DeepGO1.

        Returns:
            dict: Dictionary with the main data file name for DeepGO1.
        """
        return {"data": "data_deep_go1.pkl"}

    @property
    def processed_file_names_dict(self) -> Dict[str, str]:
        """
        Returns processed file names specific to DeepGO1.

        Returns:
            dict: Dictionary with data file name for DeepGO1.
        """
        return {"data": "data_deep_go1.pt"}


class DeepGO2MigratedData(_DeepGOMigratedData):
    """
    Migrated data class specific to DeepGO2, inheriting from DeepGO1MigratedData
    with different processed file names.

    Note:
        Refer reference number 3 at the top of this file for the corresponding research paper.

    Returns:
        dict: Dictionary with file names specific to DeepGO2.
    """

    _LABELS_START_IDX: int = 5  # additional esm2_embeddings column in the dataframe
    _ESM_EMBEDDINGS_COL_IDX: int = 4

    def __init__(self, use_esm2_embeddings=False, **kwargs):
        # https://github.com/bio-ontology-research-group/deepgo2/blob/main/deepgo/aminoacids.py#L11
        assert int(kwargs.get("max_sequence_length")) == 1000
        self.use_esm2_embeddings: bool = use_esm2_embeddings
        super(_DeepGOMigratedData, self).__init__(**kwargs)

    # ------------------------------ Phase: Setup data -----------------------------------
    def _load_data_from_file(self, path: str) -> List[Dict[str, Any]]:
        """
        Load and process data from a file into a list of dictionaries containing features and labels.

        This method processes data differently based on the `use_esm2_embeddings` flag:
        - If `use_esm2_embeddings` is True, raw dictionaries from `_load_dict` are returned, _load_dict already returns
        the numerical features (esm2 embeddings) from the data file, hence no reader is required.
        - Otherwise, a reader is used to process the data (generate numerical features).

        Args:
            path (str): The path to the input file.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries with the following keys:
                - `features`: Sequence or embedding data, depending on the context.
                - `labels`: A boolean array of labels.
                - `ident`: The identifier for the sequence.
        """
        lines = self._get_data_size(path)
        print(f"Processing {lines} lines...")

        if self.use_esm2_embeddings:
            data = [
                d
                for d in tqdm.tqdm(self._load_dict(path), total=lines)
                if d["features"] is not None
            ]
        else:
            data = [
                self.reader.to_data(d)
                for d in tqdm.tqdm(self._load_dict(path), total=lines)
                if d["features"] is not None
            ]

        # filter for missing features in resulting data
        data = [val for val in data if val["features"] is not None]

        return data

    def _load_dict(self, input_file_path: str) -> Generator[Dict[str, Any], None, None]:
        """
        Loads data from a pickled file and yields individual dictionaries for each row.

        The pickled file is expected to contain rows with the following structure:
            - Data at row index `self._ID_IDX`: ID of go data instance
            - Data at row index `self._DATA_REPRESENTATION_IDX`: Sequence representation of protein
            - Data at row index `self._ESM2_EMBEDDINGS_COL_IDX`: ESM2 embeddings of the protein
            - Data from row index `self._LABELS_START_IDX` onwards: Labels

        The method adapts based on the `use_esm2_embeddings` flag:
            - If `use_esm2_embeddings` is True, features are loaded from the column specified by `self._ESM_EMBEDDINGS_COL_IDX`.
            - Otherwise, features are loaded from the column specified by `self._DATA_REPRESENTATION_IDX`.

        Args:
            input_file_path (str): The path to the pickled input file.

        Yields:
            Dict[str, Any]: A dictionary containing:
                - `features` (Any): Sequence or embedding data for the instance.
                - `labels` (np.ndarray): A boolean array of labels starting from row index 4.
                - `ident` (Any): The identifier from row index 0.
        """
        with open(input_file_path, "rb") as input_file:
            df = pd.read_pickle(input_file)

            if self.use_esm2_embeddings:
                features_idx = self._ESM_EMBEDDINGS_COL_IDX
            else:
                features_idx = self._DATA_REPRESENTATION_IDX

            for row in df.values:
                labels = row[self._LABELS_START_IDX :].astype(bool)
                yield dict(
                    features=row[features_idx],
                    labels=labels,
                    ident=row[self._ID_IDX],
                )

    # ------------------------------ Phase: Raw Properties -----------------------------------
    @property
    def processed_main_file_names_dict(self) -> Dict[str, str]:
        """
        Returns main processed file names specific to DeepGO2.

        Returns:
            dict: Dictionary with the main data file name for DeepGO2.
        """
        return {"data": "data_deep_go2.pkl"}

    @property
    def processed_file_names_dict(self) -> Dict[str, str]:
        """
        Returns processed file names specific to DeepGO2.

        Returns:
            dict: Dictionary with data file name for DeepGO2.
        """
        return {"data": "data_deep_go2.pt"}

    @property
    def identifier(self) -> tuple:
        """Identifier for the dataset."""
        if self.use_esm2_embeddings:
            return (dr.ESM2EmbeddingReader.name(),)
        return (self.reader.name(),)
