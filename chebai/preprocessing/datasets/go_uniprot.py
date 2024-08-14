# Reference for this file :
# Maxat Kulmanov, Mohammed Asif Khan, Robert Hoehndorf;
# DeepGO: Predicting protein functions from sequence and interactions
# using a deep ontology-aware classifier, Bioinformatics, 2017.
# https://doi.org/10.1093/bioinformatics/btx624
# Github: https://github.com/bio-ontology-research-group/deepgo
# https://www.ebi.ac.uk/GOA/downloads
# https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/docs/keywlist.txt
# https://www.uniprot.org/uniprotkb

__all__ = ["GoUniProtOver100", "GoUniProtOver50"]

import gzip
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
from Bio import SwissProt

from chebai.preprocessing import reader as dr
from chebai.preprocessing.datasets.base import _DynamicDataset


class _GOUniprotDataExtractor(_DynamicDataset, ABC):
    """
    A class for extracting and processing data from the Gene Ontology (GO) dataset and the Swiss UniProt dataset.

    Args:
        dynamic_data_split_seed (int, optional): The seed for random data splitting. Defaults to 42.
        splits_file_path (str, optional): Path to the splits CSV file. Defaults to None.
        **kwargs: Additional keyword arguments passed to XYBaseDataModule.

    Attributes:
        dynamic_data_split_seed (int): The seed for random data splitting, default is 42.
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
        super(_GOUniprotDataExtractor, self).__init__(**kwargs)

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
            [(parent, node["go_id"]) for node in elements for parent in node["parents"]]
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
                    and clause.namespace != self._GO_BRANCH_NAMESPACE[self.go_branch]
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

        # Initialize the GO term labels/columns to False
        data_df[self.select_classes(g)] = False
        # Set True for the corresponding GO IDs in the DataFrame go labels/columns
        for index, row in data_df.iterrows():
            for go_id in row["go_ids"]:
                if go_id in data_df.columns:
                    data_df.at[index, go_id] = True

        # This filters the DataFrame to include only the rows where at least one value in the row from 5th column
        # onwards is True/non-zero.
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

            if not record.sequence:
                # Consider protein with only sequence representation
                continue

            go_ids = []
            for cross_ref in record.cross_references:
                if cross_ref[0] == self._GO_DATA_INIT:
                    # One swiss data protein can correspond to many GO data instances
                    go_ids.append(self._parse_go_id(cross_ref[1]))

            if not go_ids:
                # Swiss protein with no mapping to Gene Ontology is skipped
                continue

            go_ids.sort()

            swiss_ids.append(record.entry_name)
            sequences.append(record.sequence)
            accessions.append(",".join(record.accessions))
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
            data_go = torch.load(os.path.join(self.processed_dir, filename))
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


class _GoUniProtOverX(_GOUniprotDataExtractor, ABC):
    """
    A class for extracting data from the Gene Ontology (GO) dataset with a threshold for selecting classes based on
    the number of subclasses.

    This class is designed to filter GO classes based on a specified threshold, selecting only those classes
    which have a certain number of subclasses in the hierarchy.

    Attributes:
        READER (dr.ProteinDataReader): The reader used for reading the dataset.
        THRESHOLD (int): The threshold for selecting classes based on the number of subclasses.

    Property:
        label_number (int): The number of labels in the dataset. This property must be implemented by subclasses.
    """

    READER: dr.ProteinDataReader = dr.ProteinDataReader
    THRESHOLD: int = None

    @property
    @abstractmethod
    def label_number(self) -> int:
        raise NotImplementedError

    @property
    def _name(self) -> str:
        """
        Returns the name of the dataset.

        Returns:
            str: The dataset name, formatted with the current threshold value and/or given go_branch.
        """
        if self.go_branch != self._ALL_GO_BRANCHES:
            return f"GO{self.THRESHOLD}_{self.go_branch}"

        return f"GO{self.THRESHOLD}"

    def select_classes(self, g: nx.DiGraph, *args, **kwargs) -> List:
        """
        Selects classes from the GO dataset based on the number of successors meeting a specified threshold.

        This method iterates over the nodes in the graph, counting the number of successors for each node.
        Nodes with a number of successors greater than or equal to the defined threshold are selected.

        Note:
            The input graph must be transitive closure of a directed acyclic graph.

        Args:
            g (nx.DiGraph): The graph representing the dataset. Each node should have a 'sequence' attribute.
            *args: Additional positional arguments (not used).
            **kwargs: Additional keyword arguments (not used).

        Returns:
            List: A sorted list of node IDs that meet the successor threshold criteria.

        Side Effects:
            Writes the list of selected nodes to a file named "classes.txt" in the specified processed directory.

        Notes:
            - The `THRESHOLD` attribute should be defined in the subclass.
        """
        nodes = []
        for node in g.nodes:
            # Count the number of successors (direct child nodes) for each node
            if len(list(g.successors(node))) >= self.THRESHOLD:
                nodes.append(node)

        nodes.sort()

        # Write the selected node IDs/classes to the file
        filename = "classes.txt"
        with open(os.path.join(self.processed_dir_main, filename), "wt") as fout:
            fout.writelines(str(node) + "\n" for node in nodes)
        return nodes


class GoUniProtOver100(_GoUniProtOverX):
    """
    A class for extracting data from the Gene Ontology (GO) dataset with a threshold of 100 for selecting classes.

    Inherits from `_GoUniProtOverX` and sets the threshold for selecting classes to 100.

    Attributes:
        THRESHOLD (int): The threshold for selecting classes (100).
    """

    THRESHOLD: int = 100

    def label_number(self) -> int:
        """
        Returns the number of labels in the dataset for this threshold.

        Overrides the base class method to provide the correct number of labels for a threshold of 100.

        Returns:
            int: The number of labels (854).
        """
        return 854


class GoUniProtOver50(_GoUniProtOverX):
    """
    A class for extracting data from the Gene Ontology (GO) dataset with a threshold of 50 for selecting classes.

    Inherits from `_GoUniProtOverX` and sets the threshold for selecting classes to 50.

    Attributes:
        THRESHOLD (int): The threshold for selecting classes (50).
    """

    THRESHOLD: int = 50

    def label_number(self) -> int:
        """
        Returns the number of labels in the dataset for this threshold.

        Overrides the base class method to provide the correct number of labels for a threshold of 50.

        Returns:
            int: The number of labels (1332).
        """
        return 1332
