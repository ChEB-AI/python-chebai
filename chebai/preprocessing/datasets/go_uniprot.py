# Reference for this file :
# Maxat Kulmanov, Mohammed Asif Khan, Robert Hoehndorf;
# DeepGO: Predicting protein functions from sequence and interactions
# using a deep ontology-aware classifier, Bioinformatics, 2017.
# https://doi.org/10.1093/bioinformatics/btx624
# Github: https://github.com/bio-ontology-research-group/deepgo
# https://www.ebi.ac.uk/GOA/downloads


__all__ = ["GoUniProtOver100", "GoUniProtOver50"]

import gzip
import os
import shutil
from abc import ABC, abstractmethod
from collections import OrderedDict
from tempfile import NamedTemporaryFile
from typing import Any, Dict, Generator, List, Optional, Tuple

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

    # ---- Index for columns of processed `data.pkl` (derived from `_graph_to_raw_dataset` method) ------
    # "id" at                 row index 0
    # "name" at               row index 1
    # "sequence" at           row index 2
    # "swiss_ident" at        row index 3
    # labels starting from    row index 4
    _ID_IDX: int = 0
    _DATA_REPRESENTATION_IDX: int = 2
    _LABELS_START_IDX: int = 4

    _GO_DATA_URL = "http://purl.obolibrary.org/obo/go/go-basic.obo"
    _SWISS_DATA_URL = "https://ftp.uniprot.org/pub/databases/uniprot/knowledgebase/complete/uniprot_sprot.dat.gz"

    def __init__(
        self,
        **kwargs,
    ):
        super(_GOUniprotDataExtractor, self).__init__(**kwargs)

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
            http://purl.obolibrary.org/obo/go/go-basic.obo
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
        Constructs a directed graph (DiGraph) using NetworkX, where nodes are annotated with GO term data
        and corresponding Swiss-Prot data (obtained via `_get_go_swiss_data_mapping`).

        Args:
            data_path (str): The path to the GO ontology.

        Returns:
            nx.DiGraph: A directed graph representing the class hierarchy, where nodes are GO terms and edges
                represent parent-child relationships.
        """
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

        go_to_swiss_mapping = self._get_go_swiss_data_mapping()

        g = nx.DiGraph()
        for n in elements:
            # Swiss data is mapped to respective go data instance
            node_mapping_dict = go_to_swiss_mapping.get(n["id"], {})
            # Combine the dictionaries for node attributes
            node_attributes = {**n, **node_mapping_dict}
            g.add_node(n["id"], **node_attributes)
        g.add_edges_from([(p, q["id"]) for q in elements for p in q["parents"]])

        print("Compute transitive closure")
        return nx.transitive_closure_dag(g)

    @staticmethod
    def term_callback(term: fastobo.term.TermFrame) -> Optional[Dict]:
        """
        Extracts information from a Gene Ontology (GO) term document.
        It also checks if the term is marked as obsolete and skips such terms.

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
            if isinstance(clause, fastobo.term.IsAClause):
                parents.append(_GOUniprotDataExtractor._parse_go_id(clause.term))
            elif isinstance(clause, fastobo.term.NameClause):
                name = clause.name
            elif isinstance(clause, fastobo.term.IsObsoleteClause):
                if clause.obsolete:
                    # if the term contains clause as obsolete as true, skips this term
                    return None

        return {
            "id": _GOUniprotDataExtractor._parse_go_id(term.id),
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
        # `is_a` clause has GO id in the following format:
        # is_a: GO:0009968 ! negative regulation of signal transduction
        return int(str(go_id).split(":")[1].split("!")[0].strip())

    def _get_go_swiss_data_mapping(self) -> Dict[int, Dict[str, str]]:
        """
        Parses the Swiss-Prot data and returns a mapping from Gene Ontology (GO) data ID to Swiss-Prot ID
        along with the sequence representation of the protein.

        This mapping is necessary because the GO data does not include the protein sequence representation.

        Returns:
            Dict[int, Dict[str, str]]: A dictionary where the keys are GO data IDs (int) and the values are
                dictionaries containing:
                    - "sequence" (str): The protein sequence.
                    - "swiss_ident" (str): The unique identifier for each Swiss-Prot record.
        """
        # # https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/docs/keywlist.txt
        #  ---------  ---------------------------     ------------------------------
        #  Line code  Content                         Occurrence in an entry
        #  ---------  ---------------------------     ------------------------------
        #  ID         Identifier (keyword)            Once; starts a keyword entry
        #  IC         Identifier (category)           Once; starts a category entry
        #  AC         Accession (KW-xxxx)             Once
        #  DE         Definition                      Once or more
        #  SY         Synonyms                        Optional; once or more
        #  GO         Gene ontology (GO) mapping      Optional; once or more
        #  HI         Hierarchy                       Optional; once or more
        #  WW         Relevant WWW site               Optional; once or more
        #  CA         Category                        Once per keyword entry;
        #                                             absent in category entries
        #  //         Terminator                      Once; ends an entry
        # ---------------------------------------------------------------------------
        print("Parsing swiss uniprot raw data....")

        swiss_go_mapping = {}
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
            # Cross-reference has mapping for each protein to each type of data set
            for cross_ref in record.cross_references:
                if cross_ref[0] == self._GO_DATA_INIT:
                    # Only consider cross-reference related to GO dataset
                    go_id = _GOUniprotDataExtractor._parse_go_id(cross_ref[1])
                    swiss_go_mapping[go_id] = {
                        "sequence": record.sequence,
                        "swiss_ident": record.entry_name,  # Unique identifier for each swiss data record
                    }
        return swiss_go_mapping

    def _graph_to_raw_dataset(self, g: nx.DiGraph) -> pd.DataFrame:
        """
        Preparation step before creating splits,
        uses the graph created by _extract_go_class_hierarchy() to extract the
        raw data in Dataframe format with extra columns corresponding to each multi-label class.

        Data Format: pd.DataFrame
            - Column 0 : ID (Identifier for GO data instance)
            - Column 1 : Name of the protein
            - Column 2 : Sequence representation of the protein
            - Column 3 : Unique identifier of the protein from swiss dataset.
            - Column 4 to Column "n": Each column corresponding to a class with value True/False indicating where the
                data instance belong to this class or not.
        Args:
            g (nx.DiGraph): The class hierarchy graph.

        Returns:
            pd.DataFrame: The raw dataset created from the graph.
        """
        sequences = nx.get_node_attributes(g, "sequence")
        names = nx.get_node_attributes(g, "name")
        swiss_idents = nx.get_node_attributes(g, "swiss_ident")

        print(f"Processing graph")

        # Gets list of node ids, names, sequences, swiss identifier where sequence is not empty/None.
        data_list = []
        for node_id, sequence in sequences.items():
            if sequence:
                data_list.append(
                    (
                        node_id,
                        names.get(node_id),
                        sequence,
                        swiss_idents.get(node_id),
                    )
                )

        node_ids, names_list, sequences_list, swiss_identifier_list = zip(*data_list)
        data = OrderedDict(id=node_ids)  # ID column at index 0

        data["name"] = names_list  # Name column at index 1
        data["sequence"] = (
            sequences_list  # Sequence (data representation) column at index 2
        )
        data["swiss_ident"] = swiss_identifier_list  # Swiss_ident column at index 3

        # Assuming select_classes is implemented and returns a list of class IDs
        for n in self.select_classes(g):
            data[n] = [
                ((n in g.predecessors(node)) or (n == node)) for node in node_ids
            ]

        data = pd.DataFrame(data)
        # This filters the DataFrame to include only the rows where at least one value in the row from 5th column
        # onwards is True/non-zero.
        data = data[data.iloc[:, self._LABELS_START_IDX :].any(axis=1)]
        return data

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
            str: The dataset name, formatted with the current threshold value.
        """
        return f"GO{self.THRESHOLD}"

    def select_classes(self, g: nx.DiGraph, *args, **kwargs) -> List:
        """
        Selects classes from the GO dataset based on the number of successors meeting a specified threshold.

        This method iterates over the nodes in the graph, counting the number of successors for each node.
        Nodes with a number of successors greater than or equal to the defined threshold are selected.

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
            - Nodes without a 'sequence' attribute are ignored in the successor count.
        """
        sequences = nx.get_node_attributes(g, "sequence")
        nodes = []
        for node in g.nodes:
            # Count the number of successors (child nodes) for each node
            no_of_successors = 0
            for s_node in g.successors(node):
                if sequences.get(s_node, None):
                    no_of_successors += 1

            if no_of_successors >= self.THRESHOLD:
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
