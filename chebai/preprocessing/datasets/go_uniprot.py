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
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import fastobo
import networkx as nx
import pandas as pd
import requests
import torch
from Bio import SwissProt
from iterstrat.ml_stratifiers import (
    MultilabelStratifiedKFold,
    MultilabelStratifiedShuffleSplit,
)

from chebai.preprocessing import reader as dr
from chebai.preprocessing.datasets import XYBaseDataModule


class _GOUniprotDataExtractor(XYBaseDataModule, ABC):
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
    # ---- Index for columns of processed `data.pkl` ------
    # "id" at                 row index 0
    # "name" at               row index 1
    # "sequence" at           row index 2
    # "swiss_ident" at        row index 3
    # labels starting from    row index 4
    _LABELS_STARTING_INDEX: int = 4
    _SEQUENCE_INDEX: int = 2
    _ID_INDEX = 0

    _GO_DATA_URL = "http://purl.obolibrary.org/obo/go/go-basic.obo"
    _SWISS_DATA_URL = "https://ftp.uniprot.org/pub/databases/uniprot/knowledgebase/complete/uniprot_sprot.dat.gz"

    def __init__(
        self,
        **kwargs,
    ):
        super(_GOUniprotDataExtractor, self).__init__(**kwargs)
        self.dynamic_data_split_seed = int(kwargs.get("seed", 42))  # default is 42
        # Class variables to store the dynamics splits
        self._dynamic_df_train = None
        self._dynamic_df_test = None
        self._dynamic_df_val = None
        # Path of csv file which contains a list of go ids & their assignment to a dataset (either train,
        # validation or test).
        self.splits_file_path = self._validate_splits_file_path(
            kwargs.get("splits_file_path", None)
        )

    @staticmethod
    def _validate_splits_file_path(splits_file_path: Optional[str]) -> Optional[str]:
        """
        Validates the file in provided splits file path.

        Args:
            splits_file_path (Optional[str]): Path to the splits CSV file.

        Returns:
            Optional[str]: Validated splits file path if checks pass, None if splits_file_path is None.

        Raises:
            FileNotFoundError: If the splits file does not exist.
            ValueError: If the splits file is empty or missing required columns ('id' and/or 'split'), or not a CSV file.
        """
        if splits_file_path is None:
            return None

        if not os.path.isfile(splits_file_path):
            raise FileNotFoundError(f"File {splits_file_path} does not exist")

        file_size = os.path.getsize(splits_file_path)
        if file_size == 0:
            raise ValueError(f"File {splits_file_path} is empty")

        # Check if the file has a CSV extension
        if not splits_file_path.lower().endswith(".csv"):
            raise ValueError(f"File {splits_file_path} is not a CSV file")

        # Read the first row of CSV file into a DataFrame
        splits_df = pd.read_csv(splits_file_path, nrows=1)

        # Check if 'id' and 'split' columns are in the DataFrame
        required_columns = {"id", "split"}
        if not required_columns.issubset(splits_df.columns):
            raise ValueError(
                f"CSV file {splits_file_path} is missing required columns ('id' and/or 'split')."
            )

        return splits_file_path

    @property
    def dynamic_split_dfs(self) -> Dict[str, pd.DataFrame]:
        """
        Property to retrieve dynamic train, validation, and test splits.

        This property checks if dynamic data splits (`_dynamic_df_train`, `_dynamic_df_val`, `_dynamic_df_test`)
        are already loaded. If any of them is None, it either generates them dynamically or retrieves them
        from data file with help of pre-existing Split csv file (`splits_file_path`) containing splits assignments.

        Returns:
            dict: A dictionary containing the dynamic train, validation, and test DataFrames.
                Keys are 'train', 'validation', and 'test'.
        """
        if any(
            split is None
            for split in [
                self._dynamic_df_test,
                self._dynamic_df_val,
                self._dynamic_df_train,
            ]
        ):
            if self.splits_file_path is None:
                # Generate splits based on given seed, create csv file to records the splits
                self._generate_dynamic_splits()
            else:
                # If user has provided splits file path, use it to get the splits from the data
                self._retrieve_splits_from_csv()
        return {
            "train": self._dynamic_df_train,
            "validation": self._dynamic_df_val,
            "test": self._dynamic_df_test,
        }

    def _generate_dynamic_splits(self) -> None:
        """
        Generate data splits during runtime and save them in class variables.

        This method loads encoded data generates train, validation, and test splits based on the loaded data.

        Raises:
            FileNotFoundError: If the required data file (`data.pt`) does not exist. It advises calling `prepare_data`
                or `setup` methods to generate the dataset files.
        """
        print("Generate dynamic splits...")
        # Load encoded data
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

        # Generate splits.csv file to store ids of each corresponding split
        split_assignment_list: List[pd.DataFrame] = [
            pd.DataFrame({"id": df_train["ident"], "split": "train"}),
            pd.DataFrame({"id": df_val["ident"], "split": "validation"}),
            pd.DataFrame({"id": df_test["ident"], "split": "test"}),
        ]
        combined_split_assignment = pd.concat(split_assignment_list, ignore_index=True)
        combined_split_assignment.to_csv(
            os.path.join(self.processed_dir_main, "splits.csv")
        )

        # Store the splits in class variables
        self._dynamic_df_train = df_train
        self._dynamic_df_val = df_val
        self._dynamic_df_test = df_test

    def _retrieve_splits_from_csv(self) -> None:
        """
        Retrieve previously saved data splits from splits.csv file or from provided file path.

        This method loads the splits.csv file located at `self.splits_file_path`.
        It then loads the encoded data (`data.pt`) and filters it based on the IDs retrieved from
        splits.csv to reconstruct the train, validation, and test splits.
        """
        print(f"Loading splits from {self.splits_file_path}...")
        splits_df = pd.read_csv(self.splits_file_path)

        filename = self.processed_file_names_dict["data"]
        data_go = torch.load(os.path.join(self.processed_dir, filename))
        df_go_data = pd.DataFrame(data_go)

        train_ids = splits_df[splits_df["split"] == "train"]["id"]
        validation_ids = splits_df[splits_df["split"] == "validation"]["id"]
        test_ids = splits_df[splits_df["split"] == "test"]["id"]

        self._dynamic_df_train = df_go_data[df_go_data["ident"].isin(train_ids)]
        self._dynamic_df_val = df_go_data[df_go_data["ident"].isin(validation_ids)]
        self._dynamic_df_test = df_go_data[df_go_data["ident"].isin(test_ids)]

    def get_test_split(
        self, df: pd.DataFrame, seed: Optional[int] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split the input DataFrame into training and testing sets based on multilabel stratified sampling.

        This method uses MultilabelStratifiedShuffleSplit to split the data such that the distribution of labels
        in the training and testing sets is approximately the same. The split is based on the "labels" column
        in the DataFrame.

        Args:
            df (pd.DataFrame): The input DataFrame containing the data to be split. It must contain a column
                               named "labels" with the multilabel data.
            seed (int, optional): The random seed to be used for reproducibility. Default is None.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the training set and testing set DataFrames.

        Raises:
            ValueError: If the DataFrame does not contain a column named "labels".
        """
        print("\nGet test data split")

        labels_list = df["labels"].tolist()

        test_size = 1 - self.train_split - (1 - self.train_split) ** 2
        msss = MultilabelStratifiedShuffleSplit(
            n_splits=1, test_size=test_size, random_state=seed
        )

        train_indices, test_indices = next(msss.split(labels_list, labels_list))

        df_train = df.iloc[train_indices]
        df_test = df.iloc[test_indices]
        return df_train, df_test

    def get_train_val_splits_given_test(
        self, df: pd.DataFrame, test_df: pd.DataFrame, seed: int = None
    ) -> Union[Dict[str, pd.DataFrame], Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Split the dataset into train and validation sets, given a test set.
        Use test set (e.g., loaded from another source or generated in get_test_split), to avoid overlap

        Args:
            df (pd.DataFrame): The original dataset.
            test_df (pd.DataFrame): The test dataset.
            seed (int, optional): The random seed to be used for reproducibility. Default is None.

        Returns:
            Union[Dict[str, pd.DataFrame], Tuple[pd.DataFrame, pd.DataFrame]]: A dictionary containing train and
                validation sets if self.use_inner_cross_validation is True, otherwise a tuple containing the train
                and validation DataFrames. The keys are the names of the train and validation sets, and the values
                are the corresponding DataFrames.
        """
        print(f"Split dataset into train / val with given test set")

        test_ids = test_df["ident"].tolist()
        df_trainval = df[~df["ident"].isin(test_ids)]
        labels_list_trainval = df_trainval["labels"].tolist()

        if self.use_inner_cross_validation:
            folds = {}
            kfold = MultilabelStratifiedKFold(
                n_splits=self.inner_k_folds, random_state=seed
            )
            for fold, (train_ids, val_ids) in enumerate(
                kfold.split(
                    labels_list_trainval,
                    labels_list_trainval,
                )
            ):
                df_validation = df_trainval.iloc[val_ids]
                df_train = df_trainval.iloc[train_ids]
                folds[self.raw_file_names_dict[f"fold_{fold}_train"]] = df_train
                folds[self.raw_file_names_dict[f"fold_{fold}_validation"]] = (
                    df_validation
                )

            return folds

        # scale val set size by 1/self.train_split to compensate for (hypothetical) test set size (1-self.train_split)
        test_size = ((1 - self.train_split) ** 2) / self.train_split
        msss = MultilabelStratifiedShuffleSplit(
            n_splits=1, test_size=test_size, random_state=seed
        )

        train_indices, validation_indices = next(
            msss.split(labels_list_trainval, labels_list_trainval)
        )

        df_validation = df_trainval.iloc[validation_indices]
        df_train = df_trainval.iloc[train_indices]
        return df_train, df_validation

    def setup_processed(self) -> None:
        """
        Transforms `data.pkl` into a model input data format (`data.pt`), ensuring that the data is in a format
        compatible for input to the model.
        The transformed data contains the following keys: `ident`, `features`, `labels`, and `group`.
        This method uses a subclass of Data Reader to perform the transformation.

        Returns:
            None
        """
        print("Transform data")
        os.makedirs(self.processed_dir, exist_ok=True)
        print("Missing transformed `data.pt` file. Transforming data.... ")
        torch.save(
            self._load_data_from_file(
                os.path.join(
                    self.processed_dir_main,
                    self.processed_dir_main_file_names_dict["data"],
                )
            ),
            os.path.join(self.processed_dir, self.processed_file_names_dict["data"]),
        )

    def _load_dict(self, input_file_path: str) -> Generator[Dict[str, Any], None, None]:
        """
        Loads data from a pickled file and yields individual dictionaries for each row.

        The pickled file is expected to contain rows with the following structure:
            - Data at row index 0: ID of go data instance
            - Data at row index 2: Sequence representation of protein
            - Data from row index 4 onwards: Labels

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
            # "id" at                 row index 0
            # "name" at               row index 1
            # "sequence" at           row index 2
            # "swiss_ident" at        row index 3
            # labels starting from    row index 4
            for row in df.values:
                labels = row[self._LABELS_STARTING_INDEX :].astype(bool)
                # chebai.preprocessing.reader.DataReader only needs features, labels, ident, group
                # "group" set to None, by default as no such entity for this data
                yield dict(
                    features=row[self._SEQUENCE_INDEX],
                    labels=labels,
                    ident=row[self._ID_INDEX],
                )

    def prepare_data(self, *args: Any, **kwargs: Any) -> None:
        """
        Prepares the data for the Go dataset.

        This method checks for the presence of raw data in the specified directory.
        If the raw data is missing, it fetches the ontology and creates a dataframe and saves it to a data.pkl file.

        The resulting dataframe/pickle file is expected to contain columns with the following structure:
            - Column at index 0: ID of go data instance
            - Column at index 2: Sequence representation of the protein
            - Column from index 4 onwards: Labels

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            None
        """
        print("Checking for processed data in", self.processed_dir_main)

        processed_name = self.processed_dir_main_file_names_dict["data"]
        if not os.path.isfile(os.path.join(self.processed_dir_main, processed_name)):
            print("Missing Gene Ontology processed data (`data.pkl` file)")
            os.makedirs(self.processed_dir_main, exist_ok=True)
            # swiss_path = self._download_swiss_uni_prot_data()
            self._download_swiss_uni_prot_data()
            go_path = self._download_gene_ontology_data()
            g = self._extract_go_class_hierarchy(go_path)
            data_df = self._graph_to_raw_dataset(g)
            self.save_processed(data_df, processed_name)

    @abstractmethod
    def select_classes(self, g: nx.DiGraph, *args, **kwargs) -> List:
        """
        Selects classes from the GO dataset based on a specified criteria.

        Args:
            g (nx.Graph): The graph representing the dataset.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            List: A sorted list of node IDs that meet the specified criteria.

        """
        raise NotImplementedError

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
        data = OrderedDict(id=node_ids)

        data["name"] = names_list
        data["sequence"] = sequences_list
        data["swiss_ident"] = swiss_identifier_list

        # Assuming select_classes is implemented and returns a list of class IDs
        for n in self.select_classes(g):
            data[n] = [
                ((n in g.predecessors(node)) or (n == node)) for node in node_ids
            ]

        data = pd.DataFrame(data)
        # This filters the DataFrame to include only the rows where at least one value in the row from 5th column
        # onwards is True/non-zero.
        data = data[data.iloc[:, self._LABELS_STARTING_INDEX :].any(axis=1)]
        return data

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

    def _extract_go_class_hierarchy(self, go_path: str) -> nx.DiGraph:
        """
        Extracts the class hierarchy from the GO ontology.
        Constructs a directed graph (DiGraph) using NetworkX, where nodes are annotated with GO term data
        and corresponding Swiss-Prot data (obtained via `_get_go_swiss_data_mapping`).

        Args:
            go_path (str): The path to the GO ontology.

        Returns:
            nx.DiGraph: A directed graph representing the class hierarchy, where nodes are GO terms and edges
                represent parent-child relationships.
        """
        elements = []
        for term in fastobo.load(go_path):
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

    def save_processed(self, data: pd.DataFrame, filename: str) -> None:
        """
        Save the processed dataset to a pickle file.

        Args:
            data (pd.DataFrame): The processed dataset to be saved.
            filename (str): The filename for the pickle file.
        """
        pd.to_pickle(data, open(os.path.join(self.processed_dir_main, filename), "wb"))

    @staticmethod
    def _get_data_size(input_file_path: str) -> int:
        """
        Get the size of the data from a pickled file.

        Args:
            input_file_path (str): The path to the file.

        Returns:
            int: The size of the data.
        """
        with open(input_file_path, "rb") as f:
            return len(pd.read_pickle(f))

    @property
    def raw_file_names_dict(self) -> dict:
        """
        Returns a dictionary of raw file names used in data processing.

        Returns:
            dict: A dictionary mapping dataset names to their respective file names.
                  For example, {"GO": "go-basic.obo", "SwissUniProt": "uniprot_sprot.dat"}.
        """
        return {"GO": "go-basic.obo", "SwissUniProt": "uniprot_sprot.dat"}

    @property
    def base_dir(self) -> str:
        """
        Returns the base directory path for storing GO-Uniprot data.

        Returns:
            str: The path to the base directory, which is "data/GO_UniProt".
        """
        return os.path.join("data", f"GO_UniProt")

    @property
    def processed_dir_main(self) -> str:
        """
        Returns the main directory path where processed data is stored.

        Returns:
            str: The path to the main processed data directory, based on the base directory and the instance's name.
        """
        return os.path.join(
            self.base_dir,
            self._name,
            "processed",
        )

    @property
    def processed_dir(self) -> str:
        """
        Returns the specific directory path for processed data, including identifiers.

        Returns:
            str: The path to the processed data directory, including additional identifiers.
        """
        return os.path.join(
            self.processed_dir_main,
            *self.identifier,
        )

    @property
    def processed_dir_main_file_names_dict(self) -> dict:
        """
        Returns a dictionary mapping processed data file names.

        Returns:
            dict: A dictionary mapping dataset types to their respective processed file names.
                  For example, {"data": "data.pkl"}.
        """
        return {"data": "data.pkl"}

    @property
    def processed_file_names_dict(self) -> dict:
        """
        Returns a dictionary mapping processed data file names to their final formats.

        Returns:
            dict: A dictionary mapping dataset types to their respective final file names.
                  For example, {"data": "data.pt"}.
        """
        return {"data": "data.pt"}

    @property
    def processed_file_names(self) -> List[str]:
        """
        Returns a list of file names for processed data.

        Returns:
            List[str]: A list of file names corresponding to the processed data.
        """
        return list(self.processed_file_names_dict.values())


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
