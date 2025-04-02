# References for this file :

# Reference 1:
# John-Marc Chandonia, Naomi K Fox, Steven E Brenner, SCOPe: classification of large macromolecular structures
# in the structural classification of proteins—extended database, Nucleic Acids Research, Volume 47,
# Issue D1, 08 January 2019, Pages D475–D481, https://doi.org/10.1093/nar/gky1134
# https://scop.berkeley.edu/about/ver=2.08

# Reference 2:
# Murzin AG, Brenner SE, Hubbard TJP, Chothia C. 1995. SCOP: a structural classification of proteins database for
# the investigation of sequences and structures. Journal of Molecular Biology 247:536-540

import gzip
import os
import re
import shutil
from abc import ABC, abstractmethod
from tempfile import NamedTemporaryFile
from typing import Any, Dict, Generator, List, Optional, Tuple

import networkx as nx
import pandas as pd
import requests
import torch
from Bio import SeqIO

from chebai.preprocessing.datasets.base import _DynamicDataset
from chebai.preprocessing.reader import ProteinDataReader


class _SCOPeDataExtractor(_DynamicDataset, ABC):
    """
    A class for extracting and processing data from the SCOPe (Structural Classification of Proteins - extended) dataset.

    This class is designed to handle the parsing, preprocessing, and hierarchical structure extraction from various
    SCOPe dataset files, such as classification (CLA), hierarchy (HIE), and description (DES) files.
    Additionally, it supports downloading related data like PDB sequence files.

    Args:
        scope_version (str): The SCOPe version to use.
        scope_version_train (Optional[str]): The training SCOPe version, if different.
        dynamic_data_split_seed (int, optional): The seed for random data splitting. Defaults to 42.
        splits_file_path (str, optional): Path to the splits CSV file. Defaults to None.
        **kwargs: Additional keyword arguments passed to DynamicDataset and  XYBaseDataModule.
    """

    # -- Index for columns of processed `data.pkl` (derived from `_graph_to_raw_dataset`)
    # "id" at                 row index 0
    # "sids" at               row index 1
    # "sequence" at           row index 2
    # labels starting from    row index 3
    _ID_IDX: int = 0
    _DATA_REPRESENTATION_IDX: int = 2  # here `sequence` column
    _LABELS_START_IDX: int = 3

    _SCOPE_GENERAL_URL = "https://scop.berkeley.edu/downloads/parse/dir.{data_type}.scope.{version_number}-stable.txt"
    _PDB_SEQUENCE_DATA_URL = (
        "https://files.rcsb.org/pub/pdb/derived_data/pdb_seqres.txt.gz"
    )

    SCOPE_HIERARCHY: Dict[str, str] = {
        "cl": "class",
        "cf": "fold",
        "sf": "superfamily",
        "fa": "family",
        "dm": "protein",
        "sp": "species",
        "px": "domain",
    }

    def __init__(
        self,
        scope_version: str,
        scope_version_train: Optional[str] = None,
        max_sequence_len: int = 1000,
        **kwargs,
    ):
        self.scope_version: str = scope_version
        self.scope_version_train: str = scope_version_train
        self.max_sequence_len: int = max_sequence_len

        super(_SCOPeDataExtractor, self).__init__(**kwargs)

        if self.scope_version_train is not None:
            # Instantiate another same class with "scope_version" as "scope_version_train", if train_version is given
            # This is to get the data from respective directory related to "scope_version_train"
            _init_kwargs = kwargs
            _init_kwargs["scope_version"] = self.scope_version_train
            self._scope_version_train_obj = self.__class__(
                **_init_kwargs,
            )

    @staticmethod
    def _get_scope_url(data_type: str, version_number: str) -> str:
        """
        Generates the URL for downloading SCOPe files.

        Args:
            data_type (str): The type of data (e.g., 'cla', 'hie', 'des').
            version_number (str): The version of the SCOPe file.

        Returns:
            str: The formatted SCOPe file URL.
        """
        return _SCOPeDataExtractor._SCOPE_GENERAL_URL.format(
            data_type=data_type, version_number=version_number
        )

    # ------------------------------ Phase: Prepare data -----------------------------------
    def _download_required_data(self) -> str:
        """
        Downloads the required raw data for SCOPe and PDB sequence datasets.

        Returns:
            str: Path to the downloaded data.
        """
        self._download_pdb_sequence_data()
        return self._download_scope_raw_data()

    def _download_pdb_sequence_data(self) -> None:
        """
        Downloads and unzips the PDB sequence dataset from the RCSB PDB repository.

        The file is downloaded as a temporary gzip file, which is then extracted to the
        specified directory.
        """
        pdb_seq_file_path = os.path.join(
            self.scope_root_dir, self.raw_file_names_dict["PDB"]
        )
        os.makedirs(os.path.dirname(pdb_seq_file_path), exist_ok=True)

        if not os.path.isfile(pdb_seq_file_path):
            print(f"Missing PDB raw data, Downloading PDB sequence data....")

            # Create a temporary file
            with NamedTemporaryFile(delete=False) as tf:
                temp_filename = tf.name
                print(f"Downloading to temporary file {temp_filename}")

                # Download the file
                response = requests.get(self._PDB_SEQUENCE_DATA_URL, stream=True)
                with open(temp_filename, "wb") as temp_file:
                    shutil.copyfileobj(response.raw, temp_file)

                print(f"Downloaded to {temp_filename}")

            # Unpack the gzipped file
            try:
                print(f"Unzipping the file....")
                with gzip.open(temp_filename, "rb") as f_in:
                    output_file_path = pdb_seq_file_path
                    with open(output_file_path, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)
                print(f"Unpacked and saved to {output_file_path}")

            except Exception as e:
                print(f"Failed to unpack the file: {e}")
            finally:
                # Clean up the temporary file
                os.remove(temp_filename)
                print(f"Removed temporary file {temp_filename}")

    def _download_scope_raw_data(self) -> str:
        """
        Downloads the raw SCOPe dataset files (CLA, HIE, DES, and COM).

        Each file is downloaded from the SCOPe repository and saved to the specified directory.
        Files are only downloaded if they do not already exist.

        Returns:
            str: A dummy path to indicate completion (can be extended for custom behavior).
        """
        os.makedirs(self.raw_dir, exist_ok=True)
        for data_type in ["CLA", "HIE", "DES"]:
            data_file_name = self.raw_file_names_dict[data_type]
            scope_path = os.path.join(self.raw_dir, data_file_name)
            if not os.path.isfile(scope_path):
                print(f"Missing Scope: {data_file_name} raw data, Downloading...")
                r = requests.get(
                    self._get_scope_url(data_type.lower(), self.scope_version),
                    allow_redirects=False,
                    verify=False,  # Disable SSL verification
                )
                r.raise_for_status()  # Check if the request was successful
                open(scope_path, "wb").write(r.content)
        return "dummy/path"

    def _extract_class_hierarchy(self, data_path: str) -> nx.DiGraph:
        """
        Extracts the class hierarchy from SCOPe data and computes its transitive closure.

        Args:
            data_path (str): Path to the processed SCOPe dataset.

        Returns:
            nx.DiGraph: A directed acyclic graph representing the SCOPe class hierarchy.
        """
        print("Extracting class hierarchy...")
        df_scope = self._get_scope_data()
        pdb_chain_df = self._parse_pdb_sequence_file()
        pdb_id_set = set(pdb_chain_df["pdb_id"])  # Search time complexity - O(1)

        # Initialize sets and dictionaries for storing edges and attributes
        parent_node_edges, node_child_edges = set(), set()
        node_attrs = {}
        px_level_nodes = set()
        sequence_nodes = dict()
        px_to_seq_edges = set()
        required_graph_nodes = set()

        # Create a lookup dictionary for PDB chain sequences
        lookup_dict = (
            pdb_chain_df.groupby("pdb_id")[["chain_id", "sequence"]]
            .apply(lambda x: dict(zip(x["chain_id"], x["sequence"])))
            .to_dict()
        )

        def add_sequence_nodes_edges(chain_sequence, px_sun_id):
            """Adds sequence nodes and edges connecting px-level nodes to sequence nodes."""
            if chain_sequence not in sequence_nodes:
                sequence_nodes[chain_sequence] = f"seq_{len(sequence_nodes)}"
            px_to_seq_edges.add((px_sun_id, sequence_nodes[chain_sequence]))

        # Step 1: Build the graph structure and store node attributes
        for row in df_scope.itertuples(index=False):
            if row.level == "px":

                pdb_id, chain_id = row.sid[1:5], row.sid[5]

                if pdb_id not in pdb_id_set or chain_id == "_":
                    # Don't add domain level nodes that don't have pdb_id in pdb_sequences.txt file
                    # Also chain_id with "_" which corresponds to no chain
                    continue
                px_level_nodes.add(row.sunid)

                # Add edges between px-level nodes and sequence nodes
                if chain_id != ".":
                    if chain_id not in lookup_dict[pdb_id]:
                        continue
                    add_sequence_nodes_edges(lookup_dict[pdb_id][chain_id], row.sunid)
                else:
                    # If chain_id is '.', connect all chains of this PDB ID
                    for chain, chain_sequence in lookup_dict[pdb_id].items():
                        add_sequence_nodes_edges(chain_sequence, row.sunid)
            else:
                required_graph_nodes.add(row.sunid)

            node_attrs[row.sunid] = {"sid": row.sid, "level": row.level}

            if row.parent_sunid != -1:
                parent_node_edges.add((row.parent_sunid, row.sunid))

            for child_id in row.children_sunids:
                node_child_edges.add((row.sunid, child_id))

        del df_scope, pdb_chain_df, pdb_id_set

        g = nx.DiGraph()
        g.add_nodes_from(node_attrs.items())
        # Note - `add_edges` internally create a node, if a node doesn't exist already
        g.add_edges_from({(p, c) for p, c in parent_node_edges if p in node_attrs})
        g.add_edges_from({(p, c) for p, c in node_child_edges if c in node_attrs})

        seq_nodes = set(sequence_nodes.values())
        g.add_nodes_from([(seq_id, {"level": "sequence"}) for seq_id in seq_nodes])
        g.add_edges_from(
            {
                (px_node, seq_node)
                for px_node, seq_node in px_to_seq_edges
                if px_node in node_attrs and seq_node in seq_nodes
            }
        )

        # Step 2: Count sequence successors for required graph nodes only
        for node in required_graph_nodes:
            num_seq_successors = sum(
                g.nodes[child]["level"] == "sequence"
                for child in nx.descendants(g, node)
            )
            g.nodes[node]["num_seq_successors"] = num_seq_successors

        # Step 3: Remove nodes which are not required before computing transitive closure for better efficiency
        g.remove_nodes_from(px_level_nodes | seq_nodes)

        print("Computing Transitive Closure.........")
        # Transitive closure is not needed in `select_classes` method but is required in _SCOPeOverXPartial
        return nx.transitive_closure_dag(g)

    def _get_scope_data(self) -> pd.DataFrame:
        """
        Merges and preprocesses the SCOPe classification, hierarchy, and description files into a unified DataFrame.

        Returns:
            pd.DataFrame: A DataFrame containing combined SCOPe data with classification and hierarchy details.
        """
        df_cla = self._get_classification_data()
        df_hie = self._get_hierarchy_data()
        df_des = self._get_node_description_data()
        df_hie_with_cla = pd.merge(df_hie, df_cla, how="left", on="sunid")
        df_all = pd.merge(
            df_hie_with_cla,
            df_des.drop(columns=["sid"], axis=1),
            how="left",
            on="sunid",
        )
        return df_all

    def _get_classification_data(self) -> pd.DataFrame:
        """
        Parses and processes the SCOPe CLA (classification) file.

        Returns:
            pd.DataFrame: A DataFrame containing classification details, including hierarchy levels.
        """
        df_cla = pd.read_csv(
            os.path.join(self.raw_dir, self.raw_file_names_dict["CLA"]),
            sep="\t",
            header=None,
            comment="#",
        )
        df_cla.columns = [
            "sid",
            "PDB_ID",
            "description",
            "sccs",
            "sunid",
            "hie_levels",
        ]

        # Convert to dict - {cl:46456, cf:46457, sf:46458, fa:46459, dm:46460, sp:116748, px:113449}
        df_cla["hie_levels"] = df_cla["hie_levels"].apply(
            lambda x: {k: int(v) for k, v in (item.split("=") for item in x.split(","))}
        )

        # Split ancestor_nodes into separate columns and assign values
        for key in self.SCOPE_HIERARCHY.keys():
            df_cla[self.SCOPE_HIERARCHY[key]] = df_cla["hie_levels"].apply(
                lambda x: x[key]
            )

        df_cla["sunid"] = df_cla["sunid"].astype("int64")

        return df_cla

    def _get_hierarchy_data(self) -> pd.DataFrame:
        """
        Parses and processes the SCOPe HIE (hierarchy) file.

        Returns:
            pd.DataFrame: A DataFrame containing hierarchy details, including parent-child relationships.
        """
        df_hie = pd.read_csv(
            os.path.join(self.raw_dir, self.raw_file_names_dict["HIE"]),
            sep="\t",
            header=None,
            comment="#",
            low_memory=False,
        )
        df_hie.columns = ["sunid", "parent_sunid", "children_sunids"]

        # if not parent id, then insert -1
        df_hie["parent_sunid"] = df_hie["parent_sunid"].replace("-", -1).astype(int)
        # convert children ids to list of ids
        df_hie["children_sunids"] = df_hie["children_sunids"].apply(
            lambda x: list(map(int, x.split(","))) if x != "-" else []
        )

        # Ensure the 'sunid' column in both DataFrames has the same type
        df_hie["sunid"] = df_hie["sunid"].astype("int64")
        return df_hie

    def _get_node_description_data(self) -> pd.DataFrame:
        """
        Parses and processes the SCOPe DES (description) file.

        Returns:
            pd.DataFrame: A DataFrame containing node-level descriptions from the SCOPe dataset.
        """
        df_des = pd.read_csv(
            os.path.join(self.raw_dir, self.raw_file_names_dict["DES"]),
            sep="\t",
            header=None,
            comment="#",
            low_memory=False,
        )
        df_des.columns = ["sunid", "level", "scss", "sid", "description"]
        df_des.loc[len(df_des)] = {"sunid": 0, "level": "root"}

        # Ensure the 'sunid' column in both DataFrames has the same type
        df_des["sunid"] = df_des["sunid"].astype("int64")
        return df_des

    def _graph_to_raw_dataset(self, graph: nx.DiGraph) -> pd.DataFrame:
        """
        Processes a directed acyclic graph (DAG) to generate a raw dataset in DataFrame format. This dataset includes
        chain-level sequences and their corresponding labels based on the hierarchical structure of the associated domains.

        The process:
            - Extracts SCOPe domain identifiers (sids) from the graph.
            - Retrieves class labels for each domain based on all applicable taxonomy levels.
            - Fetches the chain-level sequences from the Protein Data Bank (PDB) for each domain.
            - For each sequence, identifies all domains associated with the same chain and assigns their corresponding labels.

        Notes:
            - SCOPe hierarchy levels are used as labels, with each level represented by a column. The value in each column
              indicates whether a PDB chain is associated with that particular hierarchy level.
            - PDB chains are treated as samples. The method considers only domains that are mapped to the selected hierarchy levels.

        Data Format: pd.DataFrame
            - Column 0 : id (Unique identifier for each sequence entry)
            - Column 1 : sids (List of domain identifiers associated with the sequence)
            - Column 2 : sequence (Amino acid sequence of the chain)
            - Column 3 to Column "n": Each column corresponds to a SCOPe class hierarchy level with a value
              of True/False indicating whether the chain is associated with the corresponding level.

        Args:
            graph (nx.DiGraph): The class hierarchy graph.

        Returns:
            pd.DataFrame: The raw dataset created from the graph.

        Raises:
            RuntimeError: If no sunids are selected.
        """
        print(f"Process graph")

        selected_sun_ids_per_lvl = self.select_classes(graph)

        if not selected_sun_ids_per_lvl:
            raise RuntimeError("No sunid selected.")

        df_cla = self._get_classification_data()
        hierarchy_levels = list(self.SCOPE_HIERARCHY.values())
        hierarchy_levels.remove("domain")

        df_cla = df_cla[["sid", "sunid"] + hierarchy_levels]

        # Initialize selected target columns
        df_encoded = df_cla[["sid", "sunid"]].copy()

        # Collect all new columns in a dictionary first (avoids fragmentation)
        encoded_df_columns = {}

        lvl_to_target_cols_mapping = {}
        # Iterate over only the selected sun_ids (nodes) to one-hot encode them
        for level, selected_sun_ids in selected_sun_ids_per_lvl.items():
            level_column = self.SCOPE_HIERARCHY[level]
            if level_column in df_cla.columns:
                # Create binary encoding for only relevant sun_ids
                for sun_id in selected_sun_ids:
                    col_name = f"{level_column}_{sun_id}"
                    encoded_df_columns[col_name] = (
                        df_cla[level_column] == sun_id
                    ).astype(bool)

                    lvl_to_target_cols_mapping.setdefault(level_column, []).append(
                        col_name
                    )

        # Convert the dictionary into a DataFrame and concatenate at once (prevents fragmentation)
        df_encoded = pd.concat([df_encoded, pd.DataFrame(encoded_df_columns)], axis=1)

        encoded_target_columns = []
        for level in hierarchy_levels:
            if level in lvl_to_target_cols_mapping:
                encoded_target_columns.extend(lvl_to_target_cols_mapping[level])

        print(
            f"{len(encoded_target_columns)} labels has been selected for specified threshold, "
        )
        print("Constructing data.pkl file .....")

        df_encoded = df_encoded[["sid", "sunid"] + encoded_target_columns]

        # Filter to select only domains that atleast map to any one selected sunid in any level
        df_encoded = df_encoded[df_encoded.iloc[:, 2:].any(axis=1)]

        df_encoded["pdb_id"] = df_encoded["sid"].str[1:5]
        df_encoded["chain_id"] = df_encoded["sid"].str[5]

        # "_" (underscore) means it has no chain
        df_encoded = df_encoded[df_encoded["chain_id"] != "_"]

        pdb_chain_df = self._parse_pdb_sequence_file()

        # Handle chain_id == "." - Multiple chain case
        # Split df_encoded into two: One for specific chains, one for "multiple chains" (".")
        df_specific_chains = df_encoded[df_encoded["chain_id"] != "."]
        df_multiple_chains = df_encoded[df_encoded["chain_id"] == "."].drop(
            columns=["chain_id"]
        )

        # Merge specific chains normally
        merged_specific = df_specific_chains.merge(
            pdb_chain_df, on=["pdb_id", "chain_id"], how="left"
        )

        # Merge all chains case -> Join by pdb_id (not chain_id)
        merged_all_chains = df_multiple_chains.merge(
            pdb_chain_df, on="pdb_id", how="left"
        )

        # Combine both cases
        sequence_hierarchy_df = pd.concat(
            [merged_specific, merged_all_chains], ignore_index=True
        ).dropna(subset=["sequence"])

        # Vectorized Aggregation Instead of Row-wise Updates
        sequence_hierarchy_df = (
            sequence_hierarchy_df.groupby("sequence", as_index=False)
            .agg(
                {
                    "sid": list,  # Collect all SIDs per sequence
                    **{
                        col: "max" for col in encoded_target_columns
                    },  # Max works as Bitwise OR for labels
                }
            )
            .rename(columns={"sid": "sids"})
        )  # Rename for clarity

        sequence_hierarchy_df = sequence_hierarchy_df.assign(
            id=range(1, len(sequence_hierarchy_df) + 1)
        )[["id", "sids", "sequence"] + encoded_target_columns]

        # Ensure atleast one label is true for each protein sequence
        sequence_hierarchy_df = sequence_hierarchy_df[
            sequence_hierarchy_df.iloc[:, self._LABELS_START_IDX :].any(axis=1)
        ]

        with open(os.path.join(self.processed_dir_main, "classes.txt"), "wt") as fout:
            fout.writelines(str(sun_id) + "\n" for sun_id in encoded_target_columns)

        return sequence_hierarchy_df

    def _parse_pdb_sequence_file(self) -> pd.DataFrame:
        """
        Parses the PDB sequence file and returns a DataFrame containing PDB IDs, chain IDs, and sequences.

        Returns:
            pd.DataFrame: A DataFrame with columns ["pdb_id", "chain_id", "sequence"].
        """
        records = []
        valid_amino_acids = "".join(ProteinDataReader.AA_LETTER)

        for record in SeqIO.parse(
            os.path.join(self.scope_root_dir, self.raw_file_names_dict["PDB"]), "fasta"
        ):

            if not record.seq or len(record.seq) > self.max_sequence_len:
                continue

            pdb_id, chain = record.id.split("_")
            sequence = re.sub(f"[^{valid_amino_acids}]", "X", str(record.seq))

            # Store as a dictionary entry (list of dicts -> DataFrame later)
            records.append(
                {
                    "pdb_id": pdb_id.lower(),
                    "chain_id": chain.lower(),
                    "sequence": sequence,
                }
            )

        # Convert list of dictionaries to a DataFrame
        pdb_chain_df = pd.DataFrame.from_records(records)

        return pdb_chain_df

    @abstractmethod
    def select_classes(self, g: nx.DiGraph, *args, **kwargs) -> Dict[str, List[int]]:
        # Override the return type of the method from superclass
        pass

    # ------------------------------ Phase: Setup data -----------------------------------
    def setup_processed(self) -> None:
        """
        Transform and prepare processed data for the SCOPe dataset.

        Main function of this method is to transform `data.pkl` into a model input data format (`data.pt`),
        ensuring that the data is in a format compatible for input to the model.
        The transformed data must contain the following keys: `ident`, `features`, `labels`, and `group`.
        This method uses a subclass of Data Reader to perform the transformation.

        It will transform the data related to `scope_version_train`, if specified.
        """
        super().setup_processed()

        # Transform the data related to "scope_version_train" to encoded data, if it doesn't exist
        if self.scope_version_train is not None and not os.path.isfile(
            os.path.join(
                self._scope_version_train_obj.processed_dir,
                self._scope_version_train_obj.processed_file_names_dict["data"],
            )
        ):
            print(
                f"Missing encoded data related to train version: {self.scope_version_train}"
            )
            print("Calling the setup method related to it")
            self._scope_version_train_obj.setup()

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
        Loads encoded/transformed data and generates training, validation, and test splits.

        This method first loads encoded data from a file named `data.pt`, which is derived from either
        `scope_version` or `scope_version_train`. It then splits the data into training, validation, and test sets.

        If `scope_version_train` is provided:
            - Loads additional encoded data from `scope_version_train`.
            - Splits this data into training and validation sets, while using the test set from `scope_version`.
            - Prunes the test set from `scope_version` to include only labels that exist in `scope_version_train`.

        If `scope_version_train` is not provided:
            - Splits the data from `scope_version` into training, validation, and test sets without modification.

        Raises:
            FileNotFoundError: If the required `data.pt` file(s) do not exist. Ensure that `prepare_data`
            and/or `setup` methods have been called to generate the dataset files.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple containing three DataFrames:
                - Training set
                - Validation set
                - Test set
        """
        try:
            filename = self.processed_file_names_dict["data"]
            data_scope_version = torch.load(
                os.path.join(self.processed_dir, filename), weights_only=False
            )
        except FileNotFoundError:
            raise FileNotFoundError(
                f"File data.pt doesn't exists. "
                f"Please call 'prepare_data' and/or 'setup' methods to generate the dataset files"
            )

        df_scope_version = pd.DataFrame(data_scope_version)
        train_df_scope_ver, df_test_scope_ver = self.get_test_split(
            df_scope_version, seed=self.dynamic_data_split_seed
        )

        if self.scope_version_train is not None:
            # Load encoded data derived from "scope_version_train"
            try:
                filename_train = (
                    self._scope_version_train_obj.processed_file_names_dict["data"]
                )
                data_scope_train_version = torch.load(
                    os.path.join(
                        self._scope_version_train_obj.processed_dir, filename_train
                    ),
                    weights_only=False,
                )
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"File data.pt doesn't exists related to scope_version_train {self.scope_version_train}."
                    f"Please call 'prepare_data' and/or 'setup' methods to generate the dataset files"
                )

            df_scope_train_version = pd.DataFrame(data_scope_train_version)
            # Get train/val split of data based on "scope_version_train", but
            # using test set from "scope_version"
            df_train, df_val = self.get_train_val_splits_given_test(
                df_scope_train_version,
                df_test_scope_ver,
                seed=self.dynamic_data_split_seed,
            )
            # Modify test set from "scope_version" to only include the labels that
            # exists in "scope_version_train", all other entries remains same.
            df_test = self._setup_pruned_test_set(df_test_scope_ver)
        else:
            # Get all splits based on "scope_version"
            df_train, df_val = self.get_train_val_splits_given_test(
                train_df_scope_ver,
                df_test_scope_ver,
                seed=self.dynamic_data_split_seed,
            )
            df_test = df_test_scope_ver

        return df_train, df_val, df_test

    def _setup_pruned_test_set(
        self, df_test_scope_version: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Create a test set with the same leaf nodes, but use only classes that appear in the training set.

        Args:
            df_test_scope_version (pd.DataFrame): The test dataset.

        Returns:
            pd.DataFrame: The pruned test dataset.
        """
        # TODO: find a more efficient way to do this
        filename_old = "classes.txt"
        # filename_new = f"classes_v{self.scope_version_train}.txt"
        # dataset = torch.load(os.path.join(self.processed_dir, "test.pt"))

        # Load original classes (from the current SCOPe version - scope_version)
        with open(os.path.join(self.processed_dir_main, filename_old), "r") as file:
            orig_classes = file.readlines()

        # Load new classes (from the training SCOPe version - scope_version_train)
        with open(
            os.path.join(
                self._scope_version_train_obj.processed_dir_main, filename_old
            ),
            "r",
        ) as file:
            new_classes = file.readlines()

        # Create a mapping which give index of a class from scope_version, if the corresponding
        # class exists in scope_version_train, Size = Number of classes in scope_version
        mapping = [
            None if or_class not in new_classes else new_classes.index(or_class)
            for or_class in orig_classes
        ]

        # Iterate over each data instance in the test set which is derived from scope_version
        for _, row in df_test_scope_version.iterrows():
            # Size = Number of classes in scope_version_train
            new_labels = [False for _ in new_classes]
            for ind, label in enumerate(row["labels"]):
                # If the scope_version class exists in the scope_version_train and has a True label,
                # set the corresponding label in new_labels to True
                if mapping[ind] is not None and label:
                    new_labels[mapping[ind]] = label
            # Update the labels from test instance from scope_version to the new labels, which are compatible to both versions
            row["labels"] = new_labels

        return df_test_scope_version

    # ------------------------------ Phase: Raw Properties -----------------------------------
    @property
    def scope_root_dir(self) -> str:
        """
        Returns the root directory of scope data.

        Returns:
            str: The path to the base directory, which is "data/GO_UniProt".
        """
        return os.path.join("data", "SCOPe")

    @property
    def base_dir(self) -> str:
        """
        Returns the base directory path for storing SCOPe data.

        Returns:
            str: The path to the base directory, which is "data/GO_UniProt".
        """
        return os.path.join(self.scope_root_dir, f"version_{self.scope_version}")

    @property
    def raw_file_names_dict(self) -> dict:
        """
        Returns a dictionary of raw file names used in data processing.

        Returns:
            dict: A dictionary mapping dataset names to their respective file names.
        """
        return {
            "CLA": "cla.txt",
            "DES": "des.txt",
            "HIE": "hie.txt",
            "PDB": "pdb_sequences.txt",
        }


class _SCOPeOverX(_SCOPeDataExtractor, ABC):
    """
    A class for extracting data from the SCOPe dataset with a threshold for selecting classes/labels based on
    the number of subclasses.

    This class is designed to filter SCOPe classes/labels based on a specified threshold, selecting only those classes
    which have a certain number of subclasses in the hierarchy.

    Attributes:
        READER (dr.ProteinDataReader): The reader used for reading the dataset.
        THRESHOLD (int): The threshold for selecting classes/labels based on the number of subclasses.

    """

    READER = ProteinDataReader
    THRESHOLD: int = None

    @property
    def _name(self) -> str:
        """
        Returns the name of the dataset.

        Returns:
            str: The dataset name, formatted with the current threshold.
        """
        return f"SCOPe{self.THRESHOLD}"

    def select_classes(self, g: nx.DiGraph, *args, **kwargs) -> Dict[str, List[int]]:
        """
        Selects classes from the SCOPe dataset based on the number of successors meeting a specified threshold.

        This method iterates over the nodes in the graph, counting the number of successors for each node.
        Nodes with a number of successors greater than or equal to the defined threshold are selected.

        Note:
            The input graph must be transitive closure of a directed acyclic graph.

        Args:
            g (nx.Graph): The graph representing the dataset.
            *args: Additional positional arguments (not used).
            **kwargs: Additional keyword arguments (not used).

        Returns:
            Dict: A dict containing selected nodes at each hierarchy level.

        Notes:
            - The `THRESHOLD` attribute should be defined in the subclass of this class.
        """
        selected_sunids_for_level = {}
        for node, attr_dict in g.nodes(data=True):
            if attr_dict["level"] in {"root", "px", "sequence"}:
                # Skip nodes with level "root", "px", or "sequence"
                continue

            # Check if the number of "sequence"-level successors meets or exceeds the threshold
            if g.nodes[node]["num_seq_successors"] >= self.THRESHOLD:
                selected_sunids_for_level.setdefault(attr_dict["level"], []).append(
                    node
                )
        return selected_sunids_for_level


class _SCOPeOverXPartial(_SCOPeOverX, ABC):
    """
    Dataset that doesn't use the full SCOPe dataset, but extracts a part of SCOPe (subclasses of a given top class)

    Attributes:
        top_class_sunid (int): The Sun-ID of the top class from which to extract subclasses.
    """

    def __init__(self, top_class_sunid: int, **kwargs):
        """
        Initializes the _SCOPeOverXPartial dataset.

        Args:
            top_class_sunid (int): The Sun-ID of the top class from which to extract subclasses.
            **kwargs: Additional keyword arguments passed to the superclass initializer.
        """
        if "top_class_sunid" not in kwargs:
            kwargs["top_class_sunid"] = top_class_sunid

        self.top_class_sunid: int = top_class_sunid
        super().__init__(**kwargs)

    @property
    def processed_dir_main(self) -> str:
        """
        Returns the main processed data directory specific to the top class.

        Returns:
            str: The processed data directory path.
        """
        return os.path.join(
            self.base_dir,
            self._name,
            f"partial_{self.top_class_sunid}",
            "processed",
        )

    def _extract_class_hierarchy(self, data_path: str) -> nx.DiGraph:
        """
        Extracts a subset of SCOPe based on subclasses of the top class ID.

        This method calls the superclass method to extract the full class hierarchy,
        then extracts the subgraph containing only the descendants of the top class ID, including itself.

        Args:
            data_path (str): The file path to the SCOPe ontology file.

        Returns:
            nx.DiGraph: The extracted class hierarchy as a directed graph, limited to the
            descendants of the top class ID.
        """
        g = super()._extract_class_hierarchy(data_path)
        g = g.subgraph(
            list(g.successors(self.top_class_sunid)) + [self.top_class_sunid]
        )
        return g


class SCOPeOver2000(_SCOPeOverX):
    """
    A class for extracting data from the SCOPe dataset with a threshold of 2000 for selecting classes.

    Inherits from `_SCOPeOverX` and sets the threshold for selecting classes to 2000.

    Attributes:
        THRESHOLD (int): The threshold for selecting classes (2000).
    """

    THRESHOLD: int = 2000


class SCOPeOver50(_SCOPeOverX):

    THRESHOLD = 50


class SCOPeOverPartial2000(_SCOPeOverXPartial):
    """
    A class for extracting data from the SCOPe dataset with a threshold of 2000 for selecting classes.

    Inherits from `_SCOPeOverXPartial` and sets the threshold for selecting classes to 2000.

    Attributes:
        THRESHOLD (int): The threshold for selecting classes (2000).
    """

    THRESHOLD: int = 2000


if __name__ == "__main__":
    scope = SCOPeOver50(scope_version="2.08")

    # g = scope._extract_class_hierarchy("dummy/path")
    # # Save graph
    # import pickle
    # with open("graph.gpickle", "wb") as f:
    #     pickle.dump(g, f)

    # Load graph
    import pickle

    with open("graph.gpickle", "rb") as f:
        g = pickle.load(f)

    # print(len([node for node in g.nodes() if g.out_degree(node) > 10000]))
    scope._graph_to_raw_dataset(g)
