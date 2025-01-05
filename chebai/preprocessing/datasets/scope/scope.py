import gzip
import itertools
import os
import pickle
import shutil
from abc import ABC
from collections import OrderedDict
from tempfile import NamedTemporaryFile
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import fastobo
import networkx as nx
import pandas as pd
import requests
import torch
from Bio import SeqIO
from Bio.Seq import Seq

from chebai.preprocessing.datasets.base import _DynamicDataset
from chebai.preprocessing.reader import ProteinDataReader


class _SCOPeDataExtractor(_DynamicDataset, ABC):
    """
    A class for extracting and processing data from the Gene Ontology (GO) dataset and the Swiss UniProt dataset.

    Args:
        dynamic_data_split_seed (int, optional): The seed for random data splitting. Defaults to 42.
        splits_file_path (str, optional): Path to the splits CSV file. Defaults to None.
        max_sequence_length (int, optional): Specifies the maximum allowed sequence length for a protein, with a
        default of 1002. During data preprocessing, any proteins exceeding this length will be excluded from further
        processing.
        **kwargs: Additional keyword arguments passed to DynamicDataset and  XYBaseDataModule.
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

    _SCOPE_GENERAL_URL = "https://scop.berkeley.edu/downloads/parse/dir.{data_type}.scope.{version_number}-stable.txt"
    _PDB_SEQUENCE_DATA_URL = (
        "https://files.rcsb.org/pub/pdb/derived_data/pdb_seqres.txt.gz"
    )

    def __init__(
        self,
        scope_version: float,
        scope_version_train: Optional[float] = None,
        **kwargs,
    ):

        self.scope_version: float = scope_version
        self.scope_version_train: float = scope_version_train

        super(_SCOPeDataExtractor, self).__init__(**kwargs)

        if self.scope_version_train is not None:
            # Instantiate another same class with "scope_version" as "scope_version_train", if train_version is given
            # This is to get the data from respective directory related to "scope_version_train"
            _init_kwargs = kwargs
            _init_kwargs["chebi_version"] = self.scope_version_train
            self._scope_version_train_obj = self.__class__(
                **_init_kwargs,
            )

    @staticmethod
    def _get_scope_url(data_type: str, version_number: float) -> str:
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
        Downloads the required raw data related to Gene Ontology (GO) and Swiss-UniProt dataset.

        Returns:
            str: Path to the downloaded data.
        """
        self._download_pdb_sequence_data()
        return self._download_scope_raw_data()

    def _download_pdb_sequence_data(self) -> None:
        pdb_seq_file_path = os.path.join(self.raw_dir, self.raw_file_names_dict["PDB"])
        os.makedirs(os.path.dirname(pdb_seq_file_path), exist_ok=True)

        if not os.path.isfile(pdb_seq_file_path):
            print(f"Downloading PDB sequence data....")

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
        os.makedirs(self.raw_dir, exist_ok=True)
        for data_type in ["CLA", "COM", "HIE", "DES"]:
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

    def _parse_pdb_sequence_file(self) -> Dict[str, Dict[str, str]]:
        pdb_chain_seq_mapping: Dict[str, Dict[str, str]] = {}
        for record in SeqIO.parse(
            os.path.join(self.raw_dir, self.raw_file_names_dict["PDB"]), "fasta"
        ):
            pdb_id, chain = record.id.split("_")
            pdb_chain_seq_mapping.setdefault(pdb_id, {})[chain] = str(record.seq)
        return pdb_chain_seq_mapping

    def _extract_class_hierarchy(self, data_path: str) -> nx.DiGraph:
        print("Extracting class hierarchy...")

        # Load and preprocess CLA file
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
            "ancestor_nodes",
        ]
        df_cla["sunid"] = pd.to_numeric(
            df_cla["sunid"], errors="coerce", downcast="integer"
        )
        df_cla["ancestor_nodes"] = df_cla["ancestor_nodes"].apply(
            lambda x: {k: int(v) for k, v in (item.split("=") for item in x.split(","))}
        )
        df_cla.set_index("sunid", inplace=True)

        # Load and preprocess HIE file
        df_hie = pd.read_csv(
            os.path.join(self.raw_dir, self.raw_file_names_dict["HIE"]),
            sep="\t",
            header=None,
            comment="#",
        )
        df_hie.columns = ["sunid", "parent_sunid", "children_sunids"]
        df_hie["sunid"] = pd.to_numeric(
            df_hie["sunid"], errors="coerce", downcast="integer"
        )
        df_hie["parent_sunid"] = df_hie["parent_sunid"].replace("-", -1).astype(int)
        df_hie["children_sunids"] = df_hie["children_sunids"].apply(
            lambda x: list(map(int, x.split(","))) if x != "-" else []
        )

        # Initialize directed graph
        g = nx.DiGraph()

        # Add nodes and edges efficiently
        g.add_edges_from(
            df_hie[df_hie["parent_sunid"] != -1].apply(
                lambda row: (row["parent_sunid"], row["sunid"]), axis=1
            )
        )
        g.add_edges_from(
            df_hie.explode("children_sunids")
            .dropna()
            .apply(lambda row: (row["sunid"], row["children_sunids"]), axis=1)
        )

        pdb_chain_seq_mapping = self._parse_pdb_sequence_file()

        node_to_pdb_id = df_cla["PDB_ID"].to_dict()

        for node in g.nodes():
            pdb_id = node_to_pdb_id[node]
            chain_mapping = pdb_chain_seq_mapping.get(pdb_id, {})

            # Add nodes and edges for chains in the mapping
            for chain, sequence in chain_mapping.items():
                chain_node = f"{pdb_id}_{chain}"
                g.add_node(chain_node, sequence=sequence)
                g.add_edge(node, chain_node)

        print("Compute transitive closure...")
        return nx.transitive_closure_dag(g)

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

    # ------------------------------ Phase: Setup data -----------------------------------
    def _load_dict(self, input_file_path: str) -> Generator[Dict[str, Any], None, None]:
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
        return os.path.join("data", "SCOPe", f"version_{self.scope_version}")

    @property
    def raw_file_names_dict(self) -> dict:
        """
        Returns a dictionary of raw file names used in data processing.

        Returns:
            dict: A dictionary mapping dataset names to their respective file names.
                  For example, {"GO": "go-basic.obo", "SwissUniProt": "uniprot_sprot.dat"}.
        """
        return {
            "CLA": "cla.txt",
            "DES": "des.txt",
            "HIE": "hie.txt",
            "COM": "com.txt",
            "PDB": "pdb_sequences.txt",
        }


class SCOPE(_SCOPeDataExtractor):
    READER = ProteinDataReader

    @property
    def _name(self) -> str:
        return "test"

    def select_classes(self, g: nx.DiGraph, *args, **kwargs) -> List:
        pass


if __name__ == "__main__":
    scope = SCOPE(scope_version=2.08)
    scope._parse_pdb_sequence_file()
