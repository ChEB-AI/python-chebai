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

    # -- Index for columns of processed `data.pkl` (derived from `_get_swiss_to_go_mapping` & `_graph_to_raw_dataset`
    # "swiss_id" at           row index 0
    # "accession" at          row index 1
    # "go_ids" at             row index 2
    # "sequence" at           row index 3
    # labels starting from    row index 4
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
            _init_kwargs["scope_version"] = self.scope_version_train
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

    def _extract_class_hierarchy(self, data_path: str) -> nx.DiGraph:
        print("Extracting class hierarchy...")
        df_scope = self._get_scope_data()

        g = nx.DiGraph()

        egdes = []
        for _, row in df_scope.iterrows():
            g.add_node(row["sunid"], **{"sid": row["sid"], "level": row["level"]})
            if row["parent_sunid"] != -1:
                egdes.append((row["parent_sunid"], row["sunid"]))

            for children_id in row["children_sunids"]:
                egdes.append((row["sunid"], children_id))

        g.add_edges_from(egdes)

        print("Computing transitive closure")
        return nx.transitive_closure_dag(g)

    def _get_scope_data(self) -> pd.DataFrame:
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
        # Load and preprocess HIE file
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

    def _get_node_description_data(self):
        # Load and preprocess HIE file
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
        print(f"Process graph")

        sids = nx.get_node_attributes(graph, "sid")
        levels = nx.get_node_attributes(graph, "level")

        sun_ids = {}
        sids_list = []

        selected_sids_dict = self.select_classes(graph)

        for sun_id, level in levels.items():
            if sun_id in selected_sids_dict:
                sun_ids.setdefault(level, []).append(sun_id)
                sids_list.append(sids.get(sun_id))

        # Remove root node, as it will True for all instances
        sun_ids.pop("root", None)

        # data_df = pd.DataFrame(OrderedDict(sun_id=sun_ids, sids=sids_list))
        df_cla = self._get_classification_data()

        for level, selected_sun_ids in sun_ids.items():
            df_cla = df_cla[df_cla[self.SCOPE_HIERARCHY[level]].isin(selected_sun_ids)]

        assert (
            len(df_cla) > 1
        ), "dataframe should have more than one instance for `pd.get_dummies` to work as expected"
        df_encoded = pd.get_dummies(
            df_cla,
            columns=list(self.SCOPE_HIERARCHY.values()),
            drop_first=False,
            sparse=True,
        )

        pdb_chain_seq_mapping = self._parse_pdb_sequence_file()

        encoded_target_cols = {}
        for col in self.SCOPE_HIERARCHY.values():
            encoded_target_cols[col] = [
                t_col for t_col in df_encoded.columns if t_col.startswith(col)
            ]

        encoded_target_columns = []
        for level in self.SCOPE_HIERARCHY.values():
            encoded_target_columns.extend(encoded_target_cols[level])

        sequence_hierarchy_df = pd.DataFrame(columns=["sids"] + encoded_target_columns)

        for _, row in df_encoded.iterrows():
            sid = row["sid"]
            # SID: 7-char identifier ("d" + 4-char PDB ID + chain ID ('_' for none, '.' for multiple)
            # + domain specifier ('_' if not needed))
            assert len(sid) == 7, "sid should have 7 characters"
            pdb_id, chain_id = sid[1:5], sid[5]

            pdb_to_chain_mapping = pdb_chain_seq_mapping.get(pdb_id, None)
            if not pdb_to_chain_mapping:
                continue

            if chain_id != "_":
                chain_sequence = pdb_to_chain_mapping.get(chain_id, None)
                if chain_sequence:
                    self._update_or_add_sequence(
                        chain_sequence, row, sequence_hierarchy_df, encoded_target_cols
                    )

            else:
                # Add nodes and edges for chains in the mapping
                for chain, chain_sequence in pdb_to_chain_mapping.items():
                    self._update_or_add_sequence(
                        chain_sequence, row, sequence_hierarchy_df, encoded_target_cols
                    )

        sequence_hierarchy_df.drop(columns=["sid"], axis=1, inplace=True)
        sequence_hierarchy_df.reset_index(inplace=True)
        sequence_hierarchy_df["id"] = range(1, len(sequence_hierarchy_df) + 1)

        sequence_hierarchy_df = sequence_hierarchy_df[
            ["id", "sids", "sequence"] + encoded_target_columns
        ]

        # This filters the DataFrame to include only the rows where at least one value in the row from 5th column
        # onwards is True/non-zero.
        sequence_hierarchy_df = sequence_hierarchy_df[
            sequence_hierarchy_df.iloc[:, self._LABELS_START_IDX :].any(axis=1)
        ]
        return sequence_hierarchy_df

    def _parse_pdb_sequence_file(self) -> Dict[str, Dict[str, str]]:
        pdb_chain_seq_mapping: Dict[str, Dict[str, str]] = {}
        for record in SeqIO.parse(
            os.path.join(self.raw_dir, self.raw_file_names_dict["PDB"]), "fasta"
        ):
            pdb_id, chain = record.id.split("_")
            if str(record.seq):
                pdb_chain_seq_mapping.setdefault(pdb_id.lower(), {})[chain.lower()] = (
                    str(record.seq)
                )
        return pdb_chain_seq_mapping

    @staticmethod
    def _update_or_add_sequence(
        sequence, row, sequence_hierarchy_df, encoded_col_names
    ):
        if sequence in sequence_hierarchy_df.index:
            # Update encoded columns only if they are True
            for col in encoded_col_names:
                assert (
                    sum(row[encoded_col_names[col]].tolist()) == 1
                ), "A instance can belong to only one hierarchy level"
                sliced_data = row[
                    encoded_col_names[col]
                ]  # Slice starting from the second column (index 1)
                # Get the column name with the True value
                true_column = sliced_data.idxmax() if sliced_data.any() else None
                sequence_hierarchy_df.loc[sequence, true_column] = True

            sequence_hierarchy_df.loc[sequence, "sids"].append(row["sid"])

        else:
            # Add new row with sequence as the index and hierarchy data
            new_row = row
            new_row["sids"] = [row["sid"]]
            sequence_hierarchy_df.loc[sequence] = new_row

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
    THRESHOLD = 10000

    @property
    def _name(self) -> str:
        return "test"

    def select_classes(self, g: nx.DiGraph, *args, **kwargs) -> Dict:
        # Filter nodes and create a dictionary of node and out-degree
        sun_ids_dict = {
            node: g.out_degree(node)  # Store node and its out-degree
            for node in g.nodes
            if g.out_degree(node) >= self.THRESHOLD
        }

        # Return a sorted dictionary (by out-degree or node id)
        sorted_dict = dict(
            sorted(sun_ids_dict.items(), key=lambda item: item[0], reverse=False)
        )

        filename = "classes.txt"
        with open(os.path.join(self.processed_dir_main, filename), "wt") as fout:
            fout.writelines(str(sun_id) + "\n" for sun_id in sorted_dict.keys())

        return sorted_dict


if __name__ == "__main__":
    scope = SCOPE(scope_version=2.08)
    g = scope._extract_class_hierarchy("d")
    scope._graph_to_raw_dataset(g)
