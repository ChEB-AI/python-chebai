# Reference for this file :
# Maxat Kulmanov, Mohammed Asif Khan, Robert Hoehndorf;
# DeepGO: Predicting protein functions from sequence and interactions
# using a deep ontology-aware classifier, Bioinformatics, 2017.
# https://doi.org/10.1093/bioinformatics/btx624
# Github: https://github.com/bio-ontology-research-group/deepgo
# https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/docs/keywlist.txt
# https://www.ebi.ac.uk/GOA/downloads


__all__ = ["GOUniprotDataModule"]

import gzip
import os
from abc import ABC, abstractmethod
from collections import OrderedDict
from tempfile import NamedTemporaryFile, TemporaryDirectory, gettempdir
from typing import Any, Dict, Generator, List
from urllib import request

import fastobo
import networkx as nx
import pandas as pd
import requests
import torch
from Bio import SwissProt
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

from chebai.preprocessing.datasets import XYBaseDataModule


class _GOUniprotDataExtractor(XYBaseDataModule, ABC):
    _GO_DATA_INIT = "GO"

    def __init__(self):
        pass

    @property
    def dynamic_split_dfs(self) -> Dict[str, pd.DataFrame]:
        """
        Property to retrieve dynamic train, validation, and test splits.

        This property checks if dynamic data splits (`dynamic_df_train`, `dynamic_df_val`, `dynamic_df_test`)
        are already loaded. If any of them is None, it either generates them dynamically or retrieves them
        from data file with help of pre-existing Split csv file (`splits_file_path`) containing splits assignments.

        Returns:
            dict: A dictionary containing the dynamic train, validation, and test DataFrames.
                Keys are 'train', 'validation', and 'test'.
        """
        if any(
            split is None
            for split in [
                self.dynamic_df_test,
                self.dynamic_df_val,
                self.dynamic_df_train,
            ]
        ):
            if self.splits_file_path is None:
                # Generate splits based on given seed, create csv file to records the splits
                self._generate_dynamic_splits()
            else:
                # If user has provided splits file path, use it to get the splits from the data
                self._retrieve_splits_from_csv()
        return {
            "train": self.dynamic_df_train,
            "validation": self.dynamic_df_val,
            "test": self.dynamic_df_test,
        }

    def _generate_dynamic_splits(self) -> None:
        """
        Generate data splits during runtime and save them in class variables.

        This method loads encoded data derived from either `chebi_version` or `chebi_version_train`
        and generates train, validation, and test splits based on the loaded data.
        If `chebi_version_train` is specified, the test set is pruned to include only labels that
        exist in `chebi_version_train`.

        Raises:
            FileNotFoundError: If the required data file (`data.pt`) for either `chebi_version` or `chebi_version_train`
                               does not exist. It advises calling `prepare_data` or `setup` methods to generate
                               the dataset files.
        """
        print("Generate dynamic splits...")
        # Load encoded data derived from "chebi_version"
        try:
            filename = self.processed_file_names_dict["data"]
            data_chebi_version = torch.load(os.path.join(self.processed_dir, filename))
        except FileNotFoundError:
            raise FileNotFoundError(
                f"File data.pt doesn't exists. "
                f"Please call 'prepare_data' and/or 'setup' methods to generate the dataset files"
            )

        df_chebi_version = pd.DataFrame(data_chebi_version)
        train_df_chebi_ver, df_test_chebi_ver = self.get_test_split(
            df_chebi_version, seed=self.dynamic_data_split_seed
        )

        if self.chebi_version_train is not None:
            # Load encoded data derived from "chebi_version_train"
            try:
                filename_train = (
                    self._chebi_version_train_obj.processed_file_names_dict["data"]
                )
                data_chebi_train_version = torch.load(
                    os.path.join(
                        self._chebi_version_train_obj.processed_dir, filename_train
                    )
                )
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"File data.pt doesn't exists related to chebi_version_train {self.chebi_version_train}."
                    f"Please call 'prepare_data' and/or 'setup' methods to generate the dataset files"
                )

            df_chebi_train_version = pd.DataFrame(data_chebi_train_version)
            # Get train/val split of data based on "chebi_version_train", but
            # using test set from "chebi_version"
            df_train, df_val = self.get_train_val_splits_given_test(
                df_chebi_train_version,
                df_test_chebi_ver,
                seed=self.dynamic_data_split_seed,
            )
            # Modify test set from "chebi_version" to only include the labels that
            # exists in "chebi_version_train", all other entries remains same.
            df_test = self._setup_pruned_test_set(df_test_chebi_ver)
        else:
            # Get all splits based on "chebi_version"
            df_train, df_val = self.get_train_val_splits_given_test(
                train_df_chebi_ver,
                df_test_chebi_ver,
                seed=self.dynamic_data_split_seed,
            )
            df_test = df_test_chebi_ver

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
        self.dynamic_df_train = df_train
        self.dynamic_df_val = df_val
        self.dynamic_df_test = df_test

    def _retrieve_splits_from_csv(self) -> None:
        """
        Retrieve previously saved data splits from splits.csv file or from provided file path.

        This method loads the splits.csv file located at `self.splits_file_path`.
        It then loads the encoded data (`data.pt`) derived from `chebi_version` and filters
        it based on the IDs retrieved from splits.csv to reconstruct the train, validation,
        and test splits.
        """
        print(f"Loading splits from {self.splits_file_path}...")
        splits_df = pd.read_csv(self.splits_file_path)

        filename = self.processed_file_names_dict["data"]
        data_chebi_version = torch.load(os.path.join(self.processed_dir, filename))
        df_chebi_version = pd.DataFrame(data_chebi_version)

        train_ids = splits_df[splits_df["split"] == "train"]["id"]
        validation_ids = splits_df[splits_df["split"] == "validation"]["id"]
        test_ids = splits_df[splits_df["split"] == "test"]["id"]

        self.dynamic_df_train = df_chebi_version[
            df_chebi_version["ident"].isin(train_ids)
        ]
        self.dynamic_df_val = df_chebi_version[
            df_chebi_version["ident"].isin(validation_ids)
        ]
        self.dynamic_df_test = df_chebi_version[
            df_chebi_version["ident"].isin(test_ids)
        ]

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
        Use test set (e.g., loaded from another chebi version or generated in get_test_split), to avoid overlap

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
        # ---- list comprehension degrades performance, dataframe operations are faster
        # mask = [trainval_id not in test_ids for trainval_id in df_trainval["ident"]]
        # df_trainval = df_trainval[mask]
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

    def setup_processed(self):
        print("Transform data")
        os.makedirs(self.processed_dir, exist_ok=True)

        processed_name = self.processed_file_names_dict["data"]
        if not os.path.isfile(
            os.path.join(self.processed_dir, self.processed_dir_file_names["data"])
        ):
            print("Missing transformed `data.pt` file. Transforming data.... ")

            torch.save(
                self._load_data_from_file(
                    os.path.join(
                        self.processed_dir_main,
                        self.processed_dir_main_file_names["data"],
                    )
                ),
                os.path.join(self.processed_dir, self.processed_file_names["data"]),
            )

    def _load_dict(self, input_file_path: str) -> Generator[Dict[str, Any], None, None]:
        """
        Loads a dictionary from a pickled file, yielding individual dictionaries for each row.

        Args:
            input_file_path (str): The path to the file.

        Yields:
            Dict[str, Any]: The dictionary, keys are `features`, `labels` and `ident`.
        """
        with open(input_file_path, "rb") as input_file:
            df = pd.read_pickle(input_file)
            for row in df.values:
                yield dict(features=row[2], labels=row[1], ident=row[0])

    def prepare_data(self) -> None:
        print("Checking for processed data in", self.processed_dir_main)

        if not os.path.isfile(
            self.processed_dir_main, self.processed_dir_main_names_dict["GO"]
        ):
            print("Missing Gene Ontology processed data")
            os.makedirs(self.processed_dir_main, exist_ok=True)
            # swiss_path = self._download_swiss_uni_prot_data()

            go_path = self._download_gene_ontology_data()
            g = self._extract_go_class_hierarchy(go_path)
            data_df = self._graph_to_raw_dataset(g)
            self.save_processed(data_df, self.processed_dir_main_file_names["data"])

    @abstractmethod
    def select_classes(self, g, *args, **kwargs):
        raise NotImplementedError

    def _graph_to_raw_dataset(self, g: nx.DiGraph) -> pd.DataFrame:
        """
        Preparation step before creating splits, uses the graph created by _extract_go_class_hierarchy().

        Args:
            g (nx.DiGraph): The class hierarchy graph.

        Returns:
            pd.DataFrame: The raw dataset created from the graph.
        """
        names = nx.get_node_attributes(g, "name")
        ids = nx.get_node_attributes(g, "id")
        go_to_swiss_mapping = self._get_go_swiss_data_mapping()

        print(f"Processing graph")

        terms = list(g.nodes)
        data = OrderedDict(id=terms)

        data_list = []
        for node in terms:
            data_list.append(
                (
                    names.get(node),
                    ids.get(node),
                    go_to_swiss_mapping.get(ids.get(node))["sequence"],
                    go_to_swiss_mapping.get(ids.get(node))["swiss_ident"],
                )
            )

        names_list, ids_list, sequences_list, swiss_identifier_list = zip(*data_list)

        data["go_id"] = ids_list
        data["name"] = names_list
        data["sequence"] = sequences_list
        data["swiss_ident"] = swiss_identifier_list

        # Assuming select_classes is implemented and returns a list of class IDs
        for n in self.select_classes(g):
            data[n] = [((n in g.predecessors(node)) or (n == node)) for node in terms]

        return pd.DataFrame(data)

    def _get_go_swiss_data_mapping(self) -> Dict[int : Dict[str:str]]:
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
            open(self.raw_file_names_dict["SwissUniProt"], "r")
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
        elements = []
        for term in fastobo.load(go_path):
            if isinstance(term, fastobo.typedef.TypedefFrame):
                # To avoid term frame of the below format/structure
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
        for n in elements:
            g.add_node(n["id"], **n)
        g.add_edges_from([(p, q["id"]) for q in elements for p in q["parents"]])

        print("Compute transitive closure")
        g = nx.transitive_closure_dag(g)
        # g = g.subgraph(list(nx.descendants(g, self.top_class_id)) + [self.top_class_id])
        return g

    @staticmethod
    def term_callback(term: fastobo.term.TermFrame) -> dict:
        """
        Extracts information from a Gene Ontology (GO) term document.

        Args:
            term: A Gene Ontology term Frame document.

        Returns:
            dict: A dictionary containing the extracted information:
                - "id": The ID of the GO term.
                - "parents": A list of parent term IDs.
                - "name": The name of the GO term.
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
            url = f"http://purl.obolibrary.org/obo/go/go-basic.obo"
            r = requests.get(url, allow_redirects=True)
            r.raise_for_status()  # Check if the request was successful
            open(go_path, "wb").write(r.content)
        return go_path

    def _download_swiss_uni_prot_data(self) -> str:
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
        temp_dir = gettempdir()

        if not os.path.isfile(uni_prot_file_path):
            print(f"Downloading Swiss UniProt data....")
            url = f"https://ftp.uniprot.org/pub/databases/uniprot/knowledgebase/complete/uniprot_sprot.dat.gz"
            # TODO : Permission error, manually extracted the data as of now
            temp_file_path = os.path.join(temp_dir, "uniprot_sprot.dat.gz")
            try:
                # Download the gzip file
                request.urlretrieve(url, temp_file_path)
                print(f"Downloaded to temporary file: {temp_file_path}")

                # Extract the gzip file
                with gzip.open(temp_file_path, "rb") as gfile:
                    file_content = gfile.read()
                    print("Extracted the content from the gzip file.")

                # Decode and write the contents to the target file
                with open(uni_prot_file_path, "wt", encoding="utf-8") as fout:
                    fout.write(file_content.decode("utf-8"))
                    print(f"Data written to: {uni_prot_file_path}")

            except PermissionError as e:
                print(f"PermissionError: {e}")
                return None
            except Exception as e:
                print(f"An error occurred: {e}")
                return None
            finally:
                # Clean up the temporary file
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
                    print(f"Temporary file {temp_file_path} removed.")

        return uni_prot_file_path

    def select_classes(self, g, split_name, *args, **kwargs):
        raise NotImplementedError

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
        return {"GO": "go-basic.obo", "SwissUniProt": "uniprot_sprot.dat"}

    @property
    def base_dir(self):
        return os.path.join("data", f"Go_UniProt")

    @property
    def processed_dir_main(self):
        return os.path.join(
            self.base_dir,
            self._name,
            "processed",
        )

    @property
    def processed_dir_main_file_names(self) -> dict:
        return {"data": "data.pkl"}

    @property
    def processed_file_names(self) -> dict:
        return {"data": "data.pt"}


class GOUniprotDataModule(_GOUniprotDataExtractor):
    @property
    def _name(self):
        return f"GoUniProt_v1"
