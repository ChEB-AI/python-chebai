__all__ = [
    "JCIData",
    "JCIExtendedTokenData",
    "JCIExtendedBPEData",
    "JCIExtSelfies",
    "JCITokenData",
    "ChEBIOver100",
    "JCI_500_COLUMNS",
    "JCI_500_COLUMNS_INT",
]

import os
import pickle
from abc import ABC
from collections import OrderedDict
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import fastobo
import networkx as nx
import pandas as pd
import requests
import torch
from iterstrat.ml_stratifiers import (
    MultilabelStratifiedKFold,
    MultilabelStratifiedShuffleSplit,
)
from rdkit import Chem, RDLogger
from tqdm import tqdm

from chebai.preprocessing import reader as dr
from chebai.preprocessing.datasets.base import XYBaseDataModule
import random

# Suppress RDKit warnings and errors
RDLogger.DisableLog('rdApp.*')  # Disable all RDKit logging

# exclude some entities from the dataset because the violate disjointness axioms
CHEBI_BLACKLIST = [
    194026,
    144321,
    156504,
    167175,
    167174,
    167178,
    183506,
    74635,
    3311,
    190439,
    92386,
]


class JCIBase(XYBaseDataModule):
    LABEL_INDEX = 2
    SMILES_INDEX = 1

    @property
    def _name(self):
        return "JCI"

    @property
    def processed_file_names(self):
        return ["test.pt", "train.pt", "validation.pt"]

    def download(self):
        pass

    @property
    def raw_file_names(self):
        return ["test.pkl", "train.pkl", "validation.pkl"]

    def prepare_data(self, *args, **kwargs):
        print("Check for raw data in", self.raw_dir)
        if any(
            not os.path.isfile(os.path.join(self.raw_dir, f))
            for f in self.raw_file_names
        ):
            raise ValueError("Raw data is missing")

    @staticmethod
    def _load_tuples(input_file_path):
        with open(input_file_path, "rb") as input_file:
            for row in pickle.load(input_file).values:
                yield row[1], row[2:].astype(bool), row[0]

    @staticmethod
    def _get_data_size(input_file_path):
        with open(input_file_path, "rb") as f:
            return len(pickle.load(f))

    def setup_processed(self):
        print("Transform splits")
        os.makedirs(self.processed_dir, exist_ok=True)
        for k in ["test", "train", "validation"]:
            print("transform", k)
            torch.save(
                self._load_data_from_file(os.path.join(self.raw_dir, f"{k}.pkl")),
                os.path.join(self.processed_dir, f"{k}.pt"),
            )

    @property
    def label_number(self):
        return 500


class JCIData(JCIBase):
    READER = dr.OrdReader


class JCISelfies(JCIBase):
    READER = dr.SelfiesReader


class JCITokenData(JCIBase):
    READER = dr.ChemDataReader


class _ChEBIDataExtractor(XYBaseDataModule, ABC):
    """
    A class for extracting and processing data from the ChEBI dataset.

    Args:
        chebi_version_train (int, optional): The version of ChEBI to use for training and validation. If not set,
            chebi_version will be used for training, validation and test. Defaults to None.
        single_class (int, optional): The ID of the single class to predict. If not set, all available labels will be
            predicted. Defaults to None.
        dynamic_data_split_seed (int, optional): The seed for random data splitting. Defaults to 42.
        splits_file_path (str, optional): Path to the splits CSV file. Defaults to None.
        **kwargs: Additional keyword arguments (passed to XYBaseDataModule).

    Attributes:
        single_class (Optional[int]): The ID of the single class to predict.
        chebi_version_train (Optional[int]): The version of ChEBI to use for training and validation.
        dynamic_data_split_seed (int): The seed for random data splitting, default is 42.
        dynamic_df_train (Optional[pd.DataFrame]): DataFrame to store the training data split.
        dynamic_df_test (Optional[pd.DataFrame]): DataFrame to store the test data split.
        dynamic_df_val (Optional[pd.DataFrame]): DataFrame to store the validation data split.
        splits_file_path (Optional[str]): Path to csv file containing split assignments.
    """

    def __init__(
            self,
            chebi_version_train: Optional[int] = None,
            single_class: Optional[int] = None,
            aug_data: Optional[bool] = False,
            augment_data_batch_size:Optional[int]= 10000,
            num_smiles_variations:Optional[int]=5,
            **kwargs,
    ):
        # predict only single class (given as id of one of the classes present in the raw data set)
        self.single_class = single_class
        super(_ChEBIDataExtractor, self).__init__(**kwargs)
        # use different version of chebi for training and validation (if not None)
        # (still uses self.chebi_version for test set)
        self.chebi_version_train = chebi_version_train
        self.dynamic_data_split_seed = int(kwargs.get("seed", 42))  # default is 42
        # Class variables to store the dynamics splits
        self.dynamic_df_train = None
        self.dynamic_df_test = None
        self.dynamic_df_val = None
        self.aug_data = aug_data
        self.augment_data_batch_size=augment_data_batch_size
        self.num_smiles_variations=num_smiles_variations

        if self.chebi_version_train is not None:
            # Instantiate another same class with "chebi_version" as "chebi_version_train", if train_version is given
            # This is to get the data from respective directory related to "chebi_version_train"
            _init_kwargs = kwargs
            _init_kwargs["chebi_version"] = self.chebi_version_train
            self._chebi_version_train_obj = self.__class__(
                single_class=self.single_class,
                **_init_kwargs,
            )
        # Path of csv file which contains a list of chebi ids & their assignment to a dataset (either train, validation or test).
        self.splits_file_path = self._validate_splits_file_path(
            kwargs.get("splits_file_path", None)
        )

    @staticmethod
    def _validate_splits_file_path(splits_file_path: Optional[str]) -> Optional[str]:
        """
        Validates the provided splits file path.

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

    def extract_class_hierarchy(self, chebi_path: str) -> nx.DiGraph:
        """
        Extracts the class hierarchy from the ChEBI ontology.

        Args:
            chebi_path (str): The path to the ChEBI ontology.

        Returns:
            nx.DiGraph: The class hierarchy.
        """
        with open(chebi_path, encoding="utf-8") as chebi:
            chebi = "\n".join(l for l in chebi if not l.startswith("xref:"))
        elements = [
            term_callback(clause)
            for clause in fastobo.loads(chebi)
            if clause and ":" in str(clause.id)
        ]
        g = nx.DiGraph()
        for n in elements:
            g.add_node(n["id"], **n)
        g.add_edges_from([(p, q["id"]) for q in elements for p in q["parents"]])
        print("Compute transitive closure")
        return nx.transitive_closure_dag(g)

    def select_classes(self, g, split_name, *args, **kwargs):
        raise NotImplementedError

    def graph_to_raw_dataset(
        self, g: nx.DiGraph, split_name: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Preparation step before creating splits, uses graph created by extract_class_hierarchy(),
        split_name is only relevant, if a separate train_version is set.

        Args:
            g (nx.DiGraph): The class hierarchy graph.
            split_name (Optional[str], optional): Name of the split. Defaults to None.

        Returns:
            pd.DataFrame: The raw dataset created from the graph.
        """
        smiles = nx.get_node_attributes(g, "smiles")
        names = nx.get_node_attributes(g, "name")

        print(f"Process graph")

        molecules, smiles_list = zip(
            *(
                (n, smiles)
                for n, smiles in ((n, smiles.get(n)) for n in smiles.keys())
                if smiles
            )
        )
        data = OrderedDict(id=molecules)
        data["name"] = [names.get(node) for node in molecules]
        data["SMILES"] = smiles_list
        for n in self.select_classes(g, split_name):
            data[n] = [
                ((n in g.predecessors(node)) or (n == node)) for node in molecules
            ]

        data = pd.DataFrame(data)
        data = data[~data["SMILES"].isnull()]
        data = data[[name not in CHEBI_BLACKLIST for name, _ in data.iterrows()]]
        data = data[data.iloc[:, 3:].any(axis=1)]
        return data

    def save_raw(self, data: pd.DataFrame, filename: str) -> None:
        """
        Save the raw dataset to a pickle file.

        Args:
            data (pd.DataFrame): The raw dataset to be saved.
            filename (str): The filename for the pickle file.
        """
        pd.to_pickle(data, open(os.path.join(self.raw_dir, filename), "wb"))

    def save_processed(self, data: pd.DataFrame, filename: str) -> None:
        """
        Save the processed dataset to a pickle file.

        Args:
            data (pd.DataFrame): The processed dataset to be saved.
            filename (str): The filename for the pickle file.
        """
        pd.to_pickle(data, open(os.path.join(self.processed_dir_main, filename), "wb"))

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
            if self.single_class is not None:
                single_cls_index = list(df.columns).index(int(self.single_class))
            for row in df.values:
                if self.single_class is None:
                    labels = row[3:].astype(bool)
                else:
                    labels = [bool(row[single_cls_index])]
                yield dict(features=row[2], labels=labels, ident=row[0])

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

    def _setup_pruned_test_set(
        self, df_test_chebi_version: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Create a test set with the same leaf nodes, but use only classes that appear in the training set.

        Args:
            df_test_chebi_version (pd.DataFrame): The test dataset.

        Returns:
            pd.DataFrame: The pruned test dataset.
        """
        # TODO: find a more efficient way to do this
        filename_old = "classes.txt"
        # filename_new = f"classes_v{self.chebi_version_train}.txt"
        # dataset = torch.load(os.path.join(self.processed_dir, "test.pt"))

        # Load original classes (from the current ChEBI version - chebi_version)
        with open(os.path.join(self.processed_dir_main, filename_old), "r") as file:
            orig_classes = file.readlines()

        # Load new classes (from the training ChEBI version - chebi_version_train)
        with open(
            os.path.join(
                self._chebi_version_train_obj.processed_dir_main, filename_old
            ),
            "r",
        ) as file:
            new_classes = file.readlines()

        # Create a mapping which give index of a class from chebi_version, if the corresponding
        # class exists in chebi_version_train, Size = Number of classes in chebi_version
        mapping = [
            None if or_class not in new_classes else new_classes.index(or_class)
            for or_class in orig_classes
        ]

        # Iterate over each data instance in the test set which is derived from chebi_version
        for _, row in df_test_chebi_version.iterrows():
            # Size = Number of classes in chebi_version_train
            new_labels = [False for _ in new_classes]
            for ind, label in enumerate(row["labels"]):
                # If the chebi_version class exists in the chebi_version_train and has a True label,
                # set the corresponding label in new_labels to True
                if mapping[ind] is not None and label:
                    new_labels[mapping[ind]] = label
            # Update the labels from test instance from chebi_version to the new labels, which are compatible to both versions
            row["labels"] = new_labels

        # torch.save(
        #     chebi_ver_test_data,
        #     os.path.join(self.processed_dir, self.processed_file_names_dict["test"]),
        # )
        return df_test_chebi_version

    def setup_processed(self) -> None:
        """
        Transform and prepare processed data for the ChEBI dataset.

        This method sets up the processed data directories and files based on the ChEBI version
        and train version (if specified). It ensures that the required processed data files exist
        by loading raw data, transforming it into processed format, and saving it.

        It also handles special cases, such as generating a pruned test set if `chebi_version_train`
        is specified and the test set does not already exist. This pruned test set includes only
        classes that appear in the training set.

        """
        print("Transform data")
        os.makedirs(self.processed_dir, exist_ok=True)
        # -------- Commented the code for Data Handling Restructure for Issue No.10
        # -------- https://github.com/ChEB-AI/python-chebai/issues/10
        # for k in self.processed_file_names_dict.keys():
        #     processed_name = (
        #         "test.pt" if k == "test" else self.processed_file_names_dict[k]
        #     )
        #     if not os.path.isfile(os.path.join(self.processed_dir, processed_name)):
        #         print("transform", k)
        #         torch.save(
        #             self._load_data_from_file(
        #                 os.path.join(self.raw_dir, self.raw_file_names_dict[k])
        #             ),
        #             os.path.join(self.processed_dir, processed_name),
        #         )
        # # create second test set with classes used in train
        # if self.chebi_version_train is not None and not os.path.isfile(
        #     os.path.join(self.processed_dir, self.processed_file_names_dict["test"])
        # ):
        #     print("transform test (select classes)")
        #     self._setup_pruned_test_set()
        #
        # processed_name = self.processed_file_names_dict[k]
        # if not os.path.isfile(os.path.join(self.processed_dir, processed_name)):
        #     print(
        #         "Missing encoded data, transform processed data into encoded data",
        #         k,
        #     )
        #     torch.save(
        #         self._load_data_from_file(
        #             os.path.join(
        #                 self.processed_dir_main, self.raw_file_names_dict[k]
        #             )
        #         ),
        #         os.path.join(self.processed_dir, processed_name),
        #     )

        # Transform the processed data into encoded data
        processed_name = self.processed_file_names_dict["data"]
        if not os.path.isfile(os.path.join(self.processed_dir, processed_name)):
            print(
                f"Missing encoded data related to version {self.chebi_version}, transform processed data into encoded data:",
                processed_name,
            )
            torch.save(
                self._load_data_from_file(
                    os.path.join(
                        self.processed_dir_main,
                        self.raw_file_names_dict["data"],
                    )
                ),
                os.path.join(self.processed_dir, processed_name),
            )
            if self.aug_data:
                augmented_dir = self.augmented_dir_main

                # Define the augmented data file path
                if not os.path.isfile(os.path.join(augmented_dir, "augmented_data.pt")):
                    print(
                        f"Missing encoded data related to version {self.chebi_version}, transform augmented data into encoded data:",
                        "augmented_data.pt",
                    )
                    torch.save(
                        self._load_data_from_file(
                            os.path.join(
                                augmented_dir,
                                "augmented_data.pkl",
                            )
                        ),
                        os.path.join(augmented_dir, "augmented_data.pt"),
                    )


        # Transform the data related to "chebi_version_train" to encoded data, if it doesn't exist
        if self.chebi_version_train is not None and not os.path.isfile(
            os.path.join(
                self._chebi_version_train_obj.processed_dir,
                self._chebi_version_train_obj.raw_file_names_dict["data"],
            )
        ):
            print(
                f"Missing encoded data related to train version: {self.chebi_version_train}"
            )
            print("Call the setup method related to it")
            self._chebi_version_train_obj.setup()

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

    @property
    def processed_dir_main(self) -> str:
        """
        Return the main directory path for processed data.

        Returns:
            str: The path to the main processed data directory.
        """
        return os.path.join(
            self.base_dir,
            self._name,
            "processed",
        )

    @property
    def augmented_dir_main(self) -> str:
        """
        Return the main directory path for processed data.

        Returns:
            str: The path to the main processed data directory.
        """
        return os.path.join(
            self.base_dir,
            self._name,
            "augmented",
        )

    @property
    def processed_dir(self) -> str:
        """
        Return the directory path for processed data.

        Returns:
            str: The path to the processed data directory.
        """
        res = os.path.join(
            self.processed_dir_main,
            *self.identifier,
        )
        if self.single_class is None:
            return res
        else:
            return os.path.join(res, f"single_{self.single_class}")

    @property
    def base_dir(self) -> str:
        """
        Return the base directory path for data.

        Returns:
            str: The base directory path for data.
        """
        return os.path.join("data", f"chebi_v{self.chebi_version}")

    @property
    def processed_file_names_dict(self) -> dict:
        """
        Return a dictionary of processed file names.

        Returns:
            dict: A dictionary where keys are file names and values are paths.
        """
        train_v_str = (
            f"_v{self.chebi_version_train}" if self.chebi_version_train else ""
        )
        # res = {"test": f"test{train_v_str}.pt"}
        res = {}

        for set in ["train", "validation"]:
            # TODO: code will be modified into CV issue for dynamic splits
            if self.use_inner_cross_validation:
                for i in range(self.inner_k_folds):
                    res[f"fold_{i}_{set}"] = os.path.join(
                        self.fold_dir, f"fold_{i}_{set}{train_v_str}.pt"
                    )
            # else:
            # res[set] = f"{set}{train_v_str}.pt"
        res["data"] = "data.pt"
        return res

    @property
    def raw_file_names_dict(self) -> dict:
        """
        Return a dictionary of raw file names.

        Returns:
            dict: A dictionary where keys are file names and values are paths.
        """
        train_v_str = (
            f"_v{self.chebi_version_train}" if self.chebi_version_train else ""
        )
        # res = {
        #     "test": f"test.pkl"
        # }  # no extra raw test version for chebi_version_train - use default test set and only
        # adapt processed file
        res = {}
        for set in ["train", "validation"]:
            # TODO: code will be modified into CV issue for dynamic splits
            if self.use_inner_cross_validation:
                for i in range(self.inner_k_folds):
                    res[f"fold_{i}_{set}"] = os.path.join(
                        self.fold_dir, f"fold_{i}_{set}{train_v_str}.pkl"
                    )
            # else:
            # res[set] = f"{set}{train_v_str}.pkl"
        res["data"] = "data.pkl"
        return res

    @property
    def processed_file_names(self) -> List[str]:
        """
        Return a list of processed file names.

        Returns:
            List[str]: A list containing processed file names.
        """
        return list(self.processed_file_names_dict.values())

    @property
    def raw_file_names(self) -> List[str]:
        """
        Return a list of raw file names.

        Returns:
            List[str]: A list containing raw file names.
        """
        return list(self.raw_file_names_dict.values())

    def _load_chebi(self, version: int) -> str:
        """
        Load the ChEBI ontology file.

        Args:
            version (int): The version of the ChEBI ontology to load.

        Returns:
            str: The file path of the loaded ChEBI ontology.
        """
        chebi_name = (
            f"chebi.obo" if version == self.chebi_version else f"chebi_v{version}.obo"
        )
        chebi_path = os.path.join(self.raw_dir, chebi_name)
        if not os.path.isfile(chebi_path):
            print(f"Load ChEBI ontology (v_{version})")
            url = f"http://purl.obolibrary.org/obo/chebi/{version}/chebi.obo"
            r = requests.get(url, allow_redirects=True)
            open(chebi_path, "wb").write(r.content)
            # # Define the source path of the backup file
            # local_backup_path = r"D:\Knowledge\Hiwi\chebi_v231\raw\chebi.obo"
            #  # Copy the file from the local backup to the target directory
            # shutil.copy(local_backup_path, chebi_path)
            # print(f"Copied local backup to {chebi_path}.")
        return chebi_path

    def prepare_data(self, *args: Any, **kwargs: Any) -> None:
        """
        Prepares the data for the Chebi dataset.

        This method checks for the presence of raw data in the specified directory.
        If the raw data is missing, it fetches the ontology and creates a test set.
        If the test set already exists, it loads it from the file.
        Then, it creates the train/validation split based on the test set.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            None
        """
        print("Check for processed data in", self.processed_dir_main)
        if any(
            not os.path.isfile(os.path.join(self.processed_dir_main, f))
            for f in self.raw_file_names
        ):
            os.makedirs(self.processed_dir_main, exist_ok=True)
            print("Missing raw data. Go fetch...")

            # -------- Commented the code for Data Handling Restructure for Issue No.10
            # -------- https://github.com/ChEB-AI/python-chebai/issues/10
            # missing test set -> create
            # if not os.path.isfile(
            #     os.path.join(self.raw_dir, self.raw_file_names_dict["test"])
            # ):
            #     chebi_path = self._load_chebi(self.chebi_version)
            #     g = self.extract_class_hierarchy(chebi_path)
            #     df = self.graph_to_raw_dataset(g, self.raw_file_names_dict["test"])
            #     _, test_df = self.get_test_split(df)
            #     self.save_raw(test_df, self.raw_file_names_dict["test"])
            # # load test_split from file
            # else:
            #     with open(
            #         os.path.join(self.raw_dir, self.raw_file_names_dict["test"]), "rb"
            #     ) as input_file:
            #         test_df = pickle.load(input_file)
            # # create train/val split based on test set
            # chebi_path = self._load_chebi(
            #     self.chebi_version_train
            #     if self.chebi_version_train is not None
            #     else self.chebi_version
            # )
            # g = self.extract_class_hierarchy(chebi_path)
            # if self.use_inner_cross_validation:
            #     df = self.graph_to_raw_dataset(
            #         g, self.raw_file_names_dict[f"fold_0_train"]
            #     )
            # else:
            #     df = self.graph_to_raw_dataset(g, self.raw_file_names_dict["train"])
            # train_val_dict = self.get_train_val_splits_given_test(df, test_df)
            # for name, df in train_val_dict.items():
            #     self.save_raw(df, name)

            # Data from chebi_version
            chebi_path = self._load_chebi(self.chebi_version)
            g = self.extract_class_hierarchy(chebi_path)
            df = self.graph_to_raw_dataset(g, self.raw_file_names_dict["data"])
            self.save_processed(df, filename=self.raw_file_names_dict["data"])
            self.augment_data(self.processed_dir_main, self.augment_data_batch_size)
            if self.chebi_version_train is not None:
                if not os.path.isfile(
                        os.path.join(
                            self._chebi_version_train_obj.processed_dir_main,
                            self._chebi_version_train_obj.raw_file_names_dict["data"],
                        )
                ):
                    print(
                        f"Missing processed data related to train version: {self.chebi_version_train}"
                    )
                    print("Call the prepare_data method related to it")
                    # Generate the "chebi_version_train" data if it doesn't exist
                    self._chebi_version_train_obj.prepare_data(*args, **kwargs)

    def augment_data(self, path: str, batch_size) -> None:
        print(("inside_augment_data"))
        print("batch_size",batch_size)
        if self.aug_data:
            if os.path.isfile(os.path.join(
                    path, self.raw_file_names_dict["data"])):
                augmented_dir = self.augmented_dir_main
                # Check if the augmented directory exists, if not, create it
                os.makedirs(augmented_dir, exist_ok=True)
                # Define the augmented data file path
                augmented_data_file = os.path.join(augmented_dir, "augmented_data.pkl")

                # If augmented_data.pkl does not already exist, proceed with the logic
                if not os.path.isfile(augmented_data_file):

                    data = self.read_file(os.path.join(
                        path, self.raw_file_names_dict["data"]))

                    total_rows = data.shape[0]
                    # Calculate the total number of batches
                    total_batches = (total_rows + batch_size - 1) // batch_size

                    for batch_num, start in enumerate(range(0, total_rows, batch_size), start=1):
                        end = min(start + batch_size, total_rows)
                        batch = data[start:end]
                        print(f"Processing batch {batch_num}/{total_batches} ({start} to {end})")

                        # Set to keep track of already seen SMILES
                        seen_smiles = set(batch['SMILES'])

                        # Store new rows in a list instead of concatenating directly
                        new_rows = []

                        # Updated tqdm to show batch number and total batches
                        for _, row in tqdm(batch.iterrows(), total=len(batch),
                                           desc=f"Batch {batch_num}/{total_batches}", unit="row"):
                            original_smiles = row['SMILES']
                            variations = self.generate_smiles_variations(original_smiles)
                            variations = [var for var in variations if var not in seen_smiles]

                            for var in variations:
                                new_row = row.copy()
                                new_row['SMILES'] = var
                                new_rows.append(new_row)
                                seen_smiles.add(var)

                        # Create a DataFrame from the new rows
                        new_df = pd.DataFrame(new_rows)

                        # Concatenate the batch and new variations, and save incrementally
                        batch_with_variations = pd.concat([batch, new_df], ignore_index=True)

                        # Append batch to augmented file
                        self.save_file(batch_with_variations, augmented_data_file, append=True)
        else:
            print("Data Augmentation config is False")

    def save_file(self, dataset: pd.DataFrame, file_path: str, append=False):
        if append and os.path.isfile(file_path):
            existing_data = pd.read_pickle(file_path)
            dataset = pd.concat([existing_data, dataset], ignore_index=True)
        pd.to_pickle(dataset, open(file_path, "wb"))

    # Function to generate SMILES variations using different configurations
    def generate_smiles_variations(self, original_smiles):
        num_variations = self.num_smiles_variations
        print(type(original_smiles), original_smiles)
        if not isinstance(original_smiles, str):
            print(f"Non-string SMILES found: {original_smiles}")
        mol = Chem.MolFromSmiles(original_smiles)
        if mol is None:
            return []  # Return an empty list if conversion fails

        # Get the number of atoms in the molecule
        num_atoms = mol.GetNumAtoms()

        # Generate the rooted_at_atoms list based on the number of atoms
        if num_atoms < num_variations:
            rooted_at_atoms = list(range(0, num_atoms))  # [0, num_atoms)
        else:
            rooted_at_atoms = list(range(0, num_variations))  # [0, num_variations)


        # Shuffle the rooted_at_atoms list to randomize the order
        random.shuffle(rooted_at_atoms)

        variations = set()

        # Loop through all combinations of doRandom and rootedAtAtom values
        for do_random in [True, False]:
            for rooted_at_atom in rooted_at_atoms:
                try:
                    # Generate SMILES with the given configuration
                    variant = Chem.MolToSmiles(
                        mol,
                        doRandom=do_random,
                        rootedAtAtom=rooted_at_atom,
                        canonical=False
                    )

                    # # Print the configuration and the generated SMILES string
                    # print(
                    #     f"Config: doRandom={do_random}, rootedAtAtom={rooted_at_atom}, canonical={False} -> SMILES: {variant}")

                    # Avoid duplicates with the original SMILES
                    if variant != original_smiles:
                        variations.add(variant)

                    # Check the number of variations after adding
                    if len(variations) >= num_variations:
                        return list(variations)  # Return immediately when enough variations are found

                except Exception as e:
                    # Skip invalid configurations
                    continue

        return list(variations)

    def read_file(self, file_path: str):
        df = pd.read_pickle(
            open(file_path, "rb"
                 )
        )
        return df

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

    def load_processed_data(
        self, kind: Optional[str] = None, filename: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Load processed data from a file.

        Args:
            kind (str, optional): The kind of dataset to load such as "train", "val", or "test". Defaults to None.
            filename (str, optional): The name of the file to load the dataset from. Defaults to None.

        Returns:
            List[Dict[str, Any]] : The loaded processed data.

        Raises:
            KeyError: If specified kind key doesn't exist.
            FileNotFoundError: If the specified file does not exist.
        """
        if kind is None and filename is None:
            raise ValueError(
                "Either kind or filename is required to load the correct dataset, both are None"
            )

        # If both kind and filename are given, use filename
        if kind is not None and filename is None:
            try:
                if self.use_inner_cross_validation and kind != "test":
                    filename = self.processed_file_names_dict[
                        f"fold_{self.fold_index}_{kind}"
                    ]
                else:
                    data_df = self.dynamic_split_dfs[kind]
                    return data_df.to_dict(orient="records")
            except KeyError:
                kind = f"{kind}"

        # If filename is provided
        try:
            return torch.load(os.path.join(self.processed_dir, filename))
        except FileNotFoundError:
            raise FileNotFoundError(f"File {filename} doesn't exist")


class JCIExtendedBase(_ChEBIDataExtractor):
    LABEL_INDEX = 3
    SMILES_INDEX = 2

    @property
    def label_number(self):
        return 500

    @property
    def _name(self):
        return "JCI_extended"

    def select_classes(self, g, *args, **kwargs):
        return JCI_500_COLUMNS_INT


class ChEBIOverX(_ChEBIDataExtractor):
    """
    A class for extracting data from the ChEBI dataset with a threshold for selecting classes.

    Attributes:
        LABEL_INDEX (int): The index of the label in the dataset.
        SMILES_INDEX (int): The index of the SMILES string in the dataset.
        READER (ChemDataReader): The reader used for reading the dataset.
        THRESHOLD (None): The threshold for selecting classes.
    """

    LABEL_INDEX: int = 3
    SMILES_INDEX: int = 2
    READER: dr.ChemDataReader = dr.ChemDataReader
    THRESHOLD: int = None

    @property
    def label_number(self) -> int:
        """
        Returns the number of labels in the dataset.

        Returns:
            int: The number of labels.
        """
        return 854

    @property
    def _name(self) -> str:
        """
        Returns the name of the dataset.

        Returns:
            str: The dataset name.
        """
        return f"ChEBI{self.THRESHOLD}"

    def select_classes(self, g: nx.Graph, split_name: str, *args, **kwargs) -> List:
        """
        Selects classes from the ChEBI dataset.

        Args:
            g (nx.Graph): The graph representing the dataset.
            split_name (str): The name of the split.
            *args: Additional arguments (not used).
            **kwargs: Additional keyword arguments (not used).

        Returns:
            list: The list of selected classes.
        """
        smiles = nx.get_node_attributes(g, "smiles")
        nodes = list(
            sorted(
                {
                    node
                    for node in g.nodes
                    if sum(
                        1 if smiles[s] is not None else 0 for s in g.successors(node)
                    )
                    >= self.THRESHOLD
                }
            )
        )
        filename = "classes.txt"
        # if (
        #     self.chebi_version_train
        #     is not None
        #     # and self.raw_file_names_dict["test"] != split_name
        # ):
        #     filename = f"classes_v{self.chebi_version_train}.txt"
        with open(os.path.join(self.processed_dir_main, filename), "wt") as fout:
            fout.writelines(str(node) + "\n" for node in nodes)
        return nodes


class ChEBIOverXDeepSMILES(ChEBIOverX):
    """
    A class for extracting data from the ChEBI dataset with DeepChem SMILES reader.

    Inherits from ChEBIOverX.

    Attributes:
        READER (DeepChemDataReader): The reader used for reading the dataset (DeepChem SMILES).
    """

    READER: dr.DeepChemDataReader = dr.DeepChemDataReader


class ChEBIOverXSELFIES(ChEBIOverX):
    """
    A class for extracting data from the ChEBI dataset with SELFIES reader.

    Inherits from ChEBIOverX.

    Attributes:
        READER (SelfiesReader): The reader used for reading the dataset (SELFIES).
    """

    READER: dr.SelfiesReader = dr.SelfiesReader


class ChEBIOver100(ChEBIOverX):
    """
    A class for extracting data from the ChEBI dataset with a threshold of 100 for selecting classes.

    Inherits from ChEBIOverX.

    Attributes:
        THRESHOLD (int): The threshold for selecting classes (100).
    """

    THRESHOLD: int = 100

    def label_number(self) -> int:
        """
        Returns the number of labels in the dataset.

        Overrides the base class method to return the correct number of labels for this threshold.

        Returns:
            int: The number of labels.
        """
        return 854


class ChEBIOver50(ChEBIOverX):
    """
    A class for extracting data from the ChEBI dataset with a threshold of 50 for selecting classes.

    Inherits from ChEBIOverX.

    Attributes:
        THRESHOLD (int): The threshold for selecting classes (50).
    """

    THRESHOLD: int = 50

    def label_number(self) -> int:
        """
        Returns the number of labels in the dataset.

        Overrides the base class method to return the correct number of labels for this threshold.

        Returns:
            int: The number of labels.
        """
        return 1332


class ChEBIOver100DeepSMILES(ChEBIOverXDeepSMILES, ChEBIOver100):
    """
    A class for extracting data from the ChEBI dataset with DeepChem SMILES reader and a threshold of 100.

    Inherits from ChEBIOverXDeepSMILES and ChEBIOver100.
    """

    pass


class ChEBIOver100SELFIES(ChEBIOverXSELFIES, ChEBIOver100):
    """
    A class for extracting data from the ChEBI dataset with SELFIES reader and a threshold of 100.

    Inherits from ChEBIOverXSELFIES and ChEBIOver100.
    """

    pass


class ChEBIOver50SELFIES(ChEBIOverXSELFIES, ChEBIOver50):
    pass


class ChEBIOverXPartial(ChEBIOverX):
    """
    Dataset that doesn't use the full ChEBI, but extracts a part of ChEBI (subclasses of a given top class)

    Attributes:
        top_class_id (int): The ID of the top class from which to extract subclasses.
    """

    def __init__(self, top_class_id: int, **kwargs):
        """
        Initializes the ChEBIOverXPartial dataset.

        Args:
            top_class_id (int): The ID of the top class from which to extract subclasses.
            **kwargs: Additional keyword arguments passed to the superclass initializer.
        """
        self.top_class_id: int = top_class_id
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
            f"partial_{self.top_class_id}",
            "processed",
        )

    def extract_class_hierarchy(self, chebi_path: str) -> nx.DiGraph:
        """
        Extracts a subset of ChEBI based on subclasses of the top class ID.

        Args:
            chebi_path (str): The file path to the ChEBI ontology file.

        Returns:
            nx.DiGraph: The extracted class hierarchy as a directed graph.
        """
        with open(chebi_path, encoding="utf-8") as chebi:
            chebi = "\n".join(l for l in chebi if not l.startswith("xref:"))
        elements = [
            term_callback(clause)
            for clause in fastobo.loads(chebi)
            if clause and ":" in str(clause.id)
        ]
        g = nx.DiGraph()
        for n in elements:
            g.add_node(n["id"], **n)
        g.add_edges_from([(p, q["id"]) for q in elements for p in q["parents"]])

        g = nx.transitive_closure_dag(g)
        g = g.subgraph(list(nx.descendants(g, self.top_class_id)) + [self.top_class_id])
        print("Compute transitive closure")
        return g


class ChEBIOver50Partial(ChEBIOverXPartial, ChEBIOver50):
    """
    Dataset that extracts a part of ChEBI based on subclasses of a given top class,
    with a threshold of 50 for selecting classes.

    Inherits from ChEBIOverXPartial and ChEBIOver50.
    """

    pass


class JCIExtendedBPEData(JCIExtendedBase):
    READER = dr.ChemBPEReader


class JCIExtendedTokenData(JCIExtendedBase):
    READER = dr.ChemDataReader


class JCIExtSelfies(JCIExtendedBase):
    READER = dr.SelfiesReader


def chebi_to_int(s: str) -> int:
    """
    Converts a ChEBI term string representation to an integer ID.

    Args:
    - s (str): A ChEBI term string, e.g., "CHEBI:12345".

    Returns:
    - int: The integer ID extracted from the ChEBI term string.
    """
    return int(s[s.index(":") + 1 :])


def term_callback(doc) -> dict:
    """
    Extracts information from a ChEBI term document.
    This function takes a ChEBI term document as input and extracts relevant information such as the term ID, parents,
    parts, name, and SMILES string. It returns a dictionary containing the extracted information.

    Args:
    - doc: A ChEBI term document.

    Returns:
    A dictionary containing the following keys:
    - "id": The ID of the ChEBI term.
    - "parents": A list of parent term IDs.
    - "has_part": A set of term IDs representing the parts of the ChEBI term.
    - "name": The name of the ChEBI term.
    - "smiles": The SMILES string associated with the ChEBI term, if available.
    """
    parts = set()
    parents = []
    name = None
    smiles = None
    for clause in doc:
        if isinstance(clause, fastobo.term.PropertyValueClause):
            t = clause.property_value
            if str(t.relation) == "http://purl.obolibrary.org/obo/chebi/smiles":
                assert smiles is None
                smiles = t.value
        # in older chebi versions, smiles strings are synonyms
        # e.g. synonym: "[F-].[Na+]" RELATED SMILES [ChEBI]
        elif isinstance(clause, fastobo.term.SynonymClause):
            if "SMILES" in clause.raw_value():
                assert smiles is None
                smiles = clause.raw_value().split('"')[1]
        elif isinstance(clause, fastobo.term.RelationshipClause):
            if str(clause.typedef) == "has_part":
                parts.add(chebi_to_int(str(clause.term)))
        elif isinstance(clause, fastobo.term.IsAClause):
            parents.append(chebi_to_int(str(clause.term)))
        elif isinstance(clause, fastobo.term.NameClause):
            name = str(clause.name)
    return {
        "id": chebi_to_int(str(doc.id)),
        "parents": parents,
        "has_part": parts,
        "name": name,
        "smiles": smiles,
    }


atom_index = (
    "\*",
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    "Nh",
    "Fl",
    "Mc",
    "Lv",
    "Ts",
    "Og",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "c",
    "n",
    "s",
    "o",
    "se",
    "p",
)

JCI_500_COLUMNS = [
    "CHEBI:25716",
    "CHEBI:72010",
    "CHEBI:60926",
    "CHEBI:39206",
    "CHEBI:24315",
    "CHEBI:22693",
    "CHEBI:23981",
    "CHEBI:23066",
    "CHEBI:35343",
    "CHEBI:18303",
    "CHEBI:60971",
    "CHEBI:35753",
    "CHEBI:24836",
    "CHEBI:59268",
    "CHEBI:35992",
    "CHEBI:51718",
    "CHEBI:27093",
    "CHEBI:38311",
    "CHEBI:46940",
    "CHEBI:26399",
    "CHEBI:27325",
    "CHEBI:33637",
    "CHEBI:37010",
    "CHEBI:36786",
    "CHEBI:59777",
    "CHEBI:36871",
    "CHEBI:26799",
    "CHEBI:50525",
    "CHEBI:26848",
    "CHEBI:52782",
    "CHEBI:75885",
    "CHEBI:37533",
    "CHEBI:47018",
    "CHEBI:27150",
    "CHEBI:26707",
    "CHEBI:131871",
    "CHEBI:134179",
    "CHEBI:24727",
    "CHEBI:59238",
    "CHEBI:26373",
    "CHEBI:46774",
    "CHEBI:33642",
    "CHEBI:38686",
    "CHEBI:74222",
    "CHEBI:23666",
    "CHEBI:46770",
    "CHEBI:16460",
    "CHEBI:37485",
    "CHEBI:21644",
    "CHEBI:52565",
    "CHEBI:33576",
    "CHEBI:76170",
    "CHEBI:46640",
    "CHEBI:61902",
    "CHEBI:22750",
    "CHEBI:35348",
    "CHEBI:48030",
    "CHEBI:2571",
    "CHEBI:38131",
    "CHEBI:83575",
    "CHEBI:136889",
    "CHEBI:26250",
    "CHEBI:36244",
    "CHEBI:23906",
    "CHEBI:38261",
    "CHEBI:22916",
    "CHEBI:35924",
    "CHEBI:24689",
    "CHEBI:32877",
    "CHEBI:50511",
    "CHEBI:26588",
    "CHEBI:24385",
    "CHEBI:5653",
    "CHEBI:48591",
    "CHEBI:38295",
    "CHEBI:58944",
    "CHEBI:134396",
    "CHEBI:49172",
    "CHEBI:26558",
    "CHEBI:64708",
    "CHEBI:35923",
    "CHEBI:25961",
    "CHEBI:47779",
    "CHEBI:46812",
    "CHEBI:37863",
    "CHEBI:22718",
    "CHEBI:36562",
    "CHEBI:38771",
    "CHEBI:36078",
    "CHEBI:26935",
    "CHEBI:33555",
    "CHEBI:23044",
    "CHEBI:15693",
    "CHEBI:33892",
    "CHEBI:33909",
    "CHEBI:35766",
    "CHEBI:51149",
    "CHEBI:35972",
    "CHEBI:38304",
    "CHEBI:46942",
    "CHEBI:24026",
    "CHEBI:33721",
    "CHEBI:38093",
    "CHEBI:38830",
    "CHEBI:26875",
    "CHEBI:37963",
    "CHEBI:61910",
    "CHEBI:47891",
    "CHEBI:74818",
    "CHEBI:50401",
    "CHEBI:24834",
    "CHEBI:33299",
    "CHEBI:63424",
    "CHEBI:63427",
    "CHEBI:15841",
    "CHEBI:33666",
    "CHEBI:26214",
    "CHEBI:22484",
    "CHEBI:27024",
    "CHEBI:46845",
    "CHEBI:64365",
    "CHEBI:63566",
    "CHEBI:38757",
    "CHEBI:83264",
    "CHEBI:24867",
    "CHEBI:37841",
    "CHEBI:33720",
    "CHEBI:36885",
    "CHEBI:59412",
    "CHEBI:64612",
    "CHEBI:36500",
    "CHEBI:37015",
    "CHEBI:84135",
    "CHEBI:51751",
    "CHEBI:18133",
    "CHEBI:57613",
    "CHEBI:38976",
    "CHEBI:25810",
    "CHEBI:24873",
    "CHEBI:35571",
    "CHEBI:83812",
    "CHEBI:37909",
    "CHEBI:51750",
    "CHEBI:15889",
    "CHEBI:48470",
    "CHEBI:24676",
    "CHEBI:22480",
    "CHEBI:139051",
    "CHEBI:23252",
    "CHEBI:51454",
    "CHEBI:88061",
    "CHEBI:46874",
    "CHEBI:38338",
    "CHEBI:62618",
    "CHEBI:59266",
    "CHEBI:84403",
    "CHEBI:27116",
    "CHEBI:77632",
    "CHEBI:38418",
    "CHEBI:35213",
    "CHEBI:35496",
    "CHEBI:78799",
    "CHEBI:38314",
    "CHEBI:35568",
    "CHEBI:35573",
    "CHEBI:33847",
    "CHEBI:16038",
    "CHEBI:33741",
    "CHEBI:33654",
    "CHEBI:17387",
    "CHEBI:33572",
    "CHEBI:36233",
    "CHEBI:22297",
    "CHEBI:23990",
    "CHEBI:38102",
    "CHEBI:24436",
    "CHEBI:35189",
    "CHEBI:79202",
    "CHEBI:68489",
    "CHEBI:18254",
    "CHEBI:78189",
    "CHEBI:47019",
    "CHEBI:61655",
    "CHEBI:24373",
    "CHEBI:26347",
    "CHEBI:36709",
    "CHEBI:73539",
    "CHEBI:35507",
    "CHEBI:35293",
    "CHEBI:140326",
    "CHEBI:46668",
    "CHEBI:17188",
    "CHEBI:61109",
    "CHEBI:35819",
    "CHEBI:33744",
    "CHEBI:73474",
    "CHEBI:134361",
    "CHEBI:33238",
    "CHEBI:26766",
    "CHEBI:17517",
    "CHEBI:25508",
    "CHEBI:22580",
    "CHEBI:26394",
    "CHEBI:35356",
    "CHEBI:50918",
    "CHEBI:24860",
    "CHEBI:2468",
    "CHEBI:33581",
    "CHEBI:26519",
    "CHEBI:37948",
    "CHEBI:33823",
    "CHEBI:59554",
    "CHEBI:46848",
    "CHEBI:24897",
    "CHEBI:26893",
    "CHEBI:63394",
    "CHEBI:29348",
    "CHEBI:35790",
    "CHEBI:25241",
    "CHEBI:58958",
    "CHEBI:24397",
    "CHEBI:25413",
    "CHEBI:24302",
    "CHEBI:46850",
    "CHEBI:51867",
    "CHEBI:35314",
    "CHEBI:50893",
    "CHEBI:36130",
    "CHEBI:33558",
    "CHEBI:24782",
    "CHEBI:36087",
    "CHEBI:26649",
    "CHEBI:47923",
    "CHEBI:33184",
    "CHEBI:23643",
    "CHEBI:25985",
    "CHEBI:33257",
    "CHEBI:61355",
    "CHEBI:24697",
    "CHEBI:36838",
    "CHEBI:23451",
    "CHEBI:33242",
    "CHEBI:26872",
    "CHEBI:50523",
    "CHEBI:16701",
    "CHEBI:36699",
    "CHEBI:35505",
    "CHEBI:24360",
    "CHEBI:59737",
    "CHEBI:26455",
    "CHEBI:51285",
    "CHEBI:35504",
    "CHEBI:36309",
    "CHEBI:33554",
    "CHEBI:47909",
    "CHEBI:50858",
    "CHEBI:53339",
    "CHEBI:25609",
    "CHEBI:23665",
    "CHEBI:35902",
    "CHEBI:35552",
    "CHEBI:139592",
    "CHEBI:35724",
    "CHEBI:38337",
    "CHEBI:35241",
    "CHEBI:29075",
    "CHEBI:62941",
    "CHEBI:140345",
    "CHEBI:59769",
    "CHEBI:28863",
    "CHEBI:47882",
    "CHEBI:35903",
    "CHEBI:33641",
    "CHEBI:47784",
    "CHEBI:23079",
    "CHEBI:25036",
    "CHEBI:50018",
    "CHEBI:28874",
    "CHEBI:35276",
    "CHEBI:26764",
    "CHEBI:65323",
    "CHEBI:51276",
    "CHEBI:37022",
    "CHEBI:22478",
    "CHEBI:23449",
    "CHEBI:72823",
    "CHEBI:63567",
    "CHEBI:50753",
    "CHEBI:38785",
    "CHEBI:46952",
    "CHEBI:36914",
    "CHEBI:33653",
    "CHEBI:62937",
    "CHEBI:36315",
    "CHEBI:37667",
    "CHEBI:38835",
    "CHEBI:35315",
    "CHEBI:33551",
    "CHEBI:18154",
    "CHEBI:79346",
    "CHEBI:26932",
    "CHEBI:39203",
    "CHEBI:25235",
    "CHEBI:23003",
    "CHEBI:64583",
    "CHEBI:46955",
    "CHEBI:33658",
    "CHEBI:59202",
    "CHEBI:28892",
    "CHEBI:33599",
    "CHEBI:33259",
    "CHEBI:64611",
    "CHEBI:37947",
    "CHEBI:65321",
    "CHEBI:63571",
    "CHEBI:25830",
    "CHEBI:50492",
    "CHEBI:26961",
    "CHEBI:33482",
    "CHEBI:63436",
    "CHEBI:47017",
    "CHEBI:51681",
    "CHEBI:48901",
    "CHEBI:52575",
    "CHEBI:35683",
    "CHEBI:24353",
    "CHEBI:61778",
    "CHEBI:13248",
    "CHEBI:35990",
    "CHEBI:33485",
    "CHEBI:35871",
    "CHEBI:27933",
    "CHEBI:27136",
    "CHEBI:26407",
    "CHEBI:33566",
    "CHEBI:47880",
    "CHEBI:24921",
    "CHEBI:38077",
    "CHEBI:48975",
    "CHEBI:59835",
    "CHEBI:83273",
    "CHEBI:22562",
    "CHEBI:33838",
    "CHEBI:35627",
    "CHEBI:51614",
    "CHEBI:36836",
    "CHEBI:63423",
    "CHEBI:22331",
    "CHEBI:25529",
    "CHEBI:36314",
    "CHEBI:83822",
    "CHEBI:38164",
    "CHEBI:51006",
    "CHEBI:28965",
    "CHEBI:38716",
    "CHEBI:76567",
    "CHEBI:35381",
    "CHEBI:51269",
    "CHEBI:37141",
    "CHEBI:25872",
    "CHEBI:36526",
    "CHEBI:51702",
    "CHEBI:25106",
    "CHEBI:51737",
    "CHEBI:38672",
    "CHEBI:36132",
    "CHEBI:38700",
    "CHEBI:25558",
    "CHEBI:17855",
    "CHEBI:18946",
    "CHEBI:83565",
    "CHEBI:15705",
    "CHEBI:35186",
    "CHEBI:33694",
    "CHEBI:36711",
    "CHEBI:23403",
    "CHEBI:35238",
    "CHEBI:36807",
    "CHEBI:47788",
    "CHEBI:24531",
    "CHEBI:33663",
    "CHEBI:22715",
    "CHEBI:57560",
    "CHEBI:38163",
    "CHEBI:23899",
    "CHEBI:50994",
    "CHEBI:26776",
    "CHEBI:51569",
    "CHEBI:35259",
    "CHEBI:77636",
    "CHEBI:35727",
    "CHEBI:35786",
    "CHEBI:24780",
    "CHEBI:26714",
    "CHEBI:26712",
    "CHEBI:26819",
    "CHEBI:63944",
    "CHEBI:36520",
    "CHEBI:25409",
    "CHEBI:22928",
    "CHEBI:23824",
    "CHEBI:79020",
    "CHEBI:26605",
    "CHEBI:139588",
    "CHEBI:52396",
    "CHEBI:37668",
    "CHEBI:50995",
    "CHEBI:52395",
    "CHEBI:61777",
    "CHEBI:38445",
    "CHEBI:24698",
    "CHEBI:63551",
    "CHEBI:35693",
    "CHEBI:83403",
    "CHEBI:36094",
    "CHEBI:35479",
    "CHEBI:25704",
    "CHEBI:25754",
    "CHEBI:38958",
    "CHEBI:21731",
    "CHEBI:23697",
    "CHEBI:38260",
    "CHEBI:33861",
    "CHEBI:22485",
    "CHEBI:2580",
    "CHEBI:18379",
    "CHEBI:23424",
    "CHEBI:33296",
    "CHEBI:37554",
    "CHEBI:33839",
    "CHEBI:36054",
    "CHEBI:23232",
    "CHEBI:18035",
    "CHEBI:63353",
    "CHEBI:23114",
    "CHEBI:76578",
    "CHEBI:26208",
    "CHEBI:32955",
    "CHEBI:24922",
    "CHEBI:36141",
    "CHEBI:24043",
    "CHEBI:35692",
    "CHEBI:46867",
    "CHEBI:38530",
    "CHEBI:24654",
    "CHEBI:38032",
    "CHEBI:26820",
    "CHEBI:35789",
    "CHEBI:62732",
    "CHEBI:26912",
    "CHEBI:22160",
    "CHEBI:26410",
    "CHEBI:36059",
    "CHEBI:51069",
    "CHEBI:33570",
    "CHEBI:24129",
    "CHEBI:37826",
    "CHEBI:16385",
    "CHEBI:26822",
    "CHEBI:46761",
    "CHEBI:83925",
    "CHEBI:25248",
    "CHEBI:37581",
    "CHEBI:35748",
    "CHEBI:26195",
    "CHEBI:33958",
    "CHEBI:58342",
    "CHEBI:17478",
    "CHEBI:36834",
    "CHEBI:25513",
    "CHEBI:57643",
    "CHEBI:38298",
    "CHEBI:64482",
    "CHEBI:33240",
    "CHEBI:47622",
    "CHEBI:33704",
    "CHEBI:83820",
    "CHEBI:33676",
    "CHEBI:32952",
    "CHEBI:131927",
    "CHEBI:26188",
    "CHEBI:35716",
    "CHEBI:28963",
    "CHEBI:22798",
    "CHEBI:60980",
    "CHEBI:17984",
    "CHEBI:37240",
    "CHEBI:28868",
    "CHEBI:27208",
    "CHEBI:15904",
    "CHEBI:35715",
    "CHEBI:22251",
    "CHEBI:61078",
    "CHEBI:61079",
    "CHEBI:58946",
    "CHEBI:37123",
    "CHEBI:33497",
    "CHEBI:50699",
    "CHEBI:22475",
    "CHEBI:35436",
]

JCI_500_COLUMNS_INT = [int(n.split(":")[-1]) for n in JCI_500_COLUMNS]



