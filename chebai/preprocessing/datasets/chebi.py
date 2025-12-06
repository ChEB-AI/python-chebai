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
import random
from abc import ABC
from collections import OrderedDict
from itertools import cycle, permutations, product
from typing import TYPE_CHECKING, Any, Dict, Generator, List, Literal, Optional, Union

import numpy as np
import pandas as pd
import torch
from rdkit import Chem

from chebai.preprocessing import reader as dr
from chebai.preprocessing.datasets.base import XYBaseDataModule, _DynamicDataset

if TYPE_CHECKING:
    import fastobo
    import networkx as nx

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

    def _perform_data_preparation(self, *args, **kwargs):
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


class JCIData(JCIBase):
    READER = dr.OrdReader


class JCISelfies(JCIBase):
    READER = dr.SelfiesReader


class JCITokenData(JCIBase):
    READER = dr.ChemDataReader


class _ChEBIDataExtractor(_DynamicDataset, ABC):
    """
    A class for extracting and processing data from the ChEBI dataset.

    Args:
        chebi_version_train (int, optional): The version of ChEBI to use for training and validation. If not set,
            chebi_version will be used for training, validation and test. Defaults to None.
        single_class (int, optional): The ID of the single class to predict. If not set, all available labels will be
            predicted. Defaults to None.
        subset (Literal["2_STAR", "3_STAR"], optional): If set, only use entities that are part of the given subset.
        **kwargs: Additional keyword arguments (passed to XYBaseDataModule).

    Attributes:
        single_class (Optional[int]): The ID of the single class to predict.
        chebi_version_train (Optional[int]): The version of ChEBI to use for training and validation.
        subset (Optional[Literal["2_STAR", "3_STAR"]]): If set, only use entities that are part of the given subset.

    """

    # ---- Index for columns of processed `data.pkl` (derived from `_graph_to_raw_dataset` method) ------
    # "id" at                 row index 0
    # "name" at               row index 1
    # "SMILES" at             row index 2
    # labels starting from    row index 3
    _ID_IDX: int = 0
    _DATA_REPRESENTATION_IDX: int = 2
    _LABELS_START_IDX: int = 3

    def __init__(
        self,
        chebi_version_train: Optional[int] = None,
        single_class: Optional[int] = None,
        subset: Optional[Literal["2_STAR", "3_STAR"]] = None,
        augment_smiles: bool = False,
        aug_smiles_variations: Optional[int] = None,
        **kwargs,
    ):
        if bool(augment_smiles):
            assert int(aug_smiles_variations) > 0, (
                "Number of variations must be greater than 0"
            )
            aug_smiles_variations = int(aug_smiles_variations)

            if not kwargs.get("splits_file_path", None):
                raise ValueError(
                    "When using SMILES augmentation, a splits_file_path must be provided to ensure consistent splits."
                )

            reader_kwargs = kwargs.get("reader_kwargs", {})
            reader_kwargs["canonicalize_smiles"] = False
            kwargs["reader_kwargs"] = reader_kwargs

        self.augment_smiles = bool(augment_smiles)
        self.aug_smiles_variations = aug_smiles_variations
        # predict only single class (given as id of one of the classes present in the raw data set)
        self.single_class = single_class
        self.subset = subset

        super(_ChEBIDataExtractor, self).__init__(**kwargs)
        # use different version of chebi for training and validation (if not None)
        # (still uses self.chebi_version for test set)
        self.chebi_version_train = chebi_version_train

        if self.chebi_version_train is not None:
            # Instantiate another same class with "chebi_version" as "chebi_version_train", if train_version is given
            # This is to get the data from respective directory related to "chebi_version_train"
            _init_kwargs = kwargs
            _init_kwargs["chebi_version"] = self.chebi_version_train
            self._chebi_version_train_obj = self.__class__(
                single_class=self.single_class,
                augment_smiles=self.augment_smiles,
                aug_smiles_variations=self.aug_smiles_variations,
                **_init_kwargs,
            )

    # ------------------------------ Phase: Prepare data -----------------------------------
    def _perform_data_preparation(self, *args: Any, **kwargs: Any) -> None:
        """
        Prepares the data for the Chebi dataset.

        This method checks for the presence of raw data in the specified directory.
        If the raw data is missing, it fetches the ontology and creates a dataframe & saves it as data.pkl pickle file.

        The resulting dataframe/pickle file is expected to contain columns with the following structure:
            - Column at index `self._ID_IDX`: ID of chebi data instance
            - Column at index `self._DATA_REPRESENTATION_IDX`: SMILES representation of the chemical
            - Column from index `self._LABELS_START_IDX` onwards: Labels

        It will pre-process the data related to `chebi_version_train`, if specified.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            None
        """
        super()._perform_data_preparation(args, kwargs)

        if self.chebi_version_train is not None:
            if not os.path.isfile(
                os.path.join(
                    self._chebi_version_train_obj.processed_dir_main,
                    self._chebi_version_train_obj.processed_main_file_names_dict[
                        "data"
                    ],
                )
            ):
                print(
                    f"Missing processed data related to train version: {self.chebi_version_train}"
                )
                print("Calling the prepare_data method related to it")
                # Generate the "chebi_version_train" data if it doesn't exist
                self._chebi_version_train_obj.prepare_data(*args, **kwargs)

    def _download_required_data(self) -> str:
        """
        Downloads the required raw data related to chebi.

        Returns:
            str: Path to the downloaded data.
        """
        return self._load_chebi(self.chebi_version)

    def _load_chebi(self, version: int) -> str:
        """
        Load the ChEBI ontology file.

        Args:
            version (int): The version of the ChEBI ontology to load.

        Returns:
            str: The file path of the loaded ChEBI ontology.
        """
        import requests

        chebi_name = self.raw_file_names_dict["chebi"]
        chebi_path = os.path.join(self.raw_dir, chebi_name)
        if not os.path.isfile(chebi_path):
            print(
                f"Missing raw chebi data related to version: v_{version}, Downloading..."
            )
            url = f"http://purl.obolibrary.org/obo/chebi/{version}/chebi.obo"
            r = requests.get(url, allow_redirects=True)
            open(chebi_path, "wb").write(r.content)
        return chebi_path

    def _extract_class_hierarchy(self, data_path: str) -> "nx.DiGraph":
        """
        Extracts the class hierarchy from the ChEBI ontology.
        Constructs a directed graph (DiGraph) using NetworkX, where nodes are annotated with fields/terms from
        the chebi term documents from `.obo` file.

        Args:
            data_path (str): The path to the ChEBI ontology.

        Returns:
            nx.DiGraph: The class hierarchy.
        """
        import fastobo
        import networkx as nx

        with open(data_path, encoding="utf-8") as chebi:
            chebi = "\n".join(line for line in chebi if not line.startswith("xref:"))

        elements = []
        for term_doc in fastobo.loads(chebi):
            if (
                term_doc
                and isinstance(term_doc.id, fastobo.id.PrefixedIdent)
                and term_doc.id.prefix == "CHEBI"
            ):
                term_dict = term_callback(term_doc)
                if term_dict and (
                    not self.subset or term_dict["subset"] == self.subset
                ):
                    elements.append(term_dict)

        g = nx.DiGraph()
        for n in elements:
            g.add_node(n["id"], **n)

        # Only take the edges which connects the existing nodes, to avoid internal creation of obsolete nodes
        # https://github.com/ChEB-AI/python-chebai/pull/55#issuecomment-2386654142
        g.add_edges_from(
            [(p, q["id"]) for q in elements for p in q["parents"] if g.has_node(p)]
        )

        print("Compute transitive closure")
        return nx.transitive_closure_dag(g)

    def _graph_to_raw_dataset(self, g: "nx.DiGraph") -> pd.DataFrame:
        """
        Converts the graph to a raw dataset.
        Uses the graph created by `_extract_class_hierarchy` method to extract the
        raw data in Dataframe format with additional columns corresponding to each multi-label class.

        Args:
            g (nx.DiGraph): The class hierarchy graph.

        Returns:
            pd.DataFrame: The raw dataset created from the graph.
        """
        import networkx as nx

        smiles = nx.get_node_attributes(g, "smiles")
        names = nx.get_node_attributes(g, "name")

        print("Process graph")

        molecules, smiles_list = zip(
            *(
                (n, smiles)
                for n, smiles in ((n, smiles.get(n)) for n in smiles.keys())
                if smiles
            )
        )
        data = OrderedDict(id=molecules)  # `id` column at index 0
        data["name"] = [
            names.get(node) for node in molecules
        ]  # `name` column at index 1
        data["SMILES"] = smiles_list  # `SMILES` (data representation) column at index 2

        # Labels columns from index 3 onwards
        for n in self.select_classes(g):
            data[n] = [
                ((n in g.predecessors(node)) or (n == node)) for node in molecules
            ]

        data = pd.DataFrame(data)
        data = data[~data["SMILES"].isnull()]
        data = data[~data["name"].isin(CHEBI_BLACKLIST)]

        return data

    def _after_prepare_data(self, *args, **kwargs) -> None:
        self._perform_smiles_augmentation()

    def _perform_smiles_augmentation(self) -> None:
        if not self.augment_smiles:
            return

        aug_pkl_file_name = self.processed_main_file_names_dict["aug_data"]
        aug_data_df = self.get_processed_pickled_df_file(aug_pkl_file_name)
        if aug_data_df is not None:
            self._data_pkl_filename = aug_pkl_file_name
            return

        data_df = self.get_processed_pickled_df_file(
            self.processed_main_file_names_dict["data"]
        )

        # +1 to account for if original SMILES is generated by random chance
        AUG_SMILES_VARIATIONS = self.aug_smiles_variations + 1

        def generate_augmented_smiles(smiles: str) -> list[str]:
            mol: Chem.Mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return [smiles]  # if mol is None, return original SMILES

            # sanitization set to False, as it can alter the fragment representation in ways you might not want.
            # As we don’t want RDKit to "fix" fragments, only need the fragments as-is, to generate SMILES strings.
            frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
            augmented = set()

            frag_smiles: list[set] = []
            for frag in frags:
                atom_ids = [atom.GetIdx() for atom in frag.GetAtoms()]
                random.shuffle(atom_ids)  # seed set by lightning
                atom_id_iter = cycle(atom_ids)
                frag_smiles.append(
                    {
                        Chem.MolToSmiles(
                            frag, rootedAtAtom=next(atom_id_iter), doRandom=True
                        )
                        for _ in range(AUG_SMILES_VARIATIONS)
                    }
                )
            if len(frags) > 1:
                # all permutations (ignoring the set order, meaning mixing sets in every order),
                aug_counter: int = 0
                for perm in permutations(frag_smiles):
                    for combo in product(*perm):
                        augmented.add(".".join(combo))
                        aug_counter += 1
                        if aug_counter >= AUG_SMILES_VARIATIONS:
                            break
                    if aug_counter >= AUG_SMILES_VARIATIONS:
                        break
            else:
                augmented = frag_smiles[0]

            if smiles in augmented:
                augmented.remove(smiles)

            if len(augmented) > AUG_SMILES_VARIATIONS - 1:
                # AUG_SMILES_VARIATIONS = number of new smiles needed to generate + original smiles
                # if 3 smiles variations are needed, and 4 are generated because none
                # correponds to original smiles and no-one is elimnated in previous if condition
                augmented = random.sample(augmented, AUG_SMILES_VARIATIONS - 1)

            # original smiles always first in the list
            return [smiles] + list(augmented)

        data_df["SMILES"] = data_df["SMILES"].apply(generate_augmented_smiles)

        # Explode the list of augmented smiles into multiple rows
        # augmented smiles will have same ident, as of the original, but does it matter ?
        # instead its helpful to group augmented smiles generated from the same original SMILES
        exploded_df = data_df.explode("SMILES").reset_index(drop=True)
        self.save_processed(
            exploded_df, self.processed_main_file_names_dict["aug_data"]
        )
        self._data_pkl_filename = aug_pkl_file_name

    # ------------------------------ Phase: Setup data -----------------------------------
    def setup_processed(self) -> None:
        """
        Transform and prepare processed data for the ChEBI dataset.

        Main function of this method is to transform `data.pkl` into a model input data format (`data.pt`),
        ensuring that the data is in a format compatible for input to the model.
        The transformed data must contain the following keys: `ident`, `features`, `labels`, and `group`.
        This method uses a subclass of Data Reader to perform the transformation.

        It will transform the data related to `chebi_version_train`, if specified.
        """
        super().setup_processed()

        # Transform the data related to "chebi_version_train" to encoded data, if it doesn't exist
        if self.chebi_version_train is not None and not os.path.isfile(
            os.path.join(
                self._chebi_version_train_obj.processed_dir,
                self._chebi_version_train_obj.processed_file_names_dict["data"],
            )
        ):
            print(
                f"Missing encoded data related to train version: {self.chebi_version_train}"
            )
            print("Calling the setup method related to it")
            self._chebi_version_train_obj.setup()

    def _load_dict(self, input_file_path: str) -> Generator[dict[str, Any], None, None]:
        """
        Loads a dictionary from a pickled file, yielding individual dictionaries for each row.

        This method reads data from a specified pickled file, processes each row to extract relevant
        information, and yields dictionaries containing the keys `features`, `labels`, and `ident`.
        If `single_class` is specified, it only includes the label for that specific class; otherwise,
        it includes labels for all classes starting from the fourth column.

        The pickled file is expected to contain rows with the following structure:
            - Data at row index `self._ID_IDX`: ID of the chebi data instance
            - Data at row index `self._DATA_REPRESENTATION_IDX` : SMILES representation for the chemical
            - Data from row index `self._LABELS_START_IDX` onwards: Labels

        This method is used in `_load_data_from_file` to process each row of data and convert it
        into the desired dictionary format before loading it into the model.

        Args:
            input_file_path (str): The path to the input pickled file.

        Yields:
            Dict[str, Any]: A dictionary with keys `features`, `labels`, and `ident`.
            `features` contains the sequence, `labels` contains the labels as boolean values,
            and `ident` contains the identifier.
        """
        with open(input_file_path, "rb") as input_file:
            df = pd.read_pickle(input_file)

            if self.single_class is None:
                all_labels = df.iloc[:, self._LABELS_START_IDX :].to_numpy(dtype=bool)
            else:
                single_cls_index = df.columns.get_loc(int(self.single_class))
                all_labels = df.iloc[:, [single_cls_index]].to_numpy(dtype=bool)

            features = df.iloc[:, self._DATA_REPRESENTATION_IDX].to_numpy()
            idents = df.iloc[:, self._ID_IDX].to_numpy()

            for feat, labels, ident in zip(features, all_labels, idents):
                yield dict(features=feat, labels=labels, ident=ident)

    # ------------------------------ Phase: Dynamic Splits -----------------------------------
    def _get_data_splits(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Loads encoded/transformed data and generates training, validation, and test splits.

        This method first loads encoded data from a file named `data.pt`, which is derived from either
        `chebi_version` or `chebi_version_train`. It then splits the data into training, validation, and test sets.

        If `chebi_version_train` is provided:
            - Loads additional encoded data from `chebi_version_train`.
            - Splits this data into training and validation sets, while using the test set from `chebi_version`.
            - Prunes the test set from `chebi_version` to include only labels that exist in `chebi_version_train`.

        If `chebi_version_train` is not provided:
            - Splits the data from `chebi_version` into training, validation, and test sets without modification.

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
            data_chebi_version = self.load_processed_data_from_file(
                os.path.join(self.processed_dir, filename)
            )
        except FileNotFoundError:
            raise FileNotFoundError(
                "File data.pt doesn't exists. "
                "Please call 'prepare_data' and/or 'setup' methods to generate the dataset files"
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
                    ),
                    weights_only=False,
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

        return df_train, df_val, df_test

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
        classes_file_name = "classes.txt"

        # Load original and new classes
        with open(os.path.join(self.processed_dir_main, classes_file_name), "r") as f:
            orig_classes = f.readlines()
        with open(
            os.path.join(
                self._chebi_version_train_obj.processed_dir_main, classes_file_name
            ),
            "r",
        ) as f:
            new_classes = f.readlines()

        # Mapping array (-1 means no match in new classes)
        mapping_array = np.array(
            [
                -1 if oc not in new_classes else new_classes.index(oc)
                for oc in orig_classes
            ],
            dtype=int,
        )

        # Convert labels column to 2D NumPy array
        labels_matrix = np.array(df_test_chebi_version["labels"].tolist(), dtype=bool)

        # Allocate new labels matrix
        num_new_classes = len(new_classes)
        new_labels_matrix = np.zeros(
            (labels_matrix.shape[0], num_new_classes), dtype=bool
        )

        # Copy only valid columns
        valid_mask = mapping_array != -1
        new_labels_matrix[:, mapping_array[valid_mask]] = labels_matrix[:, valid_mask]

        # Assign back
        df_test_chebi_version["labels"] = new_labels_matrix.tolist()
        return df_test_chebi_version

    # ------------------------------ Phase: Raw Properties -----------------------------------
    @property
    def base_dir(self) -> str:
        """
        Return the base directory path for data.

        Returns:
            str: The base directory path for data.
        """
        return os.path.join("data", f"chebi_v{self.chebi_version}")

    @property
    def processed_dir_main(self) -> str:
        """
        Returns the main directory path where processed data is stored.

        Returns:
            str: The path to the main processed data directory, based on the base directory and the instance's name.
        """
        return os.path.join(
            self.base_dir,
            self._name if self.subset is None else f"{self._name}_{self.subset}",
            "processed",
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
    def raw_file_names_dict(self) -> dict:
        return {"chebi": "chebi.obo"}

    @property
    def processed_main_file_names_dict(self) -> dict:
        """
        Returns a dictionary mapping processed data file names.

        Returns:
            dict: A dictionary mapping dataset key to their respective file names.
                  For example, {"data": "data.pkl"}.
        """
        p_dict = super().processed_main_file_names_dict
        if self.augment_smiles:
            p_dict["aug_data"] = f"aug_data_var{self.aug_smiles_variations}.pkl"
        return p_dict

    @property
    def processed_file_names_dict(self) -> dict:
        """
        Returns a dictionary for the processed and tokenized data files.

        Returns:
            dict: A dictionary mapping dataset keys to their respective file names.
                  For example, {"data": "data.pt"}.
        """
        if not self.augment_smiles:
            return super().processed_file_names_dict
        if self.n_token_limit is not None:
            return {
                "data": f"aug_data_var{self.aug_smiles_variations}_maxlen{self.n_token_limit}.pt"
            }
        return {"data": f"aug_data_var{self.aug_smiles_variations}.pt"}


class JCIExtendedBase(_ChEBIDataExtractor):
    @property
    def _name(self):
        return "JCI_extended"

    def select_classes(self, g, *args, **kwargs):
        return JCI_500_COLUMNS_INT


class ChEBIOverX(_ChEBIDataExtractor):
    """
    A class for extracting data from the ChEBI dataset with a threshold for selecting classes.
    This class is designed to filter Chebi classes based on a specified threshold, selecting only those classes
    which have a certain number of subclasses in the hierarchy.

    Attributes:
        LABEL_INDEX (int): The index of the label in the dataset.
        SMILES_INDEX (int): The index of the SMILES string in the dataset.
        READER (ChemDataReader): The reader used for reading the dataset.
        THRESHOLD (None): The threshold for selecting classes.
    """

    READER: dr.ChemDataReader = dr.ChemDataReader
    THRESHOLD: int = None

    @property
    def _name(self) -> str:
        """
        Returns the name of the dataset.

        Returns:
            str: The dataset name.
        """
        return f"ChEBI{self.THRESHOLD}"

    def select_classes(self, g: "nx.DiGraph", *args, **kwargs) -> List:
        """
        Selects classes from the ChEBI dataset based on the number of successors meeting a specified threshold.

        This method iterates over the nodes in the graph, counting the number of successors for each node.
        Nodes with a number of successors greater than or equal to the defined threshold are selected.

        Note:
            The input graph must be transitive closure of a directed acyclic graph.

        Args:
            g (nx.Graph): The graph representing the dataset.
            *args: Additional positional arguments (not used).
            **kwargs: Additional keyword arguments (not used).

        Returns:
            List: A sorted list of node IDs that meet the successor threshold criteria.

        Side Effects:
            Writes the list of selected nodes to a file named "classes.txt" in the specified processed directory.

        Notes:
            - The `THRESHOLD` attribute should be defined in the subclass of this class.
            - Nodes without a 'smiles' attribute are ignored in the successor count.
        """
        import networkx as nx

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


class ChEBIOver50(ChEBIOverX):
    """
    A class for extracting data from the ChEBI dataset with a threshold of 50 for selecting classes.

    Inherits from ChEBIOverX.

    Attributes:
        THRESHOLD (int): The threshold for selecting classes (50).
    """

    THRESHOLD: int = 50


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

    def __init__(self, top_class_id: int, external_data_ratio: float, **kwargs):
        """
        Initializes the ChEBIOverXPartial dataset.

        Args:
            top_class_id (int): The ID of the top class from which to extract subclasses.
            **kwargs: Additional keyword arguments passed to the superclass initializer.
            external_data_ratio (float): How much external data (i.e., samples where top_class_id
            is no positive label) to include in the dataset. 0 means no external data, 1 means
            the maximum amount (i.e., the complete ChEBI dataset).
        """
        if "top_class_id" not in kwargs:
            kwargs["top_class_id"] = top_class_id
        if "external_data_ratio" not in kwargs:
            kwargs["external_data_ratio"] = external_data_ratio

        self.top_class_id: int = top_class_id
        self.external_data_ratio: float = external_data_ratio
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
            f"partial_{self.top_class_id}_ext_ratio_{self.external_data_ratio:.2f}",
            "processed",
        )

    def _extract_class_hierarchy(self, chebi_path: str) -> "nx.DiGraph":
        """
        Extracts a subset of ChEBI based on subclasses of the top class ID.

        This method calls the superclass method to extract the full class hierarchy,
        then extracts the subgraph containing only the descendants of the top class ID, including itself.

        Args:
            chebi_path (str): The file path to the ChEBI ontology file.

        Returns:
            nx.DiGraph: The extracted class hierarchy as a directed graph, limited to the
            descendants of the top class ID.
        """
        g = super()._extract_class_hierarchy(chebi_path)
        top_class_successors = list(g.successors(self.top_class_id)) + [
            self.top_class_id
        ]
        external_nodes = list(set(n for n in g.nodes if n not in top_class_successors))
        if 0 < self.external_data_ratio < 1:
            n_external_nodes = int(
                len(top_class_successors)
                * self.external_data_ratio
                / (1 - self.external_data_ratio)
            )
            print(
                f"Extracting {n_external_nodes} external nodes from the ChEBI dataset (ratio: {self.external_data_ratio:.2f})"
            )
            external_nodes = external_nodes[: int(n_external_nodes)]
        elif self.external_data_ratio == 0:
            external_nodes = []

        g = g.subgraph(top_class_successors + external_nodes)
        print(
            f"Subgraph contains {len(g.nodes)} nodes, of which {len(top_class_successors)} are subclasses of the top class ID {self.top_class_id}."
        )
        return g

    def select_classes(self, g: "nx.DiGraph", *args, **kwargs) -> List:
        """Only selects classes that meet the threshold AND are subclasses of the top class ID (including itself)."""
        import networkx as nx

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
                    and (
                        self.top_class_id in g.predecessors(node)
                        or node == self.top_class_id
                    )
                }
            )
        )
        filename = "classes.txt"
        with open(os.path.join(self.processed_dir_main, filename), "wt") as fout:
            fout.writelines(str(node) + "\n" for node in nodes)
        return nodes


class ChEBIOver50Partial(ChEBIOverXPartial, ChEBIOver50):
    """
    Dataset that extracts a part of ChEBI based on subclasses of a given top class,
    with a threshold of 50 for selecting classes.

    Inherits from ChEBIOverXPartial and ChEBIOver50.
    """

    pass


class ChEBIOverXFingerprints(ChEBIOverX):
    """A class that uses Fingerprints for the processed data (used for fixed-length ML models)."""

    READER = dr.FingerprintReader


class ChEBIOver100Fingerprints(ChEBIOverXFingerprints, ChEBIOver100):
    """
    A class for extracting data from the ChEBI dataset with Fingerprints reader and a threshold of 100.

    Inherits from ChEBIOverXFingerprints and ChEBIOver100.
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


def term_callback(doc: "fastobo.term.TermFrame") -> Union[Dict, bool]:
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
    import fastobo

    parts = set()
    parents = []
    name = None
    smiles = None
    subset = None
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
        elif isinstance(clause, fastobo.term.SubsetClause):
            subset = str(clause.subset)

        if isinstance(clause, fastobo.term.IsObsoleteClause):
            if clause.obsolete:
                # if the term document contains clause as obsolete as true, skips this document.
                return False

    return {
        "id": chebi_to_int(str(doc.id)),
        "parents": parents,
        "has_part": parts,
        "name": name,
        "smiles": smiles,
        "subset": subset,
    }


atom_index = (
    r"\*",
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

if __name__ == "__main__":
    data_module_05 = ChEBIOver50Partial(
        chebi_version=241,
        splits_file_path=os.path.join(
            "data", "chebi_v241", "ChEBI50", "splits_80_10_10.csv"
        ),
        top_class_id=22712,
        external_data_ratio=0.5,
    )
    data_module_05.prepare_data()
    data_module_05.setup()
