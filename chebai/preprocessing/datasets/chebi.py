__all__ = [
    "ChEBIOver50",
    "ChEBIOver100",
]

import os
import random
from abc import ABC
from collections import OrderedDict
from itertools import cycle, permutations, product
from typing import TYPE_CHECKING, Any, Dict, Generator, List, Literal, Optional, Union

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from tqdm import tqdm

from chebai.preprocessing import reader as dr
from chebai.preprocessing.datasets.base import _DynamicDataset

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
    # "mol" at                row index 3
    # labels starting from    row index 4
    _ID_IDX: int = 0
    _DATA_REPRESENTATION_IDX: int = 3
    _LABELS_START_IDX: int = 4

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
        Downloads the required raw data related to ChEBI.

        Returns:
            str: Path to the ontology file.
        """
        self._load_sdf()
        return self._load_chebi()

    def _load_chebi(self, version: Optional[int] = None) -> str:
        """
        Load the ChEBI ontology file.

        Args:
            version (int): The version of the ChEBI ontology to load. Default: self.chebi_version

        Returns:
            str: The file path of the loaded ChEBI ontology.
        """
        import requests

        if version is None:
            version = self.chebi_version

        chebi_name = self.raw_file_names_dict["chebi"]
        chebi_path = os.path.join(self.raw_dir, chebi_name)
        if not os.path.isfile(chebi_path):
            print(
                f"Missing raw ChEBI data related for version v{version}. Downloading..."
            )
            if version < 245:
                url = f"https://ftp.ebi.ac.uk/pub/databases/chebi/archive/chebi_legacy/archive/rel{version}/ontology/chebi.obo"
            else:
                url = f"https://ftp.ebi.ac.uk/pub/databases/chebi/archive/rel{version}/ontology/chebi.obo"
            r = requests.get(url, allow_redirects=True)
            open(chebi_path, "wb").write(r.content)
        return chebi_path

    def _load_sdf(self, version: Optional[int] = None) -> str:
        """
        Load the ChEBI SDF file containing molecule data.

        Args:
            version (int): The version of the ChEBI SDF file to load. Default: self.chebi_version

        Returns:
            str: The file path of the loaded ChEBI SDF file.
        """
        import requests
        import gzip
        import shutil

        if version is None:
            version = self.chebi_version

        sdf_name = self.raw_file_names_dict["sdf"]
        sdf_path = os.path.join(self.raw_dir, sdf_name)
        if not os.path.isfile(sdf_path):
            print(f"Missing raw SDF data related to version v{version}. Downloading...")
            if version < 245:
                url = f"https://ftp.ebi.ac.uk/pub/databases/chebi/archive/chebi_legacy/archive/rel{version}/ontology/chebi.obo"
            else:
                url = f"https://ftp.ebi.ac.uk/pub/databases/chebi/archive/rel{version}/SDF/chebi.sdf.gz"
            r = requests.get(url, allow_redirects=True, stream=True)
            open(sdf_path + ".gz", "wb").write(r.content)
            with gzip.open(sdf_path + ".gz", "rb") as f_in:
                with open(sdf_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
        return sdf_path

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
                    not self.subset
                    or (
                        "subset" in term_dict
                        and term_dict["subset"] is not None
                        and term_dict["subset"][0] == self.subset[0]
                    )  # match 3:STAR to 3_STAR, 3star, 3_star, etc.
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

        print(f"Processing {g}")

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

        # # `mol` (RDKit Mol object) column at index 3
        from chembl_structure_pipeline.standardizer import (
            parse_molblock,
        )

        with open(
            os.path.join(self.raw_dir, self.raw_file_names_dict["sdf"]), "rb"
        ) as f:
            # split input into blocks separated by "$$$$"
            blocks = f.read().decode("utf-8").split("$$$$\n")
        id_to_mol = dict()
        for molfile in tqdm(blocks, desc="Processing SDF molecules"):
            if "<ChEBI ID>" not in molfile:
                print(f"Skipping molfile without ChEBI ID: {molfile[:30]}...")
                continue
            ident = int(molfile.split("<ChEBI ID>")[1].split(">")[0].split("CHEBI:")[1])
            # use same parsing strategy as CHEBI: github.com/chembl/libRDChEBI/blob/main/libRDChEBI/formats.py
            mol = parse_molblock(molfile)
            if mol is None:
                print(f"Failed to parse molfile for CHEBI:{ident}")
                continue
            mol = sanitize_molecule(mol)
            id_to_mol[ident] = mol
        data["mol"] = [id_to_mol.get(node) for node in molecules]

        # Labels columns from index 4 onwards
        for n in self.select_classes(g):
            data[n] = [
                ((n in g.predecessors(node)) or (n == node)) for node in molecules
            ]

        data = pd.DataFrame(data)
        data = data[~data["mol"].isnull()]
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
            data_chebi_version = self.load_processed_data_from_file(filename)
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
        return {"chebi": "chebi.obo", "sdf": "chebi.sdf"}

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
            # chemrof:smiles_string is the new annotation property, chebi/smiles is the old one (see https://chembl.blogspot.com/2025/07/chebi-20-data-products.html)
            if (
                str(t.relation) == "chemrof:smiles_string"
                or str(t.relation) == "http://purl.obolibrary.org/obo/chebi/smiles"
            ):
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


def sanitize_molecule(mol: Chem.Mol) -> Chem.Mol:
    # mirror ChEBI molecule processing
    from chembl_structure_pipeline.standardizer import update_mol_valences

    mol = update_mol_valences(mol)
    Chem.SanitizeMol(
        mol,
        sanitizeOps=Chem.SanitizeFlags.SANITIZE_FINDRADICALS
        | Chem.SanitizeFlags.SANITIZE_KEKULIZE
        | Chem.SanitizeFlags.SANITIZE_SETAROMATICITY
        | Chem.SanitizeFlags.SANITIZE_SETCONJUGATION
        | Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION
        | Chem.SanitizeFlags.SANITIZE_SYMMRINGS,
        catchErrors=True,
    )
    return mol


if __name__ == "__main__":
    dataset = ChEBIOver50(chebi_version=248, subset="3_STAR")
    dataset.prepare_data()
    dataset.setup()
