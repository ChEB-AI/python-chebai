__all__ = [
    "ChEBIOver50",
    "ChEBIOver100",
]

import os
import random
from abc import ABC
from itertools import cycle, permutations, product
from typing import TYPE_CHECKING, Any, Generator, Literal, Optional

import numpy as np
import pandas as pd
from rdkit import Chem

from chebai.preprocessing import reader as dr
from chebai.preprocessing.datasets.base import _DynamicDataset
from chebai.preprocessing.datasets.ml_overbagging import _ResampledDynamicDataset

if TYPE_CHECKING:
    import networkx as nx


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
    # "mol" at                row index 1
    # labels starting from    row index 2
    _ID_IDX: int = 0
    _DATA_REPRESENTATION_IDX: int = 1
    _LABELS_START_IDX: int = 2
    THRESHOLD: int = None

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

        from rdkit import RDLogger

        RDLogger.DisableLog("rdApp.*")

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
        if version is None:
            version = self.chebi_version

        if version is None:
            version = self.chebi_version

        chebi_name = self.raw_file_names_dict["chebi"]
        chebi_path = os.path.join(self.raw_dir, chebi_name)
        if not os.path.isfile(chebi_path):
            print(
                f"Missing raw ChEBI data related for version v{version}. Downloading..."
            )
            from chebi_utils import download_chebi_obo

            download_chebi_obo(version, dest_dir=self.raw_dir, filename=chebi_name)
        return chebi_path

    def _load_sdf(self, version: Optional[int] = None) -> str:
        """
        Load the ChEBI SDF file containing molecule data.

        Args:
            version (int): The version of the ChEBI SDF file to load. Default: self.chebi_version

        Returns:
            str: The file path of the loaded ChEBI SDF file.
        """
        if version is None:
            version = self.chebi_version

        sdf_name = self.raw_file_names_dict["sdf"]
        sdf_path = os.path.join(self.raw_dir, sdf_name)
        if not os.path.isfile(sdf_path):
            print(f"Missing raw SDF data related to version v{version}. Downloading...")
            from chebi_utils import download_chebi_sdf

            download_chebi_sdf(version, dest_dir=self.raw_dir, filename=sdf_name)
        return sdf_path

    def _graph_to_raw_dataset(self, g: "nx.DiGraph") -> pd.DataFrame:
        """
        Converts the graph to a raw dataset.
        Uses the graph to extract the
        raw data in Dataframe format with additional columns corresponding to each multi-label class.

        Uses :func:`chebi_utils.sdf_extractor.extract_molecules` for SDF parsing.

        Args:
            g (nx.DiGraph): The class hierarchy graph.

        Returns:
            pd.DataFrame: The raw dataset created from the graph.
        """

        # Extract mol objects from SDF using chebi-utils
        from chebi_utils import build_labeled_dataset, extract_molecules

        sdf_path = os.path.join(self.raw_dir, self.raw_file_names_dict["sdf"])
        mol_df = extract_molecules(sdf_path)
        mol_df = mol_df[mol_df["STAR"] == self.subset[0]] if self.subset else mol_df
        data, labels = build_labeled_dataset(g, mol_df, self.THRESHOLD)

        with open(os.path.join(self.classes_txt_file_path), "wt") as fout:
            fout.writelines(str(label) + "\n" for label in labels)

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

    def _get_data_splits(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Loads encoded/transformed data and generates training, validation, and test splits.
        """

        filename = self.processed_file_names_dict["data"]
        data = self.load_processed_data_from_file(filename)
        df_data = pd.DataFrame(data)

        from chebi_utils import create_multilabel_splits

        splits = create_multilabel_splits(
            df_data,
            self._LABELS_START_IDX,
            1 - self.validation_split - self.test_split,
            self.validation_split,
            self.test_split,
            self.dynamic_data_split_seed,
        )
        return splits["train"], splits["val"], splits["test"]

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
        # Load original and new classes
        with open(os.path.join(self.classes_txt_file_path), "r") as f:
            orig_classes = f.readlines()
        with open(
            os.path.join(self._chebi_version_train_obj.classes_txt_file_path),
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
        return {"chebi": "chebi.obo", "sdf": "chebi.sdf.gz"}

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

    @property
    def _name(self) -> str:
        """
        Returns the name of the dataset.

        Returns:
            str: The dataset name.
        """
        return f"ChEBI{self.THRESHOLD}"


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


class ChEBI50Resampled(ChEBIOver50, _ResampledDynamicDataset):
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

    def __init__(self, top_class_id: str, external_data_ratio: float, **kwargs):
        """
        Initializes the ChEBIOverXPartial dataset.

        Args:
            top_class_id (str): The ID of the top class from which to extract subclasses.
            **kwargs: Additional keyword arguments passed to the superclass initializer.
            external_data_ratio (float): How much external data (i.e., samples where top_class_id
            is no positive label) to include in the dataset. 0 means no external data, 1 means
            the maximum amount (i.e., the complete ChEBI dataset).
        """
        if "top_class_id" not in kwargs:
            kwargs["top_class_id"] = top_class_id
        if "external_data_ratio" not in kwargs:
            kwargs["external_data_ratio"] = external_data_ratio

        self.top_class_id: str = top_class_id
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

    def _graph_to_raw_dataset(self, g: "nx.DiGraph") -> pd.DataFrame:
        """
        Converts the graph to a raw dataset.
        Uses the graph to extract the
        raw data in Dataframe format with additional columns corresponding to each multi-label class.

        Uses :func:`chebi_utils.sdf_extractor.extract_molecules` for SDF parsing.

        Args:
            g (nx.DiGraph): The class hierarchy graph.

        Returns:
            pd.DataFrame: The raw dataset created from the graph.
        """

        # Extract mol objects from SDF using chebi-utils
        from chebi_utils import (
            build_labeled_dataset,
            extract_molecules,
            get_hierarchy_subgraph,
        )
        import networkx as nx

        sdf_path = os.path.join(self.raw_dir, self.raw_file_names_dict["sdf"])
        mol_df = extract_molecules(sdf_path)
        mol_df = mol_df[mol_df["STAR"] == self.subset[0]] if self.subset else mol_df

        # take only molecules that are subclasses of the top class ID, and a certain ratio of external nodes (nodes that are not subclasses of the top class ID)
        transitive_closure = nx.transitive_closure_dag(get_hierarchy_subgraph(g))
        top_class_successors = list(
            transitive_closure.predecessors(self.top_class_id)
        ) + [self.top_class_id]
        top_class_molecules = mol_df[mol_df["chebi_id"].isin(top_class_successors)]
        external_molecules = mol_df[~mol_df["chebi_id"].isin(top_class_successors)]
        if 0 < self.external_data_ratio < 1:
            n_external_nodes = int(
                len(top_class_molecules)
                * self.external_data_ratio
                / (1 - self.external_data_ratio)
            )
            external_molecules = external_molecules.sample(
                n=min(n_external_nodes, len(external_molecules)),
                random_state=self.dynamic_data_split_seed,
            )
        elif self.external_data_ratio == 0:
            external_molecules = mol_df.iloc[0:0]
        mol_df = pd.concat([top_class_molecules, external_molecules], ignore_index=True)

        data, labels = build_labeled_dataset(g, mol_df, self.THRESHOLD)
        # the dataset might contain classes that are not subclasses of the top class ID
        labels_top_class = [label for label in labels if label in top_class_successors]
        data = data[["chebi_id", "mol"] + labels_top_class]

        with open(os.path.join(self.classes_txt_file_path), "wt") as fout:
            fout.writelines(str(label) + "\n" for label in labels_top_class)

        return data


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


if __name__ == "__main__":
    dataset = ChEBI50Resampled(
        chebi_version="248",
        splits_file_path=os.path.join(
            "data", "chebi_v248", "ChEBI50", "processed", "splits_chebi50_v248.csv"
        ),
        batch_size=32,
    )
    dataset.prepare_data()
    dataset.setup()
