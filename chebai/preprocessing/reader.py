import inspect
import os
import sys
from abc import ABC
from itertools import islice
from typing import Any, Dict, List, Optional

from pysmiles.read_smiles import _tokenize
from rdkit import Chem

from chebai.preprocessing.collate import DefaultCollator, RaggedCollator

EMBEDDING_OFFSET = 10
PADDING_TOKEN_INDEX = 0
MASK_TOKEN_INDEX = 1
CLS_TOKEN = 2


class DataReader:
    """
    Base class for reading and preprocessing data. Turns the raw input data (e.g., a SMILES string) into the model
    input format (e.g., a list of tokens).

    Args:
        collator_kwargs: Optional dictionary of keyword arguments for the collator.
        token_path: Optional path for the token file.
        kwargs: Additional keyword arguments (not used).
    """

    COLLATOR = DefaultCollator

    def __init__(
        self,
        collator_kwargs: Optional[Dict[str, Any]] = None,
        token_path: Optional[str] = None,
        **kwargs,
    ):
        if collator_kwargs is None:
            collator_kwargs = dict()
        self.collator = self.COLLATOR(**collator_kwargs)
        self.dirname = os.path.dirname(inspect.getfile(self.__class__))
        self._token_path = token_path

    def _get_raw_data(self, row: Dict[str, Any]) -> Any:
        """Get raw data from the row."""
        return row["features"]

    def _get_raw_label(self, row: Dict[str, Any]) -> Any:
        """Get raw label from the row."""
        return row["labels"]

    def _get_raw_id(self, row: Dict[str, Any]) -> Any:
        """Get raw ID from the row."""
        return row.get("ident", row["features"])

    def _get_raw_group(self, row: Dict[str, Any]) -> Any:
        """Get raw group from the row."""
        return row.get("group", None)

    def _get_additional_kwargs(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Get additional keyword arguments from the row."""
        return row.get("additional_kwargs", dict())

    def name(cls) -> str:
        """Returns the name of the data reader."""
        raise NotImplementedError

    @property
    def token_path(self) -> str:
        """Get token path, create file if it does not exist yet."""
        if self._token_path is not None:
            return self._token_path
        token_path = os.path.join(self.dirname, "bin", self.name(), "tokens.txt")
        os.makedirs(os.path.join(self.dirname, "bin", self.name()), exist_ok=True)
        if not os.path.exists(token_path):
            with open(token_path, "x"):
                pass
        return token_path

    def _read_id(self, raw_data: Any) -> Any:
        """Read and return ID from raw data."""
        return raw_data

    def _read_data(self, raw_data: Any) -> Any:
        """Read and return data from raw data."""
        return raw_data

    def _read_label(self, raw_label: Any) -> Any:
        """Read and return label from raw label."""
        return raw_label

    def _read_group(self, raw: Any) -> Any:
        """Read and return group from raw group data."""
        return raw

    def _read_components(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Read and return components from the row. If the data contains any missing labels (`None`), they are tracked
        under the additional `missing_labels` keyword."""
        labels = self._get_raw_label(row)
        additional_kwargs = self._get_additional_kwargs(row)
        if any(label is None for label in labels):
            additional_kwargs["missing_labels"] = [label is None for label in labels]
        return dict(
            features=self._get_raw_data(row),
            labels=labels,
            ident=self._get_raw_id(row),
            group=self._get_raw_group(row),
            additional_kwargs=additional_kwargs,
        )

    def to_data(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Convert raw row data to processed data."""
        d = self._read_components(row)
        return dict(
            features=self._read_data(d["features"]),
            labels=self._read_label(d["labels"]),
            ident=self._read_id(d["ident"]),
            group=self._read_group(d["group"]),
            **d["additional_kwargs"],
        )

    def on_finish(self) -> None:
        """Hook to run at the end of preprocessing."""
        return


class TokenIndexerReader(DataReader, ABC):
    """
    Abstract base class for reading tokenized data and mapping tokens to unique indices.

    This class maintains a cache of token-to-index mappings that can be extended during runtime,
    and saves new tokens to a persistent file at the end of processing.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        with open(self.token_path, "r") as pk:
            self.cache: Dict[str, int] = {
                token.strip(): idx for idx, token in enumerate(pk)
            }
        self._loaded_tokens_count = len(self.cache)

    def _get_token_index(self, token: str) -> int:
        """Returns a unique number for each token, automatically adds new tokens."""
        if str(token) not in self.cache:
            self.cache[(str(token))] = len(self.cache)
        return self.cache[str(token)] + EMBEDDING_OFFSET

    def on_finish(self) -> None:
        """
        Saves the current cache of tokens to the token file.This method is called after all data processing is complete.
        """
        print(f"first 10 tokens: {list(islice(self.cache, 10))}")

        total_tokens = len(self.cache)
        if total_tokens > self._loaded_tokens_count:
            print("New tokens added to the cache, Saving them to token file.....")

            assert sys.version_info >= (
                3,
                7,
            ), "This code requires Python 3.7 or higher."
            # For python 3.7+, the standard dict type preserves insertion order, and is iterated over in same order
            # https://docs.python.org/3/whatsnew/3.7.html#summary-release-highlights
            # https://mail.python.org/pipermail/python-dev/2017-December/151283.html
            new_tokens = list(
                islice(self.cache, self._loaded_tokens_count, total_tokens)
            )

            with open(self.token_path, "a") as pk:
                print(f"saving new {len(new_tokens)} tokens to {self.token_path}...")
                pk.writelines([f"{c}\n" for c in new_tokens])


class ChemDataReader(TokenIndexerReader):
    """
    Data reader for chemical data using SMILES tokens.
    """

    COLLATOR = RaggedCollator

    def __init__(self, canonicalize_smiles=True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.canonicalize_smiles = canonicalize_smiles
        print(f"Using SMILES canonicalization: {self.canonicalize_smiles}")

    @classmethod
    def name(cls) -> str:
        """Returns the name of the data reader."""
        return "smiles_token"

    def _read_data(self, raw_data: str) -> List[int]:
        """
        Reads and tokenizes raw SMILES data into a list of token indices. Canonicalizes the SMILES string using RDKit.

        Args:
            raw_data (str): The raw SMILES string to be tokenized.

        Returns:
            List[int]: A list of integers representing the indices of the SMILES tokens.
        """
        try:
            mol = Chem.MolFromSmiles(raw_data.strip())
            if mol is None:
                raise ValueError(f"Invalid SMILES: {raw_data}")
        except ValueError as e:
            print(f"could not process {raw_data}")
            print(f"\tError: {e}")
            return None

        if self.canonicalize_smiles:
            try:
                raw_data = Chem.MolToSmiles(mol, canonical=True)
            except Exception as e:
                print(f"RDKit failed to canonicalize the SMILES: {raw_data}")
                print(f"\t{e}")
                return None

        return [self._get_token_index(v[1]) for v in _tokenize(raw_data)]

    def _back_to_smiles(self, smiles_encoded):
        token_file = self.reader.token_path
        token_coding = {}
        counter = 0
        smiles_decoded = ""

        # todo: for now just copied over from a notebook but ideally do this using the cache
        with open(token_file, "r") as file:
            for line in file:
                token_coding[counter] = line.strip()
                counter += 1

        for token in smiles_encoded:
            smiles_decoded += token_coding[token - EMBEDDING_OFFSET]

        return smiles_decoded


class DeepChemDataReader(ChemDataReader):
    """
    Data reader for chemical data using DeepSMILES tokens.

    Args:
        collator_kwargs: Optional dictionary of keyword arguments for the collator.
        token_path: Optional path for the token file.
        kwargs: Additional keyword arguments.
    """

    def __init__(self, *args, **kwargs):
        import deepsmiles

        super().__init__(*args, **kwargs)
        self.converter = deepsmiles.Converter(rings=True, branches=True)
        self.error_count = 0

    @classmethod
    def name(cls) -> str:
        """Returns the name of the data reader."""
        return "deepsmiles_token"

    def _read_data(self, raw_data: str) -> Optional[List[int]]:
        """Read and tokenize raw data using DeepSMILES."""
        try:
            tokenized = _tokenize(self.converter.encode(raw_data))
            tokenized = [self._get_token_index(v[1]) for v in tokenized]
        except ValueError as e:
            print(f"could not process {raw_data}")
            print(f"Corresponding deepSMILES: {self.converter.encode(raw_data)}")
            print(f"\t{e}")
            self.error_count += 1
            print(f"\terror count: {self.error_count}")
            tokenized = None
        return tokenized


class ChemDataUnlabeledReader(ChemDataReader):
    """
    Data reader for unlabeled chemical data using SMILES tokens.

    Args:
        collator_kwargs: Optional dictionary of keyword arguments for the collator.
        token_path: Optional path for the token file.
        kwargs: Additional keyword arguments.
    """

    COLLATOR = RaggedCollator

    @classmethod
    def name(cls) -> str:
        """Returns the name of the data reader."""
        return "smiles_token_unlabeled"

    def _get_raw_label(self, row: Dict[str, Any]) -> None:
        """Returns None as there are no labels."""
        return None


class ChemBPEReader(DataReader):
    """
    Data reader for chemical data using BPE tokenization.

    Args:
        data_path: Path for the pretrained BPE tokenizer.
        max_len: Maximum length of the tokenized sequence.
        vsize: Vocabulary size for the tokenizer (not used).
        collator_kwargs: Optional dictionary of keyword arguments for the collator.
        token_path: Optional path for the token file.
        kwargs: Additional keyword arguments.
    """

    COLLATOR = RaggedCollator

    @classmethod
    def name(cls) -> str:
        """Returns the name of the data reader."""
        return "smiles_bpe"

    def __init__(
        self,
        *args,
        data_path: Optional[str] = None,
        max_len: int = 1800,
        vsize: int = 4000,
        **kwargs,
    ):
        from transformers import RobertaTokenizerFast

        super().__init__(*args, **kwargs)
        self.tokenizer = RobertaTokenizerFast.from_pretrained(
            data_path, max_len=max_len
        )

    def _get_raw_data(self, row: Dict[str, Any]) -> List[int]:
        """Tokenize raw data using BPE tokenizer."""
        return self.tokenizer(row["features"])["input_ids"]


class SelfiesReader(ChemDataReader):
    """
    Data reader for chemical data using SELFIES tokens.

    Args:
        data_path: Path for the pretrained BPE tokenizer.
        max_len: Maximum length of the tokenized sequence.
        vsize: Vocabulary size for the tokenizer.
        collator_kwargs: Optional dictionary of keyword arguments for the collator.
        token_path: Optional path for the token file.
        kwargs: Additional keyword arguments.
    """

    COLLATOR = RaggedCollator

    def __init__(
        self,
        *args,
        data_path: Optional[str] = None,
        max_len: int = 1800,
        vsize: int = 4000,
        **kwargs,
    ):
        import selfies as sf

        super().__init__(*args, **kwargs)
        self.error_count = 0
        sf.set_semantic_constraints("hypervalent")

    @classmethod
    def name(cls) -> str:
        """Returns the name of the data reader."""
        return "selfies"

    def _read_data(self, raw_data: str) -> Optional[List[int]]:
        """Read and tokenize raw data using SELFIES."""
        import selfies as sf

        try:
            tokenized = sf.split_selfies(sf.encoder(raw_data.strip(), strict=True))
            tokenized = [self._get_token_index(v) for v in tokenized]
        except Exception:
            print(f"could not process {raw_data}")
            # print(f'\t{e}')
            self.error_count += 1
            print(f"\terror count: {self.error_count}")
            tokenized = None
            # if self.error_count > 20:
            #    raise Exception('Too many errors')
        return tokenized


class OrdReader(DataReader):
    """
    Data reader that converts characters to their ordinal values.

    Args:
        collator_kwargs: Optional dictionary of keyword arguments for the collator.
        token_path: Optional path for the token file.
        kwargs: Additional keyword arguments.
    """

    COLLATOR = RaggedCollator

    @classmethod
    def name(cls) -> str:
        """Returns the name of the data reader."""
        return "ord"

    def _read_data(self, raw_data: str) -> List[int]:
        """Convert characters in raw data to their ordinal values."""
        return [ord(s) for s in raw_data]


class FingerprintReader(DataReader):
    """
    Data reader for chemical data using RDKit fingerprints.

    Args:
        collator_kwargs: Optional dictionary of keyword arguments for the collator.
        kwargs: Additional keyword arguments.
    """

    COLLATOR = DefaultCollator

    def __init__(self, fingerprint_size=1024, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fingerprint_size = fingerprint_size

    @classmethod
    def name(cls) -> str:
        """Returns the name of the data reader."""
        return "rdkit_fingerprint"

    def _read_data(self, raw_data: str) -> List[int]:
        """Generate RDKit fingerprint from raw SMILES data."""
        mol = Chem.MolFromSmiles(raw_data.strip())
        if mol is None:
            raise ValueError(f"Invalid SMILES: {raw_data}")
        return list(
            Chem.RDKFingerprint(mol, fpSize=self.fingerprint_size).ToBitString()
        )
