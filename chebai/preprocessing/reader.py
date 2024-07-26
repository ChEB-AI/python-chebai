import os
from typing import Any, Dict, List, Optional

import deepsmiles
import selfies as sf
from pysmiles.read_smiles import _tokenize
from transformers import RobertaTokenizerFast

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
        self.dirname = os.path.dirname(__file__)
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
        """Read and return components from the row."""
        return dict(
            features=self._get_raw_data(row),
            labels=self._get_raw_label(row),
            ident=self._get_raw_id(row),
            group=self._get_raw_group(row),
            additional_kwargs=self._get_additional_kwargs(row),
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


class ChemDataReader(DataReader):
    """
    Data reader for chemical data using SMILES tokens.

    Args:
        collator_kwargs: Optional dictionary of keyword arguments for the collator.
        token_path: Optional path for the token file.
        kwargs: Additional keyword arguments.
    """

    COLLATOR = RaggedCollator

    @classmethod
    def name(cls) -> str:
        """Returns the name of the data reader."""
        return "smiles_token"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        with open(self.token_path, "r") as pk:
            self.cache = [x.strip() for x in pk]

    def _get_token_index(self, token: str) -> int:
        """Returns a unique number for each token, automatically adds new tokens."""
        if not str(token) in self.cache:
            self.cache.append(str(token))
        return self.cache.index(str(token)) + EMBEDDING_OFFSET

    def _read_data(self, raw_data: str) -> List[int]:
        """Read and tokenize raw data."""
        return [self._get_token_index(v[1]) for v in _tokenize(raw_data)]

    def on_finish(self) -> None:
        """Write contents of self.cache into tokens.txt."""
        with open(self.token_path, "w") as pk:
            print(f"saving {len(self.cache)} tokens to {self.token_path}...")
            print(f"first 10 tokens: {self.cache[:10]}")
            pk.writelines([f"{c}\n" for c in self.cache])


class DeepChemDataReader(ChemDataReader):
    """
    Data reader for chemical data using DeepSMILES tokens.

    Args:
        collator_kwargs: Optional dictionary of keyword arguments for the collator.
        token_path: Optional path for the token file.
        kwargs: Additional keyword arguments.
    """

    def __init__(self, *args, **kwargs):
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
        super().__init__(*args, **kwargs)
        self.error_count = 0
        sf.set_semantic_constraints("hypervalent")

    @classmethod
    def name(cls) -> str:
        """Returns the name of the data reader."""
        return "selfies"

    def _read_data(self, raw_data: str) -> Optional[List[int]]:
        """Read and tokenize raw data using SELFIES."""
        try:
            tokenized = sf.split_selfies(sf.encoder(raw_data.strip(), strict=True))
            tokenized = [self._get_token_index(v) for v in tokenized]
        except Exception as e:
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


class ProteinDataReader(DataReader):
    """
    Data reader for Protein data using protein-sequence tokens.

    Args:
        collator_kwargs: Optional dictionary of keyword arguments for the collator.
        token_path: Optional path for the token file.
        kwargs: Additional keyword arguments.
    """

    COLLATOR = RaggedCollator

    @classmethod
    def name(cls) -> str:
        """Returns the name of the data reader."""
        return "sequence_token"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        with open(self.token_path, "r") as pk:
            self.cache = [x.strip() for x in pk]

    def _get_token_index(self, token: str) -> int:
        """Returns a unique number for each token, automatically adds new tokens."""
        if not str(token) in self.cache:
            self.cache.append(str(token))
        return self.cache.index(str(token)) + EMBEDDING_OFFSET

    def _read_data(self, raw_data: str) -> List[int]:
        """Read and tokenize raw data."""
        return [self._get_token_index(v[1]) for v in _tokenize(raw_data)]

    def on_finish(self) -> None:
        """Write contents of self.cache into tokens.txt."""
        with open(self.token_path, "w") as pk:
            print(f"saving {len(self.cache)} tokens to {self.token_path}...")
            print(f"first 3 sequences tokens: {self.cache[:3]}")
            for token in self.cache[:3]:
                print(f"Sequence Token: {token}")
            pk.writelines([f"{c}\n" for c in self.cache])
