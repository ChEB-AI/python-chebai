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
        """
        Reads and tokenizes raw SMILES data into a list of token indices.

        Args:
            raw_data (str): The raw SMILES string to be tokenized.

        Returns:
            List[int]: A list of integers representing the indices of the SMILES tokens.
        """
        return [self._get_token_index(v[1]) for v in _tokenize(raw_data)]

    def on_finish(self) -> None:
        """
        Saves the current cache of tokens to the token file. This method is called after all data processing is complete.
        """
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
    Data reader for protein sequences using amino acid tokens. This class processes raw protein sequences into a format
    suitable for model input by tokenizing them and assigning unique indices to each token.

    Note:
        Refer for amino acid sequence:  https://en.wikipedia.org/wiki/Protein_primary_structure

    Args:
        collator_kwargs (Optional[Dict[str, Any]]): Optional dictionary of keyword arguments for configuring the collator.
        token_path (Optional[str]): Path to the token file. If not provided, it will be created automatically.
        kwargs: Additional keyword arguments.
    """

    COLLATOR = RaggedCollator

    # 20 natural amino acid notation
    AA_LETTER = [
        "A",
        "R",
        "N",
        "D",
        "C",
        "Q",
        "E",
        "G",
        "H",
        "I",
        "L",
        "K",
        "M",
        "F",
        "P",
        "S",
        "T",
        "W",
        "Y",
        "V",
    ]

    @classmethod
    def name(cls) -> str:
        """
        Returns the name of the data reader. This method identifies the specific type of data reader.

        Returns:
            str: The name of the data reader, which is "protein_token".
        """
        return "protein_token"

    def __init__(self, *args, n_gram: Optional[int] = None, **kwargs):
        """
        Initializes the ProteinDataReader, loading existing tokens from the specified token file.

        Args:
            *args: Additional positional arguments passed to the base class.
            **kwargs: Additional keyword arguments passed to the base class.
        """
        if n_gram is not None:
            assert (
                int(n_gram) >= 2
            ), "Ngrams must be greater than or equal to 2 if provided."
            self.n_gram = int(n_gram)
        else:
            self.n_gram = None

        super().__init__(*args, **kwargs)

        # Load the existing tokens from the token file into a cache
        with open(self.token_path, "r") as pk:
            self.cache = [x.strip() for x in pk]

    def _get_token_index(self, token: str) -> int:
        """
        Returns a unique index for each token (amino acid). If the token is not already in the cache, it is added.

        Args:
            token (str): The amino acid token to retrieve or add.

        Returns:
            int: The index of the token, offset by the predefined EMBEDDING_OFFSET.
        """
        error_str = (
            f"Please ensure that the input only contains valid amino acids "
            f"20 Valid natural amino acid notation:  {self.AA_LETTER}"
            f"Refer to the amino acid sequence details here: "
            f"https://en.wikipedia.org/wiki/Protein_primary_structure"
        )

        if self.n_gram is None:
            # Single-letter amino acid token check
            if str(token) not in self.AA_LETTER:
                raise KeyError(f"Invalid token '{token}' encountered. " + error_str)
        else:
            # n-gram token validation, ensure that each component of the n-gram is valid
            for aa in token:
                if aa not in self.AA_LETTER:
                    raise KeyError(
                        f"Invalid token '{token}' encountered as part of n-gram {self.n_gram}. "
                        + error_str
                    )

        if str(token) not in self.cache:
            self.cache.append(str(token))
        return self.cache.index(str(token)) + EMBEDDING_OFFSET

    def _read_data(self, raw_data: str) -> List[int]:
        """
        Reads and tokenizes raw protein sequence data into a list of token indices.

        Args:
            raw_data (str): The raw protein sequence to be tokenized (e.g., "MKTFF...").

        Returns:
            List[int]: A list of integers representing the indices of the amino acid tokens.
        """
        if self.n_gram is not None:
            # Tokenize the sequence into n-grams
            tokens = [
                raw_data[i : i + self.n_gram]
                for i in range(len(raw_data) - self.n_gram + 1)
            ]
            return [self._get_token_index(gram) for gram in tokens]

        # If n_gram is None, tokenize the sequence at the amino acid level (single-letter representation)
        return [self._get_token_index(aa) for aa in raw_data]

    def on_finish(self) -> None:
        """
        Saves the current cache of tokens to the token file. This method is called after all data processing is complete.
        """
        with open(self.token_path, "w") as pk:
            print(f"Saving {len(self.cache)} tokens to {self.token_path}...")
            print(f"First 10 tokens: {self.cache[:10]}")
            pk.writelines([f"{c}\n" for c in self.cache])
