import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.error import HTTPError

import deepsmiles
import selfies as sf
import torch
from esm import Alphabet
from esm.model.esm2 import ESM2
from esm.pretrained import (
    _has_regression_weights,
    load_model_and_alphabet_core,
    load_model_and_alphabet_local,
)
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

    # 21 natural amino acid notation
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
        # https://github.com/bio-ontology-research-group/deepgo2/blob/main/deepgo/aminoacids.py#L3-L5
        "X",  # Consider valid in latest paper year 2024 Reference number 3 in go_uniprot.py
    ]

    def name(self) -> str:
        """
        Returns the name of the data reader. This method identifies the specific type of data reader.

        Returns:
            str: The name of the data reader, which is "protein_token".
        """
        if self.n_gram is not None:
            return f"protein_token_{self.n_gram}_gram"

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


class ESM2EmbeddingReader(DataReader):
    """
    A data reader to process protein sequences using the ESM2 model for embeddings.

    References:
        https://github.com/bio-ontology-research-group/deepgo2/blob/main/deepgo/extract_esm.py

    Note:
        For layer availability by model, Please check below link:
            https://github.com/facebookresearch/esm?tab=readme-ov-file#pre-trained-models-

        To test this reader, try lighter models:
            esm2_t6_8M_UR50D: 6 layers (valid layers: 1–6),  (~28 Mb) - A tiny 8M parameter model.
            esm2_t12_35M_UR50D: 12 layers (valid layers: 1–12),  (~128 Mb) - A slightly larger, 35M parameter model.
        These smaller models are good for testing and debugging purposes.

    """

    # https://github.com/facebookresearch/esm/blob/main/esm/pretrained.py#L53
    _MODELS_URL = "https://dl.fbaipublicfiles.com/fair-esm/models/{}.pt"
    _REGRESSION_URL = (
        "https://dl.fbaipublicfiles.com/fair-esm/regression/{}-contact-regression.pt"
    )

    def __init__(
        self,
        save_model_dir: str = os.path.join("data", "esm2_reader"),
        model_name: str = "esm2_t36_3B_UR50D",
        device: Optional[torch.device] = None,
        truncation_length: int = 1022,
        toks_per_batch: int = 4096,
        return_contacts: bool = False,
        repr_layer: int = 36,
        *args,
        **kwargs,
    ):
        """
        Initialize the ESM2EmbeddingReader class.

        Args:
            save_model_dir (str): Directory to save/load the pretrained ESM model.
            model_name (str): Name of the pretrained model. Defaults to "esm2_t36_3B_UR50D".
            device (torch.device or str, optional): Device for computation (e.g., 'cpu', 'cuda').
            truncation_length (int): Maximum sequence length for truncation. Defaults to 1022.
            toks_per_batch (int): Tokens per batch for data processing. Defaults to 4096.
            return_contacts (bool): Whether to return contact maps. Defaults to False.
            repr_layers (int): Layer number to extract representations from. Defaults to 36.
        """
        self.save_model_dir = save_model_dir
        if not os.path.exists(self.save_model_dir):
            os.makedirs((os.path.dirname(self.save_model_dir)), exist_ok=True)
        self.model_name = model_name
        self.device = device
        self.truncation_length = truncation_length
        self.toks_per_batch = toks_per_batch
        self.return_contacts = return_contacts
        self.repr_layer = repr_layer

        self._model: Optional[ESM2] = None
        self._alphabet: Optional[Alphabet] = None

        self._model, self._alphabet = self.load_model_and_alphabet()
        self._model.eval()

        if self.device:
            self._model = self._model.to(device)

        super().__init__(*args, **kwargs)

    def load_model_and_alphabet(self) -> Tuple[ESM2, Alphabet]:
        """
        Load the ESM2 model and its alphabet.

        References:
            https://github.com/facebookresearch/esm/blob/main/esm/pretrained.py#L24-L28

        Returns:
            Tuple[ESM2, Alphabet]: Loaded model and alphabet.
        """
        model_location = os.path.join(self.save_model_dir, f"{self.model_name}.pt")
        if os.path.exists(model_location):
            return load_model_and_alphabet_local(model_location)
        else:
            return self.load_model_and_alphabet_hub()

    def load_model_and_alphabet_hub(self) -> Tuple[ESM2, Alphabet]:
        """
        Load the model and alphabet from the hub URL.

        References:
            https://github.com/facebookresearch/esm/blob/main/esm/pretrained.py#L62-L64

        Returns:
            Tuple[ESM2, Alphabet]: Loaded model and alphabet.
        """
        model_url = self._MODELS_URL.format(self.model_name)
        model_data = self.load_hub_workaround(model_url)
        regression_data = None
        if _has_regression_weights(self.model_name):
            regression_url = self._REGRESSION_URL.format(self.model_name)
            regression_data = self.load_hub_workaround(regression_url)
        return load_model_and_alphabet_core(
            self.model_name, model_data, regression_data
        )

    def load_hub_workaround(self, url) -> torch.Tensor:
        """
        Workaround to load models from the PyTorch Hub.

        References:
            https://github.com/facebookresearch/esm/blob/main/esm/pretrained.py#L31-L43

        Returns:
            torch.Tensor: Loaded model state dictionary.
        """
        try:
            data = torch.hub.load_state_dict_from_url(
                url, self.save_model_dir, progress=True, map_location=self.device
            )

        except RuntimeError:
            # Handle PyTorch version issues
            fn = Path(url).name
            data = torch.load(
                f"{torch.hub.get_dir()}/checkpoints/{fn}",
                map_location="cpu",
            )
        except HTTPError as e:
            raise Exception(
                f"Could not load {url}. Did you specify the correct model name?"
            )
        return data

    @staticmethod
    def name() -> str:
        """
        Returns the name of the data reader. This method identifies the specific type of data reader.

        Returns:
            str: The name of the data reader, which is "protein_token".
        """
        return "esm2_embedding"

    @property
    def token_path(self) -> None:
        """
        Not used as no token file is not created for this reader.

        Returns:
            str: Empty string since this method is not implemented.
        """
        return

    def _read_data(self, raw_data: str) -> List[int]:
        """
        Reads protein sequence data and generates embeddings.

        Args:
            raw_data (str): The protein sequence.

        Returns:
            List[int]: Embeddings generated for the sequence.
        """
        alp_tokens_idx = self._sequence_to_alphabet_tokens_idx(raw_data)
        return self._alphabet_tokens_to_esm_embedding(alp_tokens_idx).tolist()

    def _sequence_to_alphabet_tokens_idx(self, sequence: str) -> torch.Tensor:
        """
        Converts a protein sequence into ESM alphabet token indices.

        Args:
            sequence (str): Protein sequence.

        References:
            https://github.com/facebookresearch/esm/blob/2b369911bb5b4b0dda914521b9475cad1656b2ac/esm/data.py#L249-L250
            https://github.com/facebookresearch/esm/blob/2b369911bb5b4b0dda914521b9475cad1656b2ac/esm/data.py#L262-L297

        Returns:
            torch.Tensor: Tokenized sequence with special tokens (BOS/EOS) included.
        """
        seq_encoded = self._alphabet.encode(sequence)
        tokens = []

        # Add BOS token if configured
        if self._alphabet.prepend_bos:
            tokens.append(self._alphabet.cls_idx)

        # Add the main sequence
        tokens.extend(seq_encoded)

        # Add EOS token if configured
        if self._alphabet.append_eos:
            tokens.append(self._alphabet.eos_idx)

        # Convert to PyTorch tensor and return
        return torch.tensor([tokens], dtype=torch.int64)

    def _alphabet_tokens_to_esm_embedding(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Converts alphabet tokens into ESM embeddings.

        Args:
            tokens (torch.Tensor): Tokenized protein sequences.

        References:
            https://github.com/bio-ontology-research-group/deepgo2/blob/main/deepgo/extract_esm.py#L82-L107

        Returns:
            torch.Tensor: Protein embedding from the specified representation layer.
        """
        if self.device:
            tokens = tokens.to(self.device, non_blocking=True)

        with torch.no_grad():
            out = self._model(
                tokens,
                repr_layers=[
                    self.repr_layer,
                ],
                return_contacts=self.return_contacts,
            )

        # Extract representations and compute the mean embedding for each layer
        representations = {
            layer: t.to(self.device) for layer, t in out["representations"].items()
        }
        truncate_len = min(self.truncation_length, tokens.size(1))

        result = {
            "mean_representations": {
                layer: t[0, 1 : truncate_len + 1].mean(0).clone()
                for layer, t in representations.items()
            }
        }
        return result["mean_representations"][self.repr_layer]

    def on_finish(self) -> None:
        """
        Not used here as no token file exists for this reader.

        Returns:
            None
        """
        pass
