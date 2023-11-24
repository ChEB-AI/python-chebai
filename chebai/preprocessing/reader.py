import os

from pysmiles.read_smiles import _tokenize
from transformers import RobertaTokenizerFast
import selfies as sf
import deepsmiles

from chebai.preprocessing.collate import (
    DefaultCollater,
    RaggedCollater,
)

EMBEDDING_OFFSET = 10
PADDING_TOKEN_INDEX = 0
MASK_TOKEN_INDEX = 1
CLS_TOKEN = 2


class DataReader:
    COLLATER = DefaultCollater

    def __init__(self, collator_kwargs=None, **kwargs):
        if collator_kwargs is None:
            collator_kwargs = dict()
        self.collater = self.COLLATER(**collator_kwargs)

    def _get_raw_data(self, row):
        return row["features"]

    def _get_raw_label(self, row):
        return row["labels"]

    def _get_raw_id(self, row):
        return row.get("ident", row["features"])

    def _get_raw_group(self, row):
        return row.get("group", None)

    def _get_additional_kwargs(self, row):
        return row.get("additional_kwargs", dict())

    def name(cls):
        raise NotImplementedError

    def _read_id(self, raw_data):
        return raw_data

    def _read_data(self, raw_data):
        return raw_data

    def _read_label(self, raw_label):
        return raw_label

    def _read_group(self, raw):
        return raw

    def _read_components(self, row):
        return dict(
            features=self._get_raw_data(row),
            labels=self._get_raw_label(row),
            ident=self._get_raw_id(row),
            group=self._get_raw_group(row),
            additional_kwargs=self._get_additional_kwargs(row),
        )

    def to_data(self, row):
        d = self._read_components(row)
        return dict(
            features=self._read_data(d["features"]),
            labels=self._read_label(d["labels"]),
            ident=self._read_id(d["ident"]),
            group=self._read_group(d["group"]),
            **d["additional_kwargs"],
        )

    def save_token_cache(self):
        return


class ChemDataReader(DataReader):
    COLLATER = RaggedCollater

    @classmethod
    def name(cls):
        return "smiles_token"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dirname = os.path.dirname(__file__)
        with open(os.path.join(dirname, "bin", "tokens.txt"), "r") as pk:
            self.cache = [x.strip() for x in pk]

    def _get_token_index(self, token):
        """Returns a unique number for each token, automatically adds new tokens"""
        if not str(token) in self.cache:
            self.cache.append(str(token))
        return self.cache.index(str(token)) + EMBEDDING_OFFSET

    def _read_data(self, raw_data):
        return [
            self._get_token_index(v[1]) for v in _tokenize(raw_data)
        ]

    def save_token_cache(self):
        """write contents of self.cache into tokens.txt"""
        dirname = os.path.dirname(__file__)
        with open(os.path.join(dirname, "bin", "tokens.txt"), "w") as pk:
            print(f'saving tokens to {os.path.join(dirname, "bin", "tokens.txt")}...')
            print(f'first 10 tokens: {self.cache[:10]}')
            pk.writelines([f'{c}\n' for c in self.cache])


class DeepChemDataReader(ChemDataReader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.converter = deepsmiles.Converter(rings=True, branches=True)
        self.error_count = 0

    @classmethod
    def name(cls):
        return "deepsmiles_token"

    def _read_data(self, raw_data):
        try:
            tokenized = _tokenize(self.converter.encode(raw_data))
        except ValueError as e:
            print(f'could not process {raw_data}')
            print(f'\t{e}')
            self.error_count += 1
            print(f'\terror count: {self.error_count}')
            tokenized = []
        return [
            self._get_token_index(v[1]) for v in tokenized
        ]


class ChemDataUnlabeledReader(ChemDataReader):
    COLLATER = RaggedCollater

    @classmethod
    def name(cls):
        return "smiles_token_unlabeled"

    def _get_raw_label(self, row):
        return None


class ChemBPEReader(DataReader):
    COLLATER = RaggedCollater

    @classmethod
    def name(cls):
        return "smiles_bpe"

    def __init__(self, *args, data_path=None, max_len=1800, vsize=4000, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = RobertaTokenizerFast.from_pretrained(
            data_path, max_len=max_len
        )

    def _get_raw_data(self, row):
        return self.tokenizer(row["features"])["input_ids"]


class SelfiesReader(DataReader):
    COLLATER = RaggedCollater

    def __init__(self, *args, data_path=None, max_len=1800, vsize=4000, **kwargs):
        super().__init__(*args, **kwargs)
        with open("chebai/preprocessing/bin/selfies.txt", "rt") as pk:
            self.cache = [l.strip() for l in pk]

    @classmethod
    def name(cls):
        return "selfies"

    def _get_raw_data(self, row):
        try:
            splits = sf.split_selfies(sf.encoder(row["features"].strip(), strict=True))
        except Exception as e:
            print(e)
            return
        else:
            return [self.cache.index(x) + EMBEDDING_OFFSET for x in splits]


class OrdReader(DataReader):
    COLLATER = RaggedCollater

    @classmethod
    def name(cls):
        return "ord"

    def _read_data(self, raw_data):
        return [ord(s) for s in raw_data]
