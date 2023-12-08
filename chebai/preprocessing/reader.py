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

    def __init__(self, collator_kwargs=None, token_path=None, **kwargs):
        if collator_kwargs is None:
            collator_kwargs = dict()
        self.collater = self.COLLATER(**collator_kwargs)
        self._token_path = token_path

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

    @property
    def token_path(self):
        """Get token path, create file if it does not exist yet"""
        if self._token_path is not None:
            return self._token_path
        dirname = os.path.dirname(__file__)
        token_path = os.path.join(dirname, "bin", self.name(), "tokens.txt")
        os.makedirs(os.path.join(dirname, "bin", self.name()), exist_ok=True)
        if not os.path.exists(token_path):
            with open(token_path, "x"):
                pass
        return token_path

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
        with open(self.token_path, "r") as pk:
            self.cache = [x.strip() for x in pk]

    def _get_token_index(self, token):
        """Returns a unique number for each token, automatically adds new tokens"""
        if not str(token) in self.cache:
            self.cache.append(str(token))
        return self.cache.index(str(token)) + EMBEDDING_OFFSET

    def _read_data(self, raw_data):
        return [self._get_token_index(v[1]) for v in _tokenize(raw_data)]

    def save_token_cache(self):
        """write contents of self.cache into tokens.txt"""
        with open(self.token_path, "w") as pk:
            print(f"saving {len(self.cache)} tokens to {self.token_path}...")
            print(f"first 10 tokens: {self.cache[:10]}")
            pk.writelines([f"{c}\n" for c in self.cache])


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


class SelfiesReader(ChemDataReader):
    COLLATER = RaggedCollater

    def __init__(self, *args, data_path=None, max_len=1800, vsize=4000, **kwargs):
        super().__init__(*args, **kwargs)
        self.error_count = 0
        sf.set_semantic_constraints("hypervalent")

    @classmethod
    def name(cls):
        return "selfies"

    def _read_data(self, raw_data):
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
    COLLATER = RaggedCollater

    @classmethod
    def name(cls):
        return "ord"

    def _read_data(self, raw_data):
        return [ord(s) for s in raw_data]
