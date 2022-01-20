from abc import ABC, abstractmethod
from typing import Dict, Iterable, List
import json
import os.path

from chebai import MODULE_PATH
from chebai import preprocessing as prep
from chebai.models import base, chemberta, electra, graph
from chebai.preprocessing import datasets
from chebai.result.base import ResultFactory, ResultProcessor

EXPERIMENTS = dict()


class Experiment(ABC):
    MODEL = base.JCIBaseNet

    def __init_subclass__(cls, **kwargs):
        assert cls.identifier(), "No identifier set"
        assert (
            cls.identifier() not in EXPERIMENTS
        ), f"Identifier {cls.identifier()} is not unique."
        EXPERIMENTS[cls.identifier()] = cls

    def __init__(self, batch_size, *args, **kwargs):
        self.dataset = self.build_dataset(batch_size)

    @classmethod
    def identifier(cls) -> str:
        raise NotImplementedError

    @abstractmethod
    def model_kwargs(self, *args) -> Dict:
        raise NotImplementedError

    @abstractmethod
    def build_dataset(self, batch_size) -> datasets.XYBaseDataModule:
        raise NotImplementedError

    def train(self, *args):
        self.MODEL.run(
            self.dataset,
            self.MODEL.NAME,
            model_kwargs=self.model_kwargs(*args),
        )

    def test(self, ckpt_path, *args):
        self.MODEL.test(
            self.dataset,
            self.MODEL.NAME,
            ckpt_path,
        )

    def predict(self, data_path, model_ckpt, processors: Iterable[ResultProcessor]):
        model = self.MODEL.load_from_checkpoint(model_ckpt)
        result_factory = ResultFactory(model, self.dataset, processors)
        result_factory.execute(data_path)


class ElectraPreOnSWJ(Experiment):
    MODEL = electra.ElectraPre

    @classmethod
    def identifier(cls) -> str:
        return "ElectraPre+SWJ"

    def model_kwargs(self, *args) -> Dict:
        return dict(
            lr=1e-4,
            config=dict(
                vocab_size=1400,
                max_position_embeddings=1800,
                num_attention_heads=8,
                num_hidden_layers=6,
                type_vocab_size=1,
            ),
            epochs=100,
        )

    def build_dataset(self, batch_size) -> datasets.XYBaseDataModule:
        return datasets.SWJChem(batch_size, k=100)


class ElectraPreOnJCIExt(Experiment):
    MODEL = electra.ElectraPre

    @classmethod
    def identifier(cls) -> str:
        return "ElectraPre+JCIExt"

    def model_kwargs(self, *args) -> Dict:
        return dict(
            lr=1e-4,
            config=dict(
                vocab_size=1400,
                max_position_embeddings=1800,
                num_attention_heads=8,
                num_hidden_layers=6,
                type_vocab_size=1,
            ),
            epochs=100,
        )

    def build_dataset(self, batch_size) -> datasets.XYBaseDataModule:
        return datasets.JCIExtendedTokenData(batch_size)


class ElectraPreOnJCI(Experiment):
    MODEL = electra.ElectraPre

    @classmethod
    def identifier(cls) -> str:
        return "ElectraPre+JCI"

    def model_kwargs(self, *args) -> Dict:
        return dict(
            lr=1e-4,
            config=dict(
                vocab_size=1400,
                max_position_embeddings=1800,
                num_attention_heads=8,
                num_hidden_layers=6,
                type_vocab_size=1,
            ),
            epochs=100,
        )

    def build_dataset(self, batch_size) -> datasets.XYBaseDataModule:
        return datasets.JCITokenData(batch_size)


class ElectraPreBPEOnSWJ(Experiment):
    MODEL = electra.ElectraPre

    @classmethod
    def identifier(cls) -> str:
        return "ElectraBPEPre+SWJ"

    def model_kwargs(self, *args) -> Dict:
        return dict(
            lr=1e-4,
            config=dict(
                vocab_size=4000,
                max_position_embeddings=1800,
                num_attention_heads=8,
                num_hidden_layers=6,
                type_vocab_size=1,
            ),
            epochs=100,
        )

    def build_dataset(self, batch_size) -> datasets.XYBaseDataModule:
        return datasets.SWJBPE(
            batch_size,
            reader_kwargs=dict(
                data_path=os.path.join(MODULE_PATH, "preprocessing/bin/BPE_SWJ")
            ),
            k=100,
        )


class ElectraBPEOnJCIExt(Experiment):
    MODEL = electra.Electra

    @classmethod
    def identifier(cls) -> str:
        return "Electra+JCIExtBPE"

    def model_kwargs(self, *args) -> Dict:
        return dict(
            lr=1e-4,
            config=dict(
                vocab_size=4000,
                max_position_embeddings=1800,
                num_attention_heads=8,
                num_hidden_layers=6,
                type_vocab_size=1,
            ),
            epochs=100,
        )

    def build_dataset(self, batch_size) -> datasets.XYBaseDataModule:
        return datasets.JCIExtendedBPEData(
            batch_size,
            reader_kwargs=dict(
                data_path=os.path.join(MODULE_PATH, "preprocessing/bin/BPE_SWJ")
            ),
        )


class ChembertaPreOnSWJ(Experiment):
    MODEL = chemberta.ChembertaPre

    @classmethod
    def identifier(cls) -> str:
        return "ChembertaPre+SWJ"

    def model_kwargs(self, *args) -> Dict:
        return dict(
            lr=1e-4,
            config=dict(
                vocab_size=1400,
                max_position_embeddings=1800,
                num_attention_heads=8,
                num_hidden_layers=6,
                type_vocab_size=1,
            ),
            epochs=100,
        )

    def build_dataset(self, batch_size) -> datasets.XYBaseDataModule:
        return datasets.SWJChem(batch_size, k=100)


class ChembertaPreBPEOnSWJ(Experiment):
    MODEL = chemberta.ChembertaPre

    @classmethod
    def identifier(cls) -> str:
        return "ChembertaPreBPE+SWJ"

    def model_kwargs(self, *args) -> Dict:
        return dict(
            lr=1e-4,
            tokenizer_path=os.path.join(MODULE_PATH, "preprocessing/bin/BPE_SWJ"),
            config=dict(
                vocab_size=4000,
                max_position_embeddings=1800,
                num_attention_heads=8,
                num_hidden_layers=6,
                type_vocab_size=1,
            ),
            epochs=100,
        )

    def build_dataset(self, batch_size) -> datasets.XYBaseDataModule:
        return datasets.SWJBPE(
            batch_size,
            reader_kwargs=dict(
                data_path=os.path.join(MODULE_PATH, "preprocessing/bin/BPE_SWJ")
            ),
            k=100,
        )


class ElectraSWJ(Experiment):
    MODEL = electra.Electra

    @classmethod
    def identifier(cls) -> str:
        return "Electra+SWJ"

    def model_kwargs(self, *args) -> Dict:
        return dict(
            lr=1e-4,
            config=dict(
                vocab_size=1400,
                max_position_embeddings=1800,
                num_attention_heads=8,
                num_hidden_layers=6,
                type_vocab_size=1,
            ),
            epochs=100,
        )

    def build_dataset(self, batch_size) -> datasets.XYBaseDataModule:
        return datasets.SWJChem(batch_size, k=100)

    def train(self, batch_size, *args):
        raise Exception("This expermient is prediction only")


class ElectraLegSWJ(ElectraSWJ):
    MODEL = electra.ElectraLegacy

    @classmethod
    def identifier(cls) -> str:
        return "ElectraLeg+SWJ"


class ElectraOnJCI(Experiment):
    MODEL = electra.Electra

    @classmethod
    def identifier(cls) -> str:
        return "Electra+JCI"

    def model_kwargs(self, *args) -> Dict:
        checkpoint_path = args[0]
        return dict(
            lr=1e-4,
            pretrained_checkpoint=checkpoint_path,
            out_dim=self.dataset.label_number,
            config=dict(
                vocab_size=1400,
                max_position_embeddings=1800,
                num_attention_heads=8,
                num_hidden_layers=6,
                type_vocab_size=1,
            ),
            epochs=100,
        )

    def build_dataset(self, batch_size) -> datasets.XYBaseDataModule:
        return datasets.JCITokenData(batch_size)


class ElectraOnChEBI100(Experiment):
    MODEL = electra.Electra

    @classmethod
    def identifier(cls) -> str:
        return "Electra+ChEBI100"

    def model_kwargs(self, *args) -> Dict:
        checkpoint_path = args[0]
        return dict(
            lr=1e-4,
            pretrained_checkpoint=checkpoint_path,
            out_dim=self.dataset.label_number,
            config=dict(
                vocab_size=1400,
                max_position_embeddings=1800,
                num_attention_heads=8,
                num_hidden_layers=6,
                type_vocab_size=1,
            ),
            epochs=100,
        )

    def build_dataset(self, batch_size) -> datasets.XYBaseDataModule:
        return datasets.ChEBIOver100(batch_size)


class ElectraOnJCIExt(ElectraOnJCI):
    MODEL = electra.Electra

    @classmethod
    def identifier(cls) -> str:
        return "Electra+JCIExt"

    def build_dataset(self, batch_size) -> datasets.XYBaseDataModule:
        return datasets.JCIExtendedTokenData(batch_size)


class GATOnSWJ(Experiment):
    MODEL = graph.JCIGraphAttentionNet

    @classmethod
    def identifier(cls) -> str:
        return "GAT+JCIExt"

    def model_kwargs(self, *args) -> Dict:
        return dict(
            lr=1e-4,
            in_length=50,
            hidden_length=100,
            epochs=100,
        )

    def build_dataset(self, batch_size) -> datasets.XYBaseDataModule:
        return datasets.JCIGraphData(batch_size)
