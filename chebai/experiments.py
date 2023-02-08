from abc import ABC, abstractmethod
from typing import Dict, Iterable, List
import json
import os.path

from chebai import MODULE_PATH
import torch.nn

from chebai import preprocessing as prep
from chebai.models import base, chemberta, electra, graph
from chebai.preprocessing import datasets
from chebai.result.base import ResultFactory, ResultProcessor

EXPERIMENTS = dict()


class Experiment(ABC):
    MODEL = base.JCIBaseNet
    LOSS = torch.nn.BCEWithLogitsLoss

    def __init_subclass__(cls, **kwargs):
        assert (
            cls.identifier() not in EXPERIMENTS
        ), f"Identifier {cls.identifier()} is not unique."
        if cls.identifier() is not None:
            EXPERIMENTS[cls.identifier()] = cls

    def __init__(self, batch_size, *args, version=None, **kwargs):
        self.dataset = self.build_dataset(batch_size)
        self.version=version

    @classmethod
    def identifier(cls) -> str:
        raise NotImplementedError

    @abstractmethod
    def model_kwargs(self, *args) -> Dict:
        raise NotImplementedError

    @abstractmethod
    def build_dataset(self, batch_size) -> datasets.XYBaseDataModule:
        raise NotImplementedError

    def train(self, batch_size, epochs, *args, **kwargs):
        self.MODEL.run(
            self.dataset,
            self.MODEL.NAME,
            epochs,
            loss=self.LOSS,
            model_kwargs=self.model_kwargs(*args),
            version=self.version,
            **kwargs
        )

    def test(self, ckpt_path, *args):
        self.MODEL.test(
            self.dataset,
            self.MODEL.NAME,
            ckpt_path,
        )

    def predict(
        self, data_path, model_ckpt, processors: Iterable[ResultProcessor], **kwargs
    ):
        model = self.MODEL.load_from_checkpoint(model_ckpt)
        result_factory = ResultFactory(model, self.dataset, processors)
        result_factory.execute(data_path, **kwargs)


class ElectraPreOnSWJ(Experiment):
    MODEL = electra.ElectraPre
    LOSS = electra.ElectraPreLoss

    @classmethod
    def identifier(cls) -> str:
        return "ElectraPre+SWJ"

    def model_kwargs(self, *args) -> Dict:
        return dict(
            optimizer_kwargs=dict(lr=1e-4),
            config=dict(
                generator=dict(
                    vocab_size=1400,
                    max_position_embeddings=1800,
                    num_attention_heads=8,
                    num_hidden_layers=6,
                    type_vocab_size=1,
                ),
                discriminator=dict(
                    vocab_size=1400,
                    max_position_embeddings=1800,
                    num_attention_heads=8,
                    num_hidden_layers=6,
                    type_vocab_size=1,
                ),
            ),
        )

    def build_dataset(self, batch_size) -> datasets.XYBaseDataModule:
        return datasets.SWJChem(batch_size, k=100)


class ChembertaPreOnSWJ(Experiment):
    MODEL = chemberta.ChembertaPre

    @classmethod
    def identifier(cls) -> str:
        return "ChembertaPre+SWJ"

    def model_kwargs(self, *args) -> Dict:
        return dict(
            optimizer_kwargs=dict(lr=1e-4),
            config=dict(
                vocab_size=1400,
                max_position_embeddings=1800,
                num_attention_heads=8,
                num_hidden_layers=6,
                type_vocab_size=1,
            ),
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
            optimizer_kwargs=dict(lr=1e-4),
            tokenizer_path=os.path.join(MODULE_PATH, "preprocessing/bin/BPE_SWJ"),
            config=dict(
                vocab_size=4000,
                max_position_embeddings=1800,
                num_attention_heads=8,
                num_hidden_layers=6,
                type_vocab_size=1,
            ),
        )

    def build_dataset(self, batch_size) -> datasets.XYBaseDataModule:
        return datasets.SWJBPE(
            batch_size,
            reader_kwargs=dict(
                data_path=os.path.join(MODULE_PATH, "preprocessing/bin/BPE_SWJ")
            ),
            k=100,
        )


class _ElectraExperiment(Experiment):
    MODEL = electra.Electra

    @classmethod
    def identifier(cls) -> str:
        return None

    def model_kwargs(self, *args) -> Dict:
        checkpoint_path = args[0] if len(args) > 0 else None
        return dict(
            optimizer_kwargs=dict(lr=1e-4),
            pretrained_checkpoint=checkpoint_path,
            out_dim=self.dataset.label_number,
            config=dict(
                vocab_size=1400,
                max_position_embeddings=1800,
                num_attention_heads=8,
                num_hidden_layers=6,
                type_vocab_size=1,
            )
        )


class ElectraOnJCI(_ElectraExperiment):
    @classmethod
    def identifier(cls) -> str:
        return "Electra+JCI"

    def build_dataset(self, batch_size) -> datasets.XYBaseDataModule:
        return datasets.JCITokenData(batch_size)


class ElectraOnChEBI100(_ElectraExperiment):
    MODEL = electra.Electra

    @classmethod
    def identifier(cls) -> str:
        return "Electra+ChEBI100"

    def build_dataset(self, batch_size) -> datasets.XYBaseDataModule:
        return datasets.ChEBIOver100(batch_size)


class ElectraOnJCIExt(ElectraOnJCI):
    MODEL = electra.Electra

    @classmethod
    def identifier(cls) -> str:
        return "Electra+JCIExt"

    def build_dataset(self, batch_size) -> datasets.XYBaseDataModule:
        return datasets.JCIExtendedTokenData(batch_size)


class ElectraPreOnSWJSelfies(ElectraPreOnSWJ):
    MODEL = electra.ElectraPre

    @classmethod
    def identifier(cls) -> str:
        return "ElectraPre+SWJSelfies"

    def build_dataset(self, batch_size) -> datasets.XYBaseDataModule:
        return datasets.SWJSelfies(batch_size)


class ElectraOnJCIExtSelfies(ElectraOnJCIExt):
    MODEL = electra.Electra

    @classmethod
    def identifier(cls) -> str:
        return "Electra+JCIExtSelfies"

    def build_dataset(self, batch_size) -> datasets.XYBaseDataModule:
        return datasets.JCIExtSelfies(batch_size)


class ElectraLegJCI(ElectraOnJCI):
    MODEL = electra.ElectraLegacy

    @classmethod
    def identifier(cls) -> str:
        return "ElectraLeg+JCI"


class ElectraLegJCIExt(ElectraOnJCIExt):
    MODEL = electra.ElectraLegacy

    @classmethod
    def identifier(cls) -> str:
        return "ElectraLeg+JCIExt"


class ElectraOnTox21(_ElectraExperiment):
    MODEL = electra.Electra
    LOSS = torch.nn.BCEWithLogitsLoss

    @classmethod
    def identifier(cls) -> str:
        return "Electra+Tox21"

    def build_dataset(self, batch_size) -> datasets.XYBaseDataModule:
        return datasets.Tox21Chem(batch_size)

    def model_kwargs(self, *args) -> Dict:
        d = super().model_kwargs(*args)
        d["config"]["hidden_dropout_prob"] = 0.4
        d["config"]["word_dropout"] = 0.2
        d["optimizer_kwargs"]["weight_decay"] = 1e-4
        return d


class ElectraOnTox21MoleculeNet(_ElectraExperiment):
    MODEL = electra.Electra
    LOSS = torch.nn.BCEWithLogitsLoss

    @classmethod
    def identifier(cls) -> str:
        return "Electra+Tox21MN"

    def build_dataset(self, batch_size) -> datasets.XYBaseDataModule:
        return datasets.Tox21MolNetChem(batch_size)

    def model_kwargs(self, *args) -> Dict:
        d = super().model_kwargs(*args)
        d["config"]["hidden_dropout_prob"] = 0.4
        d["config"]["word_dropout"] = 0.2
        d["optimizer_kwargs"]["weight_decay"] = 1e-4
        return d


class ElectraConeOnTox21MoleculeNet(ElectraOnChEBI100):
    MODEL = electra.ConeElectra
    LOSS = electra.ConeLoss

    @classmethod
    def identifier(cls) -> str:
        return "ElectraCone+Chebi100"

class ElectraOnTox21Challenge(_ElectraExperiment):
    @classmethod
    def identifier(cls) -> str:
        return "Electra+Tox21Chal"

    def model_kwargs(self, *args) -> Dict:
        d = super().model_kwargs(*args)
        d["config"]["hidden_dropout_prob"] = 0.5
        d["config"]["word_dropout"] = 0.3
        d["optimizer_kwargs"]["weight_decay"] = 1e-4
        return d

    def build_dataset(self, batch_size) -> datasets.XYBaseDataModule:
        return datasets.Tox21ChallengeChem(batch_size)


class ElectraBPEOnJCIExt(_ElectraExperiment):
    @classmethod
    def identifier(cls) -> str:
        return "Electra+JCIExtBPE"

    def build_dataset(self, batch_size) -> datasets.XYBaseDataModule:
        return datasets.JCIExtendedBPEData(
            batch_size,
            reader_kwargs=dict(
                data_path=os.path.join(MODULE_PATH, "preprocessing/bin/BPE_SWJ")
            ),
        )


class GATOnSWJ(Experiment):
    MODEL = graph.JCIGraphAttentionNet

    @classmethod
    def identifier(cls) -> str:
        return "GAT+JCIExt"

    def model_kwargs(self, *args) -> Dict:
        return dict(
            in_length=50,
            hidden_length=100,
        )

    def build_dataset(self, batch_size) -> datasets.XYBaseDataModule:
        return datasets.JCIGraphData(batch_size)


class GATOnTox21(Experiment):
    MODEL = graph.JCIGraphAttentionNet

    @classmethod
    def identifier(cls) -> str:
        return "GAT+Tox21"

    def model_kwargs(self, *args) -> Dict:
        return dict(
            in_length=50,
            hidden_length=100,
        )

    def build_dataset(self, batch_size) -> datasets.XYBaseDataModule:
        return datasets.Tox21Graph(batch_size)
