from abc import ABC, abstractmethod
from typing import Dict, List
import json

from chebai import preprocessing as prep
from chebai.models import base, electra, graph
from chebai.preprocessing import datasets

EXPERIMENTS = dict()


class Experiment(ABC):
    MODEL = base.JCIBaseNet

    def __init_subclass__(cls, **kwargs):
        assert cls.identifier(), "No identifier set"
        assert (
            cls.identifier() not in EXPERIMENTS
        ), f"Identifier {cls.identifier()} is not unique."
        EXPERIMENTS[cls.identifier()] = cls

    @classmethod
    def identifier(cls) -> str:
        raise NotImplementedError

    @abstractmethod
    def model_kwargs(self, **kwargs) -> Dict:
        raise NotImplementedError

    @abstractmethod
    def datasets(self, batch_size) -> List[datasets.XYBaseDataModule]:
        raise NotImplementedError

    def train(self, batch_size, **kwargs):
        for dataset in self.datasets(batch_size):
            self.MODEL.run(
                dataset,
                self.MODEL.NAME,
                model_kwargs=self.model_kwargs(**kwargs),
            )

    def predict(self, ckpt_path, data_path):
        for dataset in self.datasets(1):
            with open(f"/tmp/{'_'.join(dataset.full_identifier)}.json", "w") as fout:
                json.dump(
                    [
                        dict(smiles=smiles, labels=label, prediction=pred)
                        for smiles, label, pred in self.MODEL.pred(
                            dataset, ckpt_path, data_path
                        )
                    ],
                    fout,
                    indent=2,
                )


class ElectraPreOnSWJ(Experiment):
    MODEL = electra.ElectraPre

    @classmethod
    def identifier(cls) -> str:
        return "ElectraPre+SWJ"

    @property
    def model_kwargs(self) -> Dict:
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

    def datasets(self, batch_size) -> List[datasets.XYBaseDataModule]:
        return [datasets.SWJUnlabeledChemToken(batch_size, k=100)]


class ElectraOnJCI(Experiment):
    MODEL = electra.Electra

    @classmethod
    def identifier(cls) -> str:
        return "Electra+JCI"

    @property
    def model_kwargs(self, checkpoint_path=None, **kwargs) -> Dict:
        return dict(
            lr=1e-4,
            pretrained_checkpoint=checkpoint_path,
            config=dict(
                vocab_size=1400,
                max_position_embeddings=1800,
                num_attention_heads=8,
                num_hidden_layers=6,
                type_vocab_size=1,
            ),
            epochs=20,
        )

    def datasets(self, batch_size) -> List[datasets.XYBaseDataModule]:
        return [datasets.JCITokenData(batch_size)]


class ElectraOnJCIExt(ElectraOnJCI):
    MODEL = electra.Electra

    @classmethod
    def identifier(cls) -> str:
        return "Electra+JCIExt"

    def datasets(self, batch_size) -> List[datasets.XYBaseDataModule]:
        return [datasets.JCIExtendedTokenData(batch_size)]


class GATOnSWJ(Experiment):
    MODEL = graph.JCIGraphAttentionNet

    @classmethod
    def identifier(cls) -> str:
        return "GAT+JCIExt"

    @property
    def model_kwargs(self) -> Dict:
        return dict(
            lr=1e-4,
            in_length=50,
            hidden_length=100,
            epochs=20,
        )

    def datasets(self, batch_size) -> List[datasets.XYBaseDataModule]:
        return [datasets.JCIGraphData(batch_size)]
