from abc import ABC, abstractmethod
from typing import Dict, List

from chem import preprocessing as prep
from chem.models import base, electra
from chem.preprocessing import datasets

EXPERIMENTS = dict()


class Experiment(ABC):
    MODEL = base.JCIBaseNet

    def __init__(self, batch_size, *args):
        self.batch_size = batch_size

    def __init_subclass__(cls, **kwargs):
        assert cls.identifier, "No identifier set"
        assert (
            cls.identifier not in EXPERIMENTS
        ), f"Identifier {cls.identifier} is not unique."
        EXPERIMENTS[cls.identifier] = cls

    @classmethod
    @property
    def identifier(cls) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def model_kwargs(self) -> Dict:
        raise NotImplementedError

    @property
    @abstractmethod
    def datasets(self) -> List[datasets.XYBaseDataModule]:
        raise NotImplementedError

    def execute(self):
        for dataset in self.datasets:
            self.MODEL.run(
                dataset,
                self.MODEL.NAME,
                model_kwargs=self.model_kwargs,
            )


class ElectraPreOnSWJ(Experiment):
    MODEL = electra.ElectraPre

    @classmethod
    @property
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
            epochs=2,
        )

    @property
    def datasets(self) -> List[datasets.XYBaseDataModule]:
        return [datasets.SWJUnlabeledChemToken(self.batch_size, k=100)]


class ElectraOnJCI(Experiment):
    MODEL = electra.Electra

    @classmethod
    @property
    def identifier(cls) -> str:
        return "Electra+JCI"

    def __init__(self, batch_size, checkpoint_path, *args):
        super().__init__(batch_size, *args)
        self.checkpoint_path = checkpoint_path

    @property
    def model_kwargs(self) -> Dict:
        return dict(
            lr=1e-4,
            pretrained_checkpoint=self.checkpoint_path,
            config=dict(
                vocab_size=1400,
                max_position_embeddings=1800,
                num_attention_heads=8,
                num_hidden_layers=6,
                type_vocab_size=1,
            ),
            epochs=20,
        )

    @property
    def datasets(self) -> List[datasets.XYBaseDataModule]:
        return [datasets.JCITokenData(self.batch_size)]
