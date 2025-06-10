import os
from pathlib import Path

import torch
from rdkit import Chem

from chebai.models import ChebaiBaseNet
from chebai.preprocessing.datasets.base import XYBaseDataModule
from chebai.preprocessing.structures import XYData

from .._constants import (
    DATA_CLS_KWARGS,
    DATA_CLS_PATH,
    MODEL_CKPT_PATH,
    MODEL_CLS_PATH,
    MODEL_LD_KWARGS,
)
from .._utils import load_class
from ._base import BaseWrapper


class NNWrapper(BaseWrapper):

    def __init__(self, **kwargs):
        self._validate_model_configs(
            model_config=kwargs["model_config"], model_name=kwargs["model_name"]
        )
        super().__init__(**kwargs)

        self._model_class_path = self._model_config[MODEL_CLS_PATH]
        self._model_ckpt_path = self._model_config[MODEL_CKPT_PATH]
        self._model_ld_kwargs: dict = self._model_config.get(MODEL_LD_KWARGS, {})

        self._data_cls_instance: XYBaseDataModule = self._load_data_instance()
        self.collated_labels = None
        self._model: ChebaiBaseNet = self._load_model_()

    @classmethod
    def _validate_model_configs(
        cls,
        model_config: dict[str, str],
        model_name: str,
    ) -> None:
        """
        Validates model configuration dictionary for required keys and uniqueness.

        Args:
            model_configs (Dict[str, Dict[str, Any]]): Model configuration dictionary.

        Raises:
            AttributeError: If any model config is missing required keys.
            ValueError: If duplicate paths are found for model checkpoint, class, or labels.
        """
        required_keys = {MODEL_CKPT_PATH, DATA_CLS_PATH, MODEL_CLS_PATH}

        missing_keys = required_keys - model_config.keys()
        if missing_keys:
            raise AttributeError(
                f"Missing keys {missing_keys} in model '{model_name}' configuration."
            )

    def _load_data_instance(self):
        data_cls = load_class(self._model_config[DATA_CLS_PATH])
        assert isinstance(data_cls, type), f"{data_cls} is not a class."
        assert issubclass(
            data_cls, XYBaseDataModule
        ), f"{data_cls} must inherit from XYBaseDataModule"
        data_kwargs = self._model_config.get(DATA_CLS_KWARGS, {})
        return data_cls(**data_kwargs)

    def _load_model_(self) -> ChebaiBaseNet:
        """
        Loads a model checkpoint and its label-related properties.

        Args:
            input_dim (int): Name of the model to load.

        Returns:
            Tuple[LightningModule, Dict[str, torch.Tensor]]: The model and its label properties.
        """

        if not os.path.exists(self._model_ckpt_path):
            raise FileNotFoundError(
                f"Model path '{self._model_ckpt_path}' for '{self._model_name}' does not exist."
            )

        lightning_cls = load_class(self._model_class_path)

        assert isinstance(lightning_cls, type), f"{lightning_cls} is not a class."
        assert issubclass(
            lightning_cls, ChebaiBaseNet
        ), f"{lightning_cls} must inherit from ChebaiBaseNet"
        try:
            model = lightning_cls.load_from_checkpoint(
                self._model_ckpt_path, input_dim=5, **self._model_ld_kwargs
            )
        except Exception as e:
            raise RuntimeError(
                f"Error loading model {self._model_name} \n Error: {e}"
            ) from e

        assert isinstance(
            model, ChebaiBaseNet
        ), f"{model} is not a ChebaiBaseNet instance."
        model.eval()
        model.freeze()
        return model

    def _predict_from_list_of_smiles(self, smiles_list: list[str]) -> list:
        token_dicts = []
        could_not_parse = []
        index_map = dict()
        for i, smiles in enumerate(smiles_list):
            try:
                # Try to parse the smiles string
                if not smiles:
                    raise ValueError()
                d = self._read_smiles(smiles)
                # This is just for sanity checks
                rdmol = Chem.MolFromSmiles(smiles, sanitize=False)
            except Exception as e:
                # Note if it fails
                could_not_parse.append(i)
                print(f"Failing to parse {smiles} due to {e}")
            else:
                if rdmol is None:
                    could_not_parse.append(i)
                else:
                    index_map[i] = len(token_dicts)
                    token_dicts.append(d)
        if token_dicts:
            model_output = self._forward_pass(token_dicts)
            if not isinstance(model_output, dict) and not "logits" in model_output:
                raise ValueError()
            return model_output
        else:
            raise ValueError()

    def _read_smiles(self, smiles: str):
        return self._data_cls_instance.reader.to_data(
            dict(features=smiles, labels=None)
        )

    def _forward_pass(self, batch):
        collated_batch: XYData = self._data_cls_instance.reader.collator(batch).to(
            self._device
        )
        self.collated_labels = collated_batch.y
        processable_data = self._model._process_batch(  # pylint: disable=W0212
            collated_batch, 0
        )
        return self._model(processable_data, **processable_data["model_kwargs"])

    def _evaluate_from_data_file(
        self, data_processed_dir_main: Path, data_file_name="data.pt"
    ) -> list:
        data = torch.load(
            data_processed_dir_main
            / self._data_cls_instance.reader.name()
            / data_file_name,
            weights_only=False,
            map_location=self._device,
        )
        return self._forward_pass(data)
