import os
from typing import Optional, Type

import torch
from rdkit import Chem

from chebai.models import ChebaiBaseNet
from chebai.preprocessing.reader import DataReader

from .._constants import MODEL_CKPT_PATH, READER_CLS_PATH, READER_KWARGS
from ._base import BaseWrapper


class NNWrapper(BaseWrapper):

    def __init__(
        self,
        **kwargs,
    ):
        self._validate_model_configs(**kwargs)
        super().__init__(**kwargs)

        self._model_ckpt_path = self._model_config[MODEL_CKPT_PATH]
        self._reader_class_path = self._model_config[READER_CLS_PATH]
        self._reader_kwargs: dict = (
            self._model_config[READER_KWARGS]
            if self._model_config[READER_KWARGS]
            else dict()
        )

        reader_cls: Type[DataReader] = self._load_class(self._reader_class_path)
        assert issubclass(reader_cls, DataReader), ""
        self._reader = reader_cls(**self._reader_kwargs)
        self._collator = reader_cls.COLLATOR()
        self._model: ChebaiBaseNet = self._load_model_()

    @classmethod
    def _validate_model_configs(
        cls, model_config: dict[str, str], model_name: str
    ) -> None:
        """
        Validates model configuration dictionary for required keys and uniqueness.

        Args:
            model_configs (Dict[str, Dict[str, Any]]): Model configuration dictionary.

        Raises:
            AttributeError: If any model config is missing required keys.
            ValueError: If duplicate paths are found for model checkpoint, class, or labels.
        """
        required_keys = {
            MODEL_CKPT_PATH,
            READER_CLS_PATH,
        }

        missing_keys = required_keys - model_config.keys()
        if missing_keys:
            raise AttributeError(
                f"Missing keys {missing_keys} in model '{model_name}' configuration."
            )

    def _load_model_(self) -> ChebaiBaseNet:
        """
        Loads a model checkpoint and its label-related properties.

        Args:
            model_name (str): Name of the model to load.

        Returns:
            Tuple[LightningModule, Dict[str, torch.Tensor]]: The model and its label properties.
        """

        if not os.path.exists(self._model_ckpt_path):
            raise FileNotFoundError(
                f"Model path '{self._model_ckpt_path}' for '{self._model_name}' does not exist."
            )

        lightning_cls = self._load_class(self._model_class_path)

        assert isinstance(lightning_cls, type), f"{lightning_cls} is not a class."
        assert issubclass(
            lightning_cls, ChebaiBaseNet
        ), f"{lightning_cls} must inherit from ChebaiBaseNet"

        try:
            model = lightning_cls.load_from_checkpoint(
                self._model_ckpt_path, input_dim=self.input_dim
            )
            model.eval()
            model.freeze()
        except Exception as e:
            raise RuntimeError(f"Error loading model {self._model_name}") from e

        return model

    def _predict_from_list_of_smiles(self, smiles_list) -> list:
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
        print(f"Predicting {len(token_dicts), token_dicts} out of {len(smiles_list)}")
        if token_dicts:
            model_output = self._forward_pass(token_dicts)
            if not isinstance(model_output, dict) and not "logits" in model_output:
                raise ValueError()
            return model_output
        else:
            raise ValueError()

    def _read_smiles(self, smiles):
        return self._reader.to_data(dict(features=smiles, labels=None))

    def _forward_pass(self, batch):
        processable_data = self._model._process_batch(
            self._collator(batch).to(self._device), 0
        )
        return self._model(processable_data, **processable_data["model_kwargs"])

    def _predict_from_data_file(
        self, processed_dir_main: str, data_file_name="data.pt"
    ) -> list:
        data = torch.load(
            os.path.join(processed_dir_main, self._reader.name(), data_file_name),
            weights_only=False,
            map_location=self._device,
        )
        return self._forward_pass(data)
