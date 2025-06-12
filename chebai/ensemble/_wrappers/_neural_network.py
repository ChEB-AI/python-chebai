from pathlib import Path

from rdkit import Chem

from chebai.models import ChebaiBaseNet
from chebai.preprocessing.datasets.base import XYBaseDataModule
from chebai.preprocessing.structures import XYData

from .._constants import DATA_CONFIG_PATH, MODEL_CKPT_PATH, MODEL_CONFIG_PATH
from .._utils import load_data_instance, load_model_for_inference, parse_config_file
from ._base import BaseWrapper


class NNWrapper(BaseWrapper):

    def __init__(self, **kwargs):
        self._validate_model_configs(
            model_config=kwargs["model_config"], model_name=kwargs["model_name"]
        )
        super().__init__(**kwargs)

        self._model_ckpt_path = self._model_config[MODEL_CKPT_PATH]
        self._model_config_path = self._model_config[MODEL_CONFIG_PATH]
        self._data_config_path = self._model_config[DATA_CONFIG_PATH]

        data_cls_path, data_kwargs = parse_config_file(self._data_config_path)
        self._data_cls_instance: XYBaseDataModule = load_data_instance(
            data_cls_path, data_kwargs
        )

        model_cls_path, model_kwargs = parse_config_file(self._model_config_path)
        self._model: ChebaiBaseNet = load_model_for_inference(
            self._model_ckpt_path,
            model_cls_path,
            model_kwargs,
            model_name=kwargs["model_name"],
        )

        self.collated_labels = None

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
        required_keys = {MODEL_CKPT_PATH, MODEL_CONFIG_PATH, DATA_CONFIG_PATH}

        missing_keys = required_keys - model_config.keys()
        if missing_keys:
            raise AttributeError(
                f"Missing keys {missing_keys} in model '{model_name}' configuration."
            )

    def _pre_load_hook(self) -> None: ...

    def _predict_from_list_of_smiles(self, smiles_list: list[str]) -> list:
        token_dicts = []
        could_not_parse = []
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
                    token_dicts.append(d)
        if token_dicts:
            model_output = self._forward_pass(token_dicts)
            if not isinstance(model_output, dict) and not "logits" in model_output:
                raise ValueError()
            return model_output
        else:
            raise ValueError()

    def _read_smiles(self, smiles: str) -> dict:
        return self._data_cls_instance.reader.to_data(
            dict(features=smiles, labels=None)
        )

    def _forward_pass(self, batch: list[dict]) -> dict:
        collated_batch: XYData = self._data_cls_instance.reader.collator(batch).to(
            self._device
        )
        self.collated_labels = collated_batch.y
        processable_data = self._model._process_batch(  # pylint: disable=W0212
            collated_batch, 0
        )
        return self._model(processable_data, **processable_data["model_kwargs"])

    def _evaluate_from_data_file(self, **kwargs) -> list:
        filename = self._data_cls_instance.processed_file_names_dict["data"]
        data_list_of_dict = self._data_cls_instance.load_processed_data_from_file(
            Path(self._data_cls_instance.processed_dir) / filename
        )
        return self._forward_pass(data_list_of_dict)
