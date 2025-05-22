import importlib
import json
import os
from typing import Optional, Type

import torch
from lightning import LightningModule
from rdkit import Chem

from chebai.models import ChebaiBaseNet
from chebai.preprocessing.reader import DataReader

from ._base import BaseWrapper


class NNWrapper(BaseWrapper):

    def __init__(
        self,
        model_config: dict,
        reader_cls: Type[DataReader],
        reader_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._model_class_path = model_config[MODEL_CLS_PATH]
        self.model: ChebaiBaseNet = model
        if reader_kwargs is None:
            reader_kwargs = dict()
        self.reader = reader_cls(**reader_kwargs)
        self.collator = reader_cls.COLLATOR()

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
        return self.reader.to_data(dict(features=smiles, labels=None))

    def _forward_pass(self, batch):
        processable_data = self.model._process_batch(
            self.collator(batch).to(self._device), 0
        )
        return self.model(processable_data, **processable_data["model_kwargs"])

    def _predict_from_data_file(
        self, processed_dir_main: str, data_file_name="data.pt"
    ) -> list:
        data = torch.load(
            os.path.join(processed_dir_main, self.reader.name(), data_file_name),
            weights_only=False,
            map_location=self._device,
        )
        return self._forward_pass(data)

    def _load_model_and_its_props(
        self, model_name: str
    ) -> tuple[LightningModule, dict[str, torch.Tensor]]:
        """
        Loads a model checkpoint and its label-related properties.

        Args:
            model_name (str): Name of the model to load.

        Returns:
            Tuple[LightningModule, Dict[str, torch.Tensor]]: The model and its label properties.
        """
        config = self.model_configs[model_name]
        model_ckpt_path = config["ckpt_path"]
        model_class_path = config["class_path"]
        model_labels_path = config["labels_path"]

        if not os.path.exists(model_ckpt_path):
            raise FileNotFoundError(
                f"Model path '{model_ckpt_path}' for '{model_name}' does not exist."
            )

        assert isinstance(lightning_cls, type), f"{class_name} is not a class."
        assert issubclass(
            lightning_cls, ChebaiBaseNet
        ), f"{class_name} must inherit from ChebaiBaseNet"

        try:
            model = lightning_cls.load_from_checkpoint(
                model_ckpt_path, input_dim=self.input_dim
            )
            model.eval()
            model.freeze()
            model_label_props = self._generate_model_label_props(model_labels_path)
        except Exception as e:
            raise RuntimeError(f"Error loading model {model_name}") from e

        return model, model_label_props

    def _generate_model_label_props(self, labels_path: str) -> dict[str, torch.Tensor]:
        """
        Generates label mask and confidence tensors (TPV, FPV) for a model.

        Args:
            labels_path (str): Path to the labels JSON file.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing mask, TPV and FPV tensors.
        """
        rank_zero_info("\t Generating model label masks and properties")
        labels_dict = self._load_model_labels(labels_path)

        model_label_indices, tpv_label_values, fpv_label_values = [], [], []

        for label, props in labels_dict.items():
            if label in self._dm_labels:
                try:
                    self._validate_model_labels_json_element(labels_dict[label])
                except Exception as e:
                    raise Exception(f"Label '{label}' has an unexpected error") from e

                model_label_indices.append(self._dm_labels[label])
                tpv_label_values.append(props["TPV"])
                fpv_label_values.append(props["FPV"])

        if not all([model_label_indices, tpv_label_values, fpv_label_values]):
            raise ValueError(f"No valid label values found in {labels_path}.")

        # Create masks to apply predictions only to known classes
        mask = torch.zeros(self._num_of_labels, dtype=torch.bool, device=self._device)
        mask[torch.tensor(model_label_indices, device=self._device)] = True

        tpv_tensor = torch.full_like(mask, -1, dtype=torch.float, device=self._device)
        fpv_tensor = torch.full_like(mask, -1, dtype=torch.float, device=self._device)

        tpv_tensor[mask] = torch.tensor(
            tpv_label_values, dtype=torch.float, device=self._device
        )
        fpv_tensor[mask] = torch.tensor(
            fpv_label_values, dtype=torch.float, device=self._device
        )

        self._num_models_per_label += mask
        return {"mask": mask, "tpv_tensor": tpv_tensor, "fpv_tensor": fpv_tensor}

    @staticmethod
    def _load_model_labels(labels_path: str) -> dict[str, dict[str, float]]:
        """
        Loads a JSON label file for a model.

        Args:
            labels_path (str): Path to the JSON file.

        Returns:
            Dict[str, Dict[str, float]]: Parsed label confidence data.

        Raises:
            FileNotFoundError: If the file is missing.
            TypeError: If the file is not a JSON.
        """
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"{labels_path} does not exist.")
        if not labels_path.endswith(".json"):
            raise TypeError(f"{labels_path} is not a JSON file.")
        with open(labels_path, "r") as f:
            return json.load(f)

    @staticmethod
    def _validate_model_labels_json_element(label_dict: dict[str, str]) -> None:
        """
        Validates a label confidence dictionary to ensure required keys and values are valid.

        Args:
            label_dict (Dict[str, Any]): Label data with TPV and FPV keys.

        Raises:
            AttributeError: If required keys are missing.
            ValueError: If values are not valid floats or are negative.
        """
        for key in ["TPV", "FPV"]:
            if key not in label_dict:
                raise AttributeError(f"Missing key '{key}' in label dict.")
            try:
                value = float(label_dict[key])
                if value < 0:
                    raise ValueError(f"'{key}' must be non-negative but got {value}")
            except (TypeError, ValueError) as e:
                raise ValueError(f"Invalid value for '{key}': {label_dict[key]}") from e
