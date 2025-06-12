import json
import os
from abc import ABC, abstractmethod
from pathlib import Path

import torch

from .._constants import MODEL_LBL_PATH

_MODEL_REGISTRY = {}


class BaseWrapper(ABC):
    def __init__(
        self,
        model_name: str,
        model_config: dict[str, str],
        dm_labels: dict[str, int],
        **kwargs,
    ):
        if MODEL_LBL_PATH not in model_config:
            raise AttributeError(
                f"Missing key {MODEL_LBL_PATH} in model '{model_name}' configuration."
            )

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model_config = model_config
        self._model_name = model_name

        self._model_labels_path = self._model_config[MODEL_LBL_PATH]
        self._model_props = self._generate_model_label_props(dm_labels=dm_labels)
        self.collated_labels = None

    @classmethod
    def _cls_name(cls) -> str:
        return cls.__name__

    @property
    def name(self):
        return f"Wrapper({self._cls_name()}) for model: {self._model_name}"

    def __init_subclass__(cls):
        """
        Automatically registers subclasses in the model registry to prevent duplicates.

        Args:
            **kwargs: Additional keyword arguments.
        """
        if cls._cls_name() in _MODEL_REGISTRY:
            raise ValueError(f"Model {cls._cls_name()} does already exist")
        else:
            _MODEL_REGISTRY[cls._cls_name()] = cls

    def _generate_model_label_props(self, dm_labels) -> dict[str, torch.Tensor]:
        """
        Generates label mask and confidence tensors (TPV, FPV) for a model.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing mask, TPV and FPV tensors.
        """
        print("\t Generating model label masks and properties")
        labels_dict = self._load_model_labels()

        model_label_indices, tpv_label_values, fpv_label_values = [], [], []

        for label, props in labels_dict.items():
            if label in dm_labels:
                try:
                    self._validate_model_labels_json_element(labels_dict[label])
                except Exception as e:
                    raise RuntimeError(
                        f"Label '{label}' has an unexpected error \n Error: {e}"
                    ) from e

                model_label_indices.append(dm_labels[label])
                tpv_label_values.append(props["PPV"])
                fpv_label_values.append(props["NPV"])

        if not all([model_label_indices, tpv_label_values, fpv_label_values]):
            raise ValueError(
                f"No valid label values found in {self._model_labels_path}."
            )

        # Create masks to apply predictions only to known classes
        mask = torch.zeros(len(dm_labels), dtype=torch.bool, device=self._device)
        mask[torch.tensor(model_label_indices, device=self._device)] = True

        tpv_tensor = torch.full_like(mask, -1, dtype=torch.float, device=self._device)
        fpv_tensor = torch.full_like(mask, -1, dtype=torch.float, device=self._device)

        tpv_tensor[mask] = torch.tensor(
            tpv_label_values, dtype=torch.float, device=self._device
        )
        fpv_tensor[mask] = torch.tensor(
            fpv_label_values, dtype=torch.float, device=self._device
        )

        return {"mask": mask, "tpv_tensor": tpv_tensor, "fpv_tensor": fpv_tensor}

    def _load_model_labels(self) -> dict[str, dict[str, float]]:
        """
        Loads a JSON label file for a model.

        Returns:
            Dict[str, Dict[str, float]]: Parsed label confidence data.

        Raises:
            FileNotFoundError: If the file is missing.
            TypeError: If the file is not a JSON.
        """
        if not os.path.exists(self._model_labels_path):
            raise FileNotFoundError(f"{self._model_labels_path} does not exist.")
        if not self._model_labels_path.endswith(".json"):
            raise TypeError(f"{self._model_labels_path} is not a JSON file.")
        with open(self._model_labels_path, "r") as f:
            return json.load(f)

    @staticmethod
    def _validate_model_labels_json_element(label_dict: dict[str, float]) -> None:
        """
        Validates a label confidence dictionary to ensure required keys and values are valid.

        Args:
            label_dict (Dict[str, Any]): Label data with PPV and NPV keys.

        Raises:
            AttributeError: If required keys are missing.
            ValueError: If values are not valid floats or are negative.
        """
        for key in ["PPV", "NPV"]:
            if key not in label_dict:
                raise AttributeError(f"Missing key '{key}' in label dict.")
            try:
                value = float(label_dict[key])
                if value < 0:
                    raise ValueError(f"'{key}' must be non-negative but got {value}")
            except (TypeError, ValueError) as e:
                raise ValueError(f"Invalid value for '{key}': {label_dict[key]}") from e

    def predict(self, x: list) -> tuple[dict, dict]:
        if not isinstance(x, list):
            raise TypeError(f"Input must be a list of SMILES strings, got {type(x)}")
        return self._predict_from_list_of_smiles(x), self._model_props

    @abstractmethod
    def _predict_from_list_of_smiles(self, smiles_list: list) -> dict: ...

    def evaluate(self, **kwargs) -> tuple[dict, dict]:
        return self._evaluate_from_data_file(**kwargs), self._model_props

    @abstractmethod
    def _evaluate_from_data_file(self, **kwargs) -> dict: ...
