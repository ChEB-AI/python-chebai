import importlib
import json
import os
from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Deque, Dict, Optional, Tuple

import torch
from lightning import LightningModule
from lightning_utilities.core.rank_zero import rank_zero_info

from chebai.models import ChebaiBaseNet
from chebai.preprocessing.structures import XYData
from chebai.result.classification import print_metrics

from ._constants import *


class EnsembleBase(ABC):
    """
    Base class for ensemble models in the Chebai framework.

    Handles loading, validating, and coordinating multiple models for ensemble prediction.
    """

    def __init__(
        self,
        model_configs: Dict[str, Dict[str, Any]],
        data_processed_dir_main: str,
        **kwargs: Any,
    ) -> None:
        """
        Initializes the ensemble model and loads configurations, labels, and sets up the environment.

        Args:
            model_configs (Dict[str, Dict[str, Any]]): Dictionary of model configurations.
            data_processed_dir_main (str): Path to the processed data directory.
            reader_dir_name (str): Name of the directory used by the reader. Defaults to 'smiles_token'.
            **kwargs (Any): Additional arguments, such as 'input_dim' and '_validate_configs'.
        """
        if bool(kwargs.get("_validate_configs", True)):
            self._validate_model_configs(model_configs)

        self.model_configs: Dict[str, Dict[str, Any]] = model_configs
        self.data_processed_dir_main: str = data_processed_dir_main
        self.input_dim: Optional[int] = kwargs.get("input_dim", None)

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._num_of_labels: Optional[int] = (
            None  # will be set by `_load_data_module_labels` method
        )
        self._models: Dict[str, LightningModule] = {}
        self._dm_labels: Dict[str, int] = {}

        self._load_data_module_labels()
        self._num_models_per_label: torch.Tensor = torch.zeros(
            1, self._num_of_labels, device=self._device
        )
        self._model_queue: Deque[str] = deque()
        self._collated_data: Optional[XYData] = None
        self._total_data_size: Optional[int] = None

    @classmethod
    def _validate_model_configs(cls, model_configs: Dict[str, Dict[str, Any]]) -> None:
        """
        Validates model configuration dictionary for required keys and uniqueness.

        Args:
            model_configs (Dict[str, Dict[str, Any]]): Model configuration dictionary.

        Raises:
            AttributeError: If any model config is missing required keys.
            ValueError: If duplicate paths are found for model checkpoint, class, or labels.
        """
        path_set, class_set, labels_set = set(), set(), set()
        required_keys = {
            MODEL_CKPT_PATH,
            MODEL_CLS_PATH,
            MODEL_LBL_PATH,
            WRAPPER_CLS_PATH,
            READER_CLS_PATH,
        }

        for model_name, config in model_configs.items():
            missing_keys = required_keys - config.keys()
            if missing_keys:
                raise AttributeError(
                    f"Missing keys {missing_keys} in model '{model_name}' configuration."
                )

            model_ckpt_path, model_class_path, model_labels_path = (
                config[MODEL_CKPT_PATH],
                config[MODEL_CLS_PATH],
                config[MODEL_LBL_PATH],
            )

            if model_ckpt_path in path_set:
                raise ValueError(f"Duplicate model path detected: '{model_ckpt_path}'.")
            if model_class_path in class_set:
                raise ValueError(
                    f"Duplicate class path detected: '{model_class_path}'."
                )
            if model_labels_path in labels_set:
                raise ValueError(f"Duplicate labels path: {model_labels_path}.")

            path_set.add(model_ckpt_path)
            class_set.add(model_class_path)
            labels_set.add(model_labels_path)

    def _load_data_module_labels(self) -> None:
        """
        Loads class labels from the classes.txt file and sets internal label mapping.

        Raises:
            FileNotFoundError: If the expected classes.txt file is not found.
        """
        classes_txt_file = os.path.join(self.data_processed_dir_main, "classes.txt")
        rank_zero_info(f"Loading {classes_txt_file} ....")

        if not os.path.exists(classes_txt_file):
            raise FileNotFoundError(f"{classes_txt_file} does not exist")

        with open(classes_txt_file, "r") as f:
            for line in f:
                label = line.strip()
                if label not in self._dm_labels:
                    self._dm_labels[label] = len(self._dm_labels)
        self._num_of_labels = len(self._dm_labels)

    def run_ensemble(self) -> None:
        """
        Executes the full ensemble prediction pipeline, aggregating predictions and printing metrics.
        """
        true_scores = torch.zeros(
            self._total_data_size, self._num_of_labels, device=self._device
        )
        false_scores = torch.zeros(
            self._total_data_size, self._num_of_labels, device=self._device
        )

        while self._model_queue:
            model_name = self._model_queue.popleft()
            rank_zero_info(f"Processing model: {model_name}")
            model, model_props = self._load_model_and_its_props(model_name)

            rank_zero_info("\t Passing model to controller to generate predictions...")
            pred_conf_dict = self._controller(model, model_props)
            del model  # Model can be huge to keep it in memory, delete as no longer needed

            rank_zero_info("\t Passing predictions to consolidator for aggregation...")
            self._consolidator(
                pred_conf_dict,
                model_props,
                true_scores=true_scores,
                false_scores=false_scores,
            )

        rank_zero_info(f"Consolidating predictions for {self.__class__.__name__}")
        final_preds = self._consolidate_on_finish(
            true_scores=true_scores, false_scores=false_scores
        )
        print_metrics(
            final_preds,
            self._collated_data.y,
            self._device,
            classes=list(self._dm_labels.keys()),
        )

    def _load_model_and_its_props(
        self, model_name: str
    ) -> Tuple[LightningModule, Dict[str, torch.Tensor]]:
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

        def load_class(class_path):
            class_name = class_path.split(".")[-1]
            module_path = ".".join(class_path.split(".")[:-1])
            module = importlib.import_module(module_path)
            return getattr(module, class_name)

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

    def _generate_model_label_props(self, labels_path: str) -> Dict[str, torch.Tensor]:
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
    def _load_model_labels(labels_path: str) -> Dict[str, Dict[str, float]]:
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
    def _validate_model_labels_json_element(label_dict: Dict[str, Any]) -> None:
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

    @abstractmethod
    def _controller(
        self,
        model: LightningModule,
        model_props: Dict[str, torch.Tensor],
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """
        Abstract method to define model-specific prediction logic.

        Returns:
            Dict[str, torch.Tensor]: Predictions or confidence scores.
        """
        pass

    @abstractmethod
    def _consolidator(
        self,
        pred_conf_dict: Dict[str, torch.Tensor],
        model_props: Dict[str, torch.Tensor],
        *,
        true_scores: torch.Tensor,
        false_scores: torch.Tensor,
        **kwargs: Any,
    ) -> None:
        """
        Abstract method to define aggregation logic.

        Should update the provided `true_scores` and `false_scores`.
        """
        pass

    @abstractmethod
    def _consolidate_on_finish(
        self, *, true_scores: torch.Tensor, false_scores: torch.Tensor
    ) -> torch.Tensor:
        """
        Abstract method to produce final predictions after all models have been evaluated.

        Returns:
            torch.Tensor: Final aggregated predictions.
        """
        pass
