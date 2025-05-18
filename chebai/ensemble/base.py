import importlib
import json
import os
from abc import ABC, abstractmethod
from collections import deque
from typing import Deque, Dict, Optional

import torch
from lightning import LightningModule

from chebai.models import ChebaiBaseNet
from chebai.result.classification import print_metrics


class EnsembleBase(ABC):
    """
    Base class for ensemble models in the Chebai framework.

    Inherits from ChebaiBaseNet and provides functionality to load multiple models,
    validate configuration, and manage predictions.

    Attributes:
        data_processed_dir_main (str): Directory where the processed data is stored.
        models (Dict[str, LightningModule]): A dictionary of loaded models.
        model_configs (Dict[str, Dict]): Configuration dictionary for models in the ensemble.
        dm_labels (Dict[str, int]): Mapping of label names to integer indices.
    """

    def __init__(
        self, model_configs: Dict[str, Dict], data_processed_dir_main: str, **kwargs
    ):
        """
        Initializes the ensemble model and loads configuration, models, and labels.

        Args:
            model_configs (Dict[str, Dict]): Dictionary of model configurations.
            data_processed_dir_main (str): Path to the processed data directory.
            **kwargs: Additional arguments for initialization.
        """
        if kwargs.get("_validate_configs", False):
            self._validate_model_configs(model_configs)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim = kwargs.get("input_dim", None)
        self.num_of_labels: Optional[int] = (
            None  # will be set by `_load_data_module_labels` method
        )
        self.data_processed_dir_main = data_processed_dir_main
        self.models: Dict[str, LightningModule] = {}
        self.model_configs = model_configs
        self.dm_labels: Dict[str, int] = {}

        self._load_data_module_labels()
        self._num_models_per_label: torch.Tensor = torch.zeros(
            1, self.num_of_labels, device=self.device
        )
        self._model_queue: Deque = deque()
        self._collated_data = None

    @classmethod
    def _validate_model_configs(cls, model_configs: Dict[str, Dict]):
        """
        Validates the model configurations to ensure required keys are present.

        Args:
            model_configs (Dict[str, Dict]): Dictionary of model configurations.

        Raises:
            AttributeError: If required keys are missing in the configuration.
            ValueError: If there are duplicate model paths or class paths.
        """
        path_set, class_set, labels_set = set(), set(), set()

        required_keys = {"class_path", "ckpt_path", "labels_path"}

        for model_name, config in model_configs.items():
            missing_keys = required_keys - config.keys()

            if missing_keys:
                raise AttributeError(
                    f"Missing keys {missing_keys} in model '{model_name}' configuration."
                )

            model_path = config["ckpt_path"]
            class_path = config["class_path"]
            labels_path = config["labels_path"]

            if model_path in path_set:
                raise ValueError(
                    f"Duplicate model path detected: '{model_path}'. "
                    f"Each model must have a unique model-checkpoint path."
                )

            if class_path in class_set:
                raise ValueError(
                    f"Duplicate class path detected: '{class_path}'. Each model must have a unique class path."
                )

            if labels_path in labels_set:
                raise ValueError(
                    f"Duplicate labels path: {labels_path}. Each model must have unique labels path."
                )

            path_set.add(model_path)
            class_set.add(class_path)
            labels_set.add(labels_path)

    def _load_data_module_labels(self):
        """
        Loads the label mapping from the classes.txt file for loaded data.

        Raises:
            FileNotFoundError: If the classes.txt file does not exist.
        """
        classes_txt_file = os.path.join(self.data_processed_dir_main, "classes.txt")
        if not os.path.exists(classes_txt_file):
            raise FileNotFoundError(f"{classes_txt_file} does not exist")
        else:
            with open(classes_txt_file, "r") as f:
                for line in f:
                    if line.strip() not in self.dm_labels:
                        self.dm_labels[line.strip()] = len(self.dm_labels)
        self.num_of_labels = len(self.dm_labels)

    def run_ensemble(self):
        batch_size = 10
        true_scores = torch.zeros(batch_size, self.num_of_labels, device=self.device)
        false_scores = torch.zeros(batch_size, self.num_of_labels, device=self.device)

        while self._model_queue:
            model, model_props = self._load_model_and_its_props(
                self._model_queue.popleft()
            )
            pred_conf_dict = self._controller(model, model_props)
            del model  # Model can be huge to keep it in memory, delete as no longer needed

            self._consolidator(
                pred_conf_dict,
                model_props,
                true_scores=true_scores,
                false_scores=false_scores,
            )

        final_preds = self._consolidate_on_finish(
            true_scores=true_scores, false_scores=false_scores
        )
        print_metrics(
            final_preds,
            self._collated_data.y,
            self.device,
            classes=list(self.dm_labels.keys()),
        )

    def _load_model_and_its_props(self, model_name):
        """
        Loads the models specified in the configuration and initializes them.
        """
        model_ckpt_path = self.model_configs[model_name]["ckpt_path"]
        model_class_path = self.model_configs[model_name]["class_path"]
        model_labels_path = self.model_configs[model_name]["labels_path"]
        if not os.path.exists(model_ckpt_path):
            raise FileNotFoundError(
                f"Model path '{model_ckpt_path}' for '{model_name}' does not exist."
            )

        class_name = model_class_path.split(".")[-1]
        module_path = ".".join(model_class_path.split(".")[:-1])
        module = importlib.import_module(module_path)
        lightning_cls: LightningModule = getattr(module, class_name)
        assert isinstance(lightning_cls, type), f"{class_name} is not a class."
        assert issubclass(
            lightning_cls, ChebaiBaseNet
        ), f"{class_name} must inherit from ChebaiBaseNet"

        model = lightning_cls.load_from_checkpoint(
            model_ckpt_path, input_dim=self.input_dim
        )
        model.eval()
        model.freeze()

        model_label_props = self._generate_model_label_props(
            model_name, model_labels_path
        )

        return model, model_label_props

    def _generate_model_label_props(self, model_name: str, labels_path: str):
        """
        Generates a mask indicating the labels handled by each model, and retrieves corresponding the TPV and FPV values
        as tensors.

        Raises:
            FileNotFoundError: If the labels path does not exist.
            ValueError: If label values are empty for any model.
        """
        labels_dict = self._load_model_labels(labels_path)

        model_label_indices, tpv_label_values, fpv_label_values = [], [], []
        for label in labels_dict.keys():
            if label in self.dm_labels:
                try:
                    self._validate_model_labels_json_element(labels_dict[label])
                except Exception as e:
                    raise Exception(f"Label '{label}' has an unexpected error: {e}")

                model_label_indices.append(self.dm_labels[label])
                tpv_label_values.append(labels_dict[label]["TPV"])
                fpv_label_values.append(labels_dict[label]["FPV"])

        if not all([model_label_indices, tpv_label_values, fpv_label_values]):
            raise ValueError(f"Values are empty for labels of model {model_name}")

        # Create masks to apply predictions only to known classes
        mask = torch.zeros(self.num_of_labels, device=self.device, dtype=torch.bool)
        mask[torch.tensor(model_label_indices, dtype=torch.int, device=self.device)] = (
            True
        )

        tpv_tensor = torch.full_like(mask, -1, dtype=torch.float, device=self.device)
        fpv_tensor = torch.full_like(mask, -1, dtype=torch.float, device=self.device)

        tpv_tensor[mask] = torch.tensor(
            tpv_label_values, dtype=torch.float, device=self.device
        )
        fpv_tensor[mask] = torch.tensor(
            fpv_label_values, dtype=torch.float, device=self.device
        )
        self._num_models_per_label += mask
        return {"mask": mask, "tpv_tensor": tpv_tensor, "fpv_tensor": fpv_tensor}

    @staticmethod
    def _load_model_labels(labels_path: str) -> Dict[str, Dict[str, float]]:
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"{labels_path} does not exist.")

        if not labels_path.endswith(".json"):
            raise TypeError(f"{labels_path} is not a JSON file.")

        with open(labels_path, "r") as f:
            model_labels = json.load(f)
        return model_labels

    @staticmethod
    def _validate_model_labels_json_element(label_dict: Dict[str, float]):
        if "TPV" not in label_dict.keys() or "FPV" not in label_dict.keys():
            raise AttributeError(f"Missing keys 'TPV' and/or 'FPV'")

        # Validate 'tpv' and 'fpv' are either floats or convertible to float
        for key in ["TPV", "FPV"]:
            try:
                value = float(label_dict[key])
                if value < 0:
                    raise ValueError(f"'{key}' must be non-negative but got {value}")
            except (TypeError, ValueError):
                raise ValueError(
                    f"'{key}' must be a float or convertible to float, but got {label_dict[key]}"
                )

    @abstractmethod
    def _controller(self, model, model_props, **kwargs):
        pass

    @abstractmethod
    def _consolidator(
        self, pred_conf_dict, model_props, *, true_scores, false_scores, **kwargs
    ):
        pass

    @abstractmethod
    def _consolidate_on_finish(self, *, true_scores, false_scores):
        pass
