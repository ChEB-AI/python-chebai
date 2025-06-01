from abc import ABC, abstractmethod
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, Optional

import pandas as pd
import torch
from lightning import LightningModule

from chebai.result.classification import print_metrics

from ._constants import MODEL_CLS_PATH, MODEL_LBL_PATH, WRAPPER_CLS_PATH


class EnsembleBase(ABC):
    """
    Base class for ensemble models in the Chebai framework.

    Handles loading, validating, and coordinating multiple models for ensemble prediction.
    """

    def __init__(
        self,
        model_configs: Dict[str, Dict[str, Any]],
        data_file_path: str,
        classes_file_path: str,
        **kwargs: Any,
    ) -> None:
        """
        Initializes the ensemble model and loads configurations, labels, and sets up the environment.

        Args:
            model_configs (Dict[str, Dict[str, Any]]): Dictionary of model configurations.
            data_file_path (str): Path to the processed data directory.
            reader_dir_name (str): Name of the directory used by the reader. Defaults to 'smiles_token'.
            **kwargs (Any): Additional arguments, such as 'input_dim' and '_validate_configs'.
        """
        if bool(kwargs.get("_validate_configs", True)):
            self._validate_model_configs(model_configs)

        self._model_configs: Dict[str, Dict[str, Any]] = model_configs
        self._data_file_path: str = data_file_path
        self._classes_file_path: str = classes_file_path
        self._input_dim: Optional[int] = kwargs.get("input_dim", None)
        self._total_data_size: int = None
        self._ensemble_input: list[str] | Path = self._process_input_to_ensemble(
            data_file_path
        )
        print(f"Total data size (data.pkl) is {self._total_data_size}")

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._models: Dict[str, LightningModule] = {}
        self._dm_labels: Dict[str, int] = self._load_data_module_labels(
            classes_file_path
        )
        self._num_of_labels: int = len(self._dm_labels)
        print(f"Number of labes for this data is {self._num_of_labels} ")

        self._num_models_per_label: torch.Tensor = torch.zeros(
            1, self._num_of_labels, device=self._device
        )
        self._model_queue: Deque[str] = deque()

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
        class_set, labels_set = set(), set()
        required_keys = {
            MODEL_CLS_PATH,
            MODEL_LBL_PATH,
            WRAPPER_CLS_PATH,
        }

        for model_name, config in model_configs.items():
            missing_keys = required_keys - config.keys()
            if missing_keys:
                raise AttributeError(
                    f"Missing keys {missing_keys} in model '{model_name}' configuration."
                )

            model_class_path, model_labels_path = (
                config[MODEL_CLS_PATH],
                config[MODEL_LBL_PATH],
            )

            if model_class_path in class_set:
                raise ValueError(
                    f"Duplicate class path detected: '{model_class_path}'."
                )
            if model_labels_path in labels_set:
                raise ValueError(f"Duplicate labels path: {model_labels_path}.")

            class_set.add(model_class_path)
            labels_set.add(model_labels_path)

    def _process_input_to_ensemble(self, path: str):
        p = Path(path)
        if p.is_file():
            smiles_list = []
            with open(p, "r") as f:
                for line in f:
                    # Skip empty or whitespace-only lines
                    if line.strip():
                        # Split on whitespace and take the first item as the SMILES
                        smiles = line.strip().split()[0]
                        smiles_list.append(smiles)
            self._total_data_size = len(smiles_list)
            return smiles_list
        elif p.is_dir():
            data_pkl_path = p / "data.pkl"
            if not data_pkl_path.exists():
                raise FileNotFoundError()
            self._total_data_size = len(pd.read_pickle(data_pkl_path))
            return p
        else:
            raise "Invalid path"

    @staticmethod
    def _load_data_module_labels(classes_file_path: str) -> dict[str, int]:
        """
        Loads class labels from the classes.txt file and sets internal label mapping.

        Raises:
            FileNotFoundError: If the expected classes.txt file is not found.
        """
        classes_file_path = Path(classes_file_path)
        if not classes_file_path.exists():
            raise FileNotFoundError(f"{classes_file_path} does not exist")
        print(f"Loading {classes_file_path} ....")

        dm_labels_dict = {}
        with open(classes_file_path, "r") as f:
            for line in f:
                label = line.strip()
                if label not in dm_labels_dict:
                    dm_labels_dict[label] = len(dm_labels_dict)
        return dm_labels_dict

    def run_ensemble(self) -> None:
        """
        Executes the full ensemble prediction pipeline, aggregating predictions and printing metrics.
        """
        assert self._total_data_size is not None and self._num_of_labels is not None
        true_scores = torch.zeros(
            self._total_data_size, self._num_of_labels, device=self._device
        )
        false_scores = torch.zeros(
            self._total_data_size, self._num_of_labels, device=self._device
        )

        while self._model_queue:
            model_name = self._model_queue.popleft()
            print(f"Processing model: {model_name}")

            print("\t Passing model to controller to generate predictions...")
            controller_output = self._controller(model_name, self._ensemble_input)

            print("\t Passing predictions to consolidator for aggregation...")
            self._consolidator(
                pred_conf_dict=controller_output["pred_conf_dict"],
                model_props=controller_output["model_props"],
                true_scores=true_scores,
                false_scores=false_scores,
            )

        print(f"Consolidating predictions for {self.__class__.__name__}")
        final_preds = self._consolidate_on_finish(
            true_scores=true_scores, false_scores=false_scores
        )
        print_metrics(
            final_preds,
            self._collated_data.y,
            self._device,
            classes=list(self._dm_labels.keys()),
        )

    @abstractmethod
    def _controller(
        self,
        model_name: str,
        model_input: list[str] | Path,
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
