from abc import ABC, abstractmethod
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict

import pandas as pd
import torch

from chebai.result.classification import print_metrics

from ._constants import EVAL_OP, PRED_OP, WRAPPER_CLS_PATH


class EnsembleBase(ABC):
    """
    Base class for ensemble models in the Chebai framework.

    Handles loading, validating, and coordinating multiple models for ensemble prediction.
    """

    def __init__(
        self,
        model_configs: Dict[str, Dict[str, Any]],
        data_processed_dir_main: str,
        operation_mode: str = EVAL_OP,
        **kwargs: Any,
    ) -> None:
        """
        Initializes the ensemble model and loads configurations, labels, and sets up the environment.

        Args:
            model_configs (Dict[str, Dict[str, Any]]): Dictionary of model configurations.
            data_processed_dir_main (str): Path to the processed data directory.
            **kwargs (Any): Additional arguments, such as 'input_dim' and '_validate_configs'.
        """
        if bool(kwargs.get("_perform_validation_checks", True)):
            self._perform_validation_checks(
                model_configs, operation=operation_mode, **kwargs
            )

        self._model_configs: Dict[str, Dict[str, Any]] = model_configs
        self._data_processed_dir_main: str = data_processed_dir_main
        self._operation_mode: str = operation_mode
        print(f"Ensemble operation: {self._operation_mode}")

        # These instance variable will be set in method `_process_input_to_ensemble`
        self._total_data_size: int | None = None
        self._ensemble_input: list[str] | Path = self._process_input_to_ensemble(
            **kwargs
        )
        print(f"Total data size (data.pkl) is {self._total_data_size}")

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._dm_labels: Dict[str, int] = self._load_data_module_labels()
        self._num_of_labels: int = len(self._dm_labels)
        print(f"Number of labels for this data is {self._num_of_labels} ")

        self._num_models_per_label: torch.Tensor = torch.zeros(
            1, self._num_of_labels, device=self._device
        )
        self._model_queue: Deque[str] = deque()
        self._collated_labels: torch.Tensor | None = None

    @classmethod
    def _perform_validation_checks(
        cls, model_configs: Dict[str, Dict[str, Any]], operation, **kwargs
    ) -> None:
        """
        Validates model configuration dictionary for required keys and uniqueness.

        Args:
            model_configs (Dict[str, Dict[str, Any]]): Model configuration dictionary.

        Raises:
            AttributeError: If any model config is missing required keys.
            ValueError: If duplicate paths are found for model checkpoint, class, or labels.
        """
        if operation not in ["evaluate", "predict"]:
            raise ValueError(
                f"Invalid operation '{operation}'. Must be 'evaluate' or 'predict'."
            )

        if operation == "predict":
            if kwargs.get("smiles_list_file_path", None):
                raise ValueError(
                    "For 'predict' operation, 'smiles_list_file_path' must be provided."
                )

            if not Path(kwargs.get("smiles_list_file_path")).exists():
                raise FileNotFoundError(f"{kwargs.get('smiles_list_file_path')}")

        required_keys = {WRAPPER_CLS_PATH}

        for model_name, config in model_configs.items():
            missing_keys = required_keys - config.keys()
            if missing_keys:
                raise AttributeError(
                    f"Missing keys {missing_keys} in model '{model_name}' configuration."
                )

    def _process_input_to_ensemble(self, **kwargs: Any) -> list[str] | Path:
        if self._operation_mode == PRED_OP:
            p = Path(kwargs["smiles_list_file_path"])
            smiles_list: list[str] = []
            with open(p, "r") as f:
                for line in f:
                    # Skip empty or whitespace-only lines
                    if line.strip():
                        # Split on whitespace and take the first item as the SMILES
                        smiles = line.strip().split()[0]
                        smiles_list.append(smiles)
            self._total_data_size = len(smiles_list)
            return smiles_list
        elif self._operation_mode == EVAL_OP:
            processed_dir_path = Path(self._data_processed_dir_main)
            data_pkl_path = processed_dir_path / "data.pkl"
            if not data_pkl_path.exists():
                raise FileNotFoundError(
                    f"data.pkl does not exist in the {processed_dir_path} directory"
                )
            self._total_data_size = len(pd.read_pickle(data_pkl_path))
            return data_pkl_path
        else:
            raise ValueError("Invalid operation")

    def _load_data_module_labels(self) -> dict[str, int]:
        """
        Loads class labels from the classes.txt file and sets internal label mapping.

        Raises:
            FileNotFoundError: If the expected classes.txt file is not found.
        """
        classes_file_path = Path(self._data_processed_dir_main) / "classes.txt"
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

        print(
            f"Running {self.__class__.__name__} ensemble for {self._operation_mode} operation..."
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

        final_preds = self._consolidate_on_finish(
            true_scores=true_scores, false_scores=false_scores
        )

        if self._operation_mode == EVAL_OP:
            assert (
                self._collated_labels is not None
            ), "Collated labels must be set for evaluation operation."
            print_metrics(
                final_preds,
                self._collated_labels,
                self._device,
                classes=list(self._dm_labels.keys()),
            )
        else:
            # Get SMILES and label names
            smiles_list = self._ensemble_input
            label_names = list(self._dm_labels.keys())
            # Efficient conversion from tensor to NumPy
            preds_np = final_preds.detach().cpu().numpy()

            assert (
                len(smiles_list) == preds_np.shape[0]
            ), "Length of SMILES list does not match number of predictions."
            assert (
                len(label_names) == preds_np.shape[1]
            ), "Number of label names does not match number of predictions."

            # Build DataFrame
            df = pd.DataFrame(preds_np, columns=label_names)
            df.insert(0, "SMILES", smiles_list)

            # Save to CSV
            output_path = (
                Path(self._data_processed_dir_main) / "ensemble_predictions.csv"
            )
            df.to_csv(output_path, index=False)

            print(f"Predictions saved to {output_path}")

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

    @abstractmethod
    def _consolidator(
        self,
        *,
        pred_conf_dict: Dict[str, torch.Tensor],
        model_props: Dict[str, torch.Tensor],
        true_scores: torch.Tensor,
        false_scores: torch.Tensor,
        **kwargs: Any,
    ) -> None:
        """
        Abstract method to define aggregation logic.

        Should update the provided `true_scores` and `false_scores`.
        """

    @abstractmethod
    def _consolidate_on_finish(
        self, *, true_scores: torch.Tensor, false_scores: torch.Tensor
    ) -> torch.Tensor:
        """
        Abstract method to produce final predictions after all models have been evaluated.

        Returns:
            torch.Tensor: Final aggregated predictions.
        """
