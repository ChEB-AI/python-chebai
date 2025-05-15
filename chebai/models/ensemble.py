import importlib
import json
import os.path
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union

import torch
from lightning.pytorch import LightningModule
from torch import Tensor

from chebai.models import ChebaiBaseNet
from chebai.preprocessing.structures import XYData


class _EnsembleBase(ChebaiBaseNet, ABC):
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
        super().__init__(**kwargs)
        if kwargs.get("_validate_configs", True):
            self._validate_model_configs(model_configs)

        self.data_processed_dir_main = data_processed_dir_main
        self.models: Dict[str, LightningModule] = {}
        self.model_configs = model_configs
        self.dm_labels: Dict[str, int] = {}

        self._load_data_module_labels()
        self._load_ensemble_models()

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

    def _load_ensemble_models(self):
        """
        Loads the models specified in the configuration and initializes them.
        """
        for model_name in self.model_configs:
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

            model = lightning_cls.load_from_checkpoint(
                model_ckpt_path, input_dim=self.input_dim
            )
            model.eval()
            model.freeze()

            self.models[model_name] = model
            self.model_configs[model_name]["labels"] = self._load_model_labels(
                model_labels_path, model_name
            )

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

    @staticmethod
    def _load_model_labels(labels_path: str, model_name: str) -> Dict[str, float]:
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"{labels_path} does not exist.")

        if not labels_path.endswith(".json"):
            raise TypeError(f"{labels_path} is not a JSON file.")

        with open(labels_path, "r") as f:
            model_labels = json.load(f)

        labels_dict = {}
        for label, label_dict in model_labels.items():
            msg = f"for model {model_name} for label {label}"
            if "TPV" not in label_dict.keys() or "FPV" not in label_dict.keys():
                raise AttributeError(f"Missing keys 'TPV' and/or 'FPV' {msg}")

            # Validate 'tpv' and 'fpv' are either floats or convertible to float
            for key in ["TPV", "FPV"]:
                try:
                    value = float(label_dict[key])
                    if value < 0:
                        raise ValueError(
                            f"'{key}' must be non-negative but got {value} {msg}"
                        )
                except (TypeError, ValueError):
                    raise ValueError(
                        f"'{key}' must be a float or convertible to float, but got {label_dict[key]} {msg}"
                    )
                labels_dict.setdefault(label, {})[key] = value
        return labels_dict

    @abstractmethod
    def _get_prediction_and_labels(
        self, data: Dict[str, Any], labels: torch.Tensor, output: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Abstract method for obtaining predictions and labels.

        Args:
            data (Dict[str, Any]): The input data.
            labels (torch.Tensor): The target labels.
            output (torch.Tensor): The model output.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The predicted labels and the ground truth labels.
        """
        pass

    def controller(self):
        pass

    def consolidator(
        self,
    ):
        pass


class ChebiEnsemble(_EnsembleBase):
    """
    Ensemble model that aggregates predictions from multiple models for the Chebai task.

    This model combines the outputs of several individual models and aggregates their predictions
    using a weighted voting strategy based on trustworthiness (TPV and FPV). This strategy can modified by overriding
    `aggregate_predictions` method by subclasses, as per needs.

    There is are relevant trainable parameters for this ensemble model, hence trainer.max_epochs should be set to 1.
    `_dummy_param` exists for only lighting module completeness and compatability purpose.
    """

    NAME = "ChebiEnsemble"

    def __init__(self, model_configs: Dict[str, Dict], **kwargs):
        """
        Initializes the ensemble model and computes the model-label mask.

        Args:
            model_configs (Dict[str, Dict]): Dictionary of model configurations.
            **kwargs: Additional arguments for initialization.
        """
        super().__init__(model_configs, **kwargs)

        # Add a dummy trainable parameter
        self.dummy_param = torch.nn.Parameter(torch.randn(1, requires_grad=True))
        self._num_models_per_label: Optional[torch.Tensor] = None
        self._generate_model_label_mask()

    def _generate_model_label_mask(self):
        """
        Generates a mask indicating the labels handled by each model, and retrieves corresponding the TPV and FPV values
        as tensors.

        Raises:
            FileNotFoundError: If the labels path does not exist.
            ValueError: If label values are empty for any model.
        """
        num_models_per_label = torch.zeros(1, self.out_dim, device=self.device)

        for model_name, model_config in self.model_configs.items():
            labels_dict = model_config["labels"]

            model_label_indices, tpv_label_values, fpv_label_values = [], [], []
            for label in labels_dict.keys():
                if label in self.dm_labels:
                    model_label_indices.append(self.dm_labels[label])
                    tpv_label_values.append(labels_dict[label]["TPV"])
                    fpv_label_values.append(labels_dict[label]["FPV"])

            if not all([model_label_indices, tpv_label_values, fpv_label_values]):
                raise ValueError(f"Values are empty for labels of model {model_name}")

            # Create masks to apply predictions only to known classes
            mask = torch.zeros(self.out_dim, device=self.device, dtype=torch.bool)
            mask[
                torch.tensor(model_label_indices, dtype=torch.int, device=self.device)
            ] = True

            tpv_tensor = torch.full_like(
                mask, -1, dtype=torch.float, device=self.device
            )
            fpv_tensor = torch.full_like(
                mask, -1, dtype=torch.float, device=self.device
            )

            tpv_tensor[mask] = torch.tensor(
                tpv_label_values, dtype=torch.float, device=self.device
            )
            fpv_tensor[mask] = torch.tensor(
                fpv_label_values, dtype=torch.float, device=self.device
            )

            self.model_configs[model_name]["labels_mask"] = mask
            self.model_configs[model_name]["tpv_tensor"] = tpv_tensor
            self.model_configs[model_name]["fpv_tensor"] = fpv_tensor
            num_models_per_label += mask

        self._num_models_per_label = num_models_per_label

    def forward(self, data: Dict[str, Tensor], **kwargs: Any) -> Dict[str, Any]:
        """
        Forward pass through the ensemble model, aggregating predictions from all models.

        Args:
            data (Dict[str, Tensor]): Input data including features and labels.
            **kwargs: Additional arguments for the forward pass.

        Returns:
            Dict[str, Any]: The aggregated logits, predictions, and confidences.
        """
        predictions = {}
        confidences = {}

        assert data["labels"].shape[1] == self.out_dim

        # Initialize total_logits with zeros
        total_logits = torch.zeros(
            data["labels"].shape[0], self.out_dim, device=self.device
        )

        for name, model in self.models.items():
            output = model(data)
            mask = self.model_configs[name]["labels_mask"]

            # Consider logits and confidence only for valid classes
            sigmoid_logits = torch.sigmoid(output["logits"])
            prediction = torch.full_like(total_logits, -1, dtype=torch.bool)
            confidence = torch.full_like(total_logits, -1, dtype=torch.float)
            prediction[:, mask] = sigmoid_logits > 0.5
            confidence[:, mask] = 2 * torch.abs(sigmoid_logits - 0.5)

            predictions[name] = prediction
            confidences[name] = confidence
            total_logits += output[
                "logits"
            ]  # This doesn't play a role here, just for lightning flow completeness

        return {
            "logits": total_logits,
            "pred_dict": predictions,
            "conf_dict": confidences,
        }

    def _get_prediction_and_labels(self, data, labels, model_output):
        """
        Gets predictions and labels from the model output.

        Args:
            data (Dict[str, Any]): The input data.
            labels (torch.Tensor): The target labels.
            model_output (Dict[str, Tensor]): The model's output.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The predictions and the ground truth labels.
        """
        d = model_output["logits"]
        # Aggregate predictions using weighted voting
        metrics_preds = self.aggregate_predictions(
            model_output["pred_dict"], model_output["conf_dict"]
        )
        loss_kwargs = data.get("loss_kwargs", dict())
        if "non_null_labels" in loss_kwargs:
            n = loss_kwargs["non_null_labels"]
            d = d[n]
            metrics_preds = metrics_preds[n]
        return (
            torch.sigmoid(d),
            labels.int() if labels is not None else None,
            metrics_preds,
        )

    def _execute(
        self,
        batch: XYData,
        batch_idx: int,
        metrics: Optional[torch.nn.Module] = None,
        prefix: Optional[str] = "",
        log: Optional[bool] = True,
        sync_dist: Optional[bool] = False,
    ) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        Executes the model on a batch of data and returns the model output and predictions.

        Args:
            batch (XYData): The input batch of data.
            batch_idx (int): The index of the current batch.
            metrics (torch.nn.Module): A dictionary of metrics to track.
            prefix (str, optional): A prefix to add to the metric names. Defaults to "".
            log (bool, optional): Whether to log the metrics. Defaults to True.
            sync_dist (bool, optional): Whether to synchronize distributed training. Defaults to False.

        Returns:
            Dict[str, Union[torch.Tensor, Any]]: A dictionary containing the processed data, labels, model output,
            predictions, and loss (if applicable).
        """
        assert isinstance(batch, XYData)
        batch = batch.to(self.device)
        data = self._process_batch(batch, batch_idx)
        labels = data["labels"]
        model_output = self(data, **data.get("model_kwargs", dict()))
        pr, tar, metrics_preds = self._get_prediction_and_labels(
            data, labels, model_output
        )
        d = dict(data=data, labels=labels, output=model_output, preds=pr)
        if log:
            if self.criterion is not None:
                loss_data, loss_labels, loss_kwargs_candidates = self._process_for_loss(
                    model_output, labels, data.get("loss_kwargs", dict())
                )
                loss_kwargs = dict()
                if self.pass_loss_kwargs:
                    loss_kwargs = loss_kwargs_candidates
                loss = self.criterion(loss_data, loss_labels, **loss_kwargs)
                if isinstance(loss, tuple):
                    loss_additional = loss[1:]
                    for i, loss_add in enumerate(loss_additional):
                        self.log(
                            f"{prefix}loss_{i}",
                            loss_add if isinstance(loss_add, int) else loss_add.item(),
                            batch_size=len(batch),
                            on_step=True,
                            on_epoch=False,
                            prog_bar=False,
                            logger=True,
                            sync_dist=sync_dist,
                        )
                    loss = loss[0]

                d["loss"] = loss + 0 * self.dummy_param.sum()

                self.log(
                    f"{prefix}loss",
                    loss.item(),
                    batch_size=len(batch),
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                    sync_dist=sync_dist,
                )
            if metrics and labels is not None:
                for metric_name, metric in metrics.items():
                    metric.update(metrics_preds, tar)
                self._log_metrics(prefix, metrics, len(batch))
        return d

    def aggregate_predictions(
        self, predictions: Dict[str, torch.Tensor], confidences: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Implements weighted voting based on trustworthiness.

        This method aggregates predictions from multiple models using a weighted voting mechanism.
        The weight of each model's prediction is determined by its True Positive Value (TPV) and
        False Positive Value (FPV), scaled by the confidence score.

        Args:
            predictions (Dict[str, torch.Tensor]):
                A dictionary mapping model names to their respective binary class predictions
                (shape: `[batch_size, num_classes]`).
            confidences (Dict[str, torch.Tensor]):
                A dictionary mapping model names to their respective confidence scores
                (shape: `[batch_size, num_classes]`).

        Returns:
            torch.Tensor:
                A tensor of final aggregated predictions based on weighted voting
                (shape: `[batch_size, num_classes]`), where values are `True` for positive class
                and `False` otherwise.
        """
        batch_size, num_classes = list(predictions.values())[0].shape
        true_scores = torch.zeros(batch_size, num_classes, device=self.device)
        false_scores = torch.zeros(batch_size, num_classes, device=self.device)

        for model, conf in confidences.items():
            tpv = self.model_configs[model]["tpv_tensor"]
            npv = self.model_configs[model]["fpv_tensor"]

            # Determine which classes the model provides predictions for
            mask = self.model_configs[model]["labels_mask"]
            weight = conf * (tpv * conf + npv * (1 - conf))

            # Apply mask: Only update scores for valid classes
            true_scores += weight * conf * mask
            false_scores += weight * (1 - conf) * mask

        # Avoid division by zero: Set valid_counts to 1 where it's zero
        valid_counts = self._num_models_per_label.clamp(min=1)

        # Normalize by valid contributions to prevent bias
        final_preds = (true_scores / valid_counts) > (false_scores / valid_counts)

        return final_preds

    def _process_for_loss(
        self,
        model_output: Dict[str, Tensor],
        labels: Tensor,
        loss_kwargs: Dict[str, Any],
    ) -> Tuple[Tensor, Tensor, Dict[str, Any]]:
        """
        Process the model output for calculating the loss.

        Args:
            model_output (Dict[str, Tensor]): The output of the model.
            labels (Tensor): The target labels.
            loss_kwargs (Dict[str, Any]): Additional loss arguments.

        Returns:
            tuple: A tuple containing the processed model output, labels, and loss arguments.
        """
        kwargs_copy = dict(loss_kwargs)
        if labels is not None:
            labels = labels.float()
        return model_output["logits"], labels, kwargs_copy
