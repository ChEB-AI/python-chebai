import importlib
import os.path
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union

import torch
from lightning.pytorch import LightningModule
from torch import Tensor

from chebai.models import ChebaiBaseNet
from chebai.preprocessing.structures import XYData


class _EnsembleBase(ChebaiBaseNet, ABC):
    def __init__(self, model_configs: Dict[str, Dict], **kwargs):
        super().__init__(**kwargs)
        self._validate_model_configs(model_configs)

        self.models: Dict[str, LightningModule] = {}
        self.model_configs = model_configs

        for model_name in self.model_configs:
            model_ckpt_path = self.model_configs[model_name]["ckpt_path"]
            model_class_path = self.model_configs[model_name]["class_path"]
            if not os.path.exists(model_ckpt_path):
                raise FileNotFoundError(
                    f"Model path '{model_ckpt_path}' for '{model_name}' does not exist."
                )

            class_name = model_class_path.split(".")[-1]
            module_path = ".".join(model_class_path.split(".")[:-1])

            try:
                module = importlib.import_module(module_path)
                lightning_cls: LightningModule = getattr(module, class_name)

                model = lightning_cls.load_from_checkpoint(model_ckpt_path)
                model.eval()
                model.freeze()
                self.models[model_name] = model

            except ModuleNotFoundError:
                print(f"Module '{module_path}' not found!")
            except AttributeError:
                print(f"Class '{class_name}' not found in '{module_path}'!")

            except Exception as e:
                raise RuntimeError(
                    f"Failed to load model '{model_name}' from {model_ckpt_path}: \n {e}"
                )

        # TODO: Later discuss whether this threshold should be independent of metric threshold or not ?
        # if kwargs.get("threshold") is None:
        #     first_metric_key = next(iter(self.train_metrics))  # Get the first key
        #     first_metric = self.train_metrics[first_metric_key]  # Get the metric object
        #     self.threshold = int(first_metric.threshold)  # Access threshold
        # else:
        #     self.threshold = int(kwargs["threshold"])

    @classmethod
    def _validate_model_configs(cls, model_configs: Dict[str, Dict]):
        path_set = set()
        class_set = set()
        labels_set = set()

        sets_ = {"path": path_set, "class": class_set, "labels": labels_set}
        required_keys = {"class_path", "ckpt_path"}

        for model_name, config in model_configs.items():
            missing_keys = required_keys - config.keys()

            if missing_keys:
                raise AttributeError(
                    f"Missing keys {missing_keys} in model '{model_name}' configuration."
                )

            model_path = config["ckpt_path"]
            class_path = config["class_path"]

            # if model_path in path_set:
            #     raise ValueError(
            #         f"Duplicate model path detected: '{model_path}'. Each model must have a unique path."
            #     )

            # if class_path not in class_set:
            #     raise ValueError(
            #             f"Duplicate class path detected: '{class_path}'. Each model must have a unique path."
            #     )

            path_set.add(model_path)
            class_set.add(class_path)

            cls._extra_validation(model_name, config, sets_)

    @classmethod
    def _extra_validation(
        cls, model_name: str, config: Dict[str, Any], sets_: Dict[str, set]
    ):
        pass

    @abstractmethod
    def _get_prediction_and_labels(
        self, data: Dict[str, Any], labels: torch.Tensor, output: torch.Tensor
    ) -> (torch.Tensor, torch.Tensor):
        pass


class ChebiEnsemble(_EnsembleBase):

    NAME = "ChebiEnsemble"

    def __init__(self, model_configs: Dict[str, Dict], **kwargs):
        super().__init__(model_configs, **kwargs)

        # Add a dummy trainable parameter
        self.dummy_param = torch.nn.Parameter(torch.randn(1, requires_grad=True))
        self._num_models_per_label: Optional[torch.Tensor] = None
        self._generate_model_label_mask()

    @classmethod
    def _extra_validation(
        cls, model_name: str, config: Dict[str, Any], sets_: Dict[str, set]
    ):

        if "labels_path" not in config:
            raise AttributeError("Missing 'labels_path' key in config!")

        labels_path = config["labels_path"]
        # if labels_path not in sets_["labels"]:
        #     raise ValueError(
        #             f"Duplicate labels path detected: '{labels_path}'. Each model must have a unique path."
        #     )
        sets_["labels"].add(labels_path)

        if "TPV" not in config.keys() or "FPV" not in config.keys():
            raise AttributeError(
                f"Missing keys 'TPV' and/or 'FPV' in model '{model_name}' configuration."
            )

        # Validate 'tpv' and 'fpv' are either floats or convertible to float
        for key in ["TPV", "FPV"]:
            try:
                value = float(config[key])
                if value < 0:
                    raise ValueError(
                        f"'{key}' in model '{model_name}' must be non-negative, but got {value}."
                    )
            except (TypeError, ValueError):
                raise ValueError(
                    f"'{key}' in model '{model_name}' must be a float or convertible to float, but got {config[key]}."
                )

    def _generate_model_label_mask(self):
        labels_dict = {}
        num_models_per_label = torch.zeros(1, self.out_dim, device=self.device)
        for model_name, model_config in self.model_configs.items():
            labels_path = model_config["labels_path"]
            if not os.path.exists(labels_path):
                raise FileNotFoundError(f"Labels path '{labels_path}' does not exist.")

            with open(labels_path, "r") as f:
                labels_list = [int(line.strip()) for line in f]

            model_label_indices = []
            for label in labels_list:
                if label not in labels_dict:
                    labels_dict[label] = len(labels_dict)

                model_label_indices.append(labels_dict[label])

            # Create masks to apply predictions only to known classes
            mask = torch.zeros(self.out_dim, device=self.device, dtype=torch.bool)
            mask[
                torch.tensor(model_label_indices, dtype=torch.int, device=self.device)
            ] = True

            self.model_configs[model_name]["labels_mask"] = mask
            num_models_per_label += mask

        self._num_models_per_label = num_models_per_label

    def forward(self, data: Dict[str, Tensor], **kwargs: Any) -> Dict[str, Any]:
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
            ]  # Don't play a role here, just for lightning flow completeness

        return {
            "logits": total_logits,
            "pred_dict": predictions,
            "conf_dict": confidences,
        }

    def _get_prediction_and_labels(self, data, labels, model_output):
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

    def aggregate_predictions(self, predictions, confidences):
        """Implements weighted voting based on trustworthiness."""
        batch_size, num_classes = list(predictions.values())[0].shape
        true_scores = torch.zeros(batch_size, num_classes, device=self.device)
        false_scores = torch.zeros(batch_size, num_classes, device=self.device)

        for model, conf in confidences.items():
            tpv = float(self.model_configs[model]["TPV"])
            npv = float(self.model_configs[model]["FPV"])

            # Determine which classes the model provides predictions for
            mask = self.model_configs[model]["labels_mask"]
            weight = conf * (tpv * conf + npv * (1 - conf))

            # Apply mask: Only update scores for valid classes
            true_scores += weight * conf * mask
            false_scores += weight * (1 - conf) * mask

        # Avoid division by zero: Set valid_counts to 1 where it's zero
        valid_counts = self._num_models_per_label.clamp(min=1)

        # Normalize by valid contributions to prevent bias, this step can be optional depending upon scenario
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


class ChebiEnsembleLearning(_EnsembleBase):

    NAME = "ChebiEnsembleLearning"

    def __init__(self, model_configs: Dict[str, Dict], **kwargs):
        super().__init__(model_configs, **kwargs)

        from chebai.models.ffn import FFN

        ffn_kwargs = kwargs.copy()
        ffn_kwargs["input_size"] = len(self.model_configs) * int(kwargs["out_dim"])
        self.ffn: FFN = FFN(**ffn_kwargs)

    def forward(self, data: Dict[str, Tensor], **kwargs: Any) -> Dict[str, Any]:
        logits_list = [model(data)["logits"] for model in self.models.values()]
        return self.ffn({"features": torch.cat(logits_list, dim=1)})

    def _get_prediction_and_labels(
        self, data: Dict[str, Any], labels: torch.Tensor, output: torch.Tensor
    ) -> (torch.Tensor, torch.Tensor):
        return self.ffn._get_prediction_and_labels(data, labels, output)

    def _process_for_loss(
        self,
        model_output: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        loss_kwargs: Dict[str, Any],
    ) -> (torch.Tensor, torch.Tensor, Dict[str, Any]):
        return self.ffn._process_for_loss(model_output, labels, loss_kwargs)


if __name__ == "__main__":
    pass
