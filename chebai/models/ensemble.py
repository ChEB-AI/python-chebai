import os.path
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import Tensor

from chebai.custom_typehints import ModelConfig
from chebai.models import ChebaiBaseNet, Electra
from chebai.preprocessing.structures import XYData


class _EnsembleBase(ChebaiBaseNet, ABC):
    def __init__(self, model_configs: Dict[str, ModelConfig], **kwargs):
        super().__init__(**kwargs)

        self.models: Dict[str, ChebaiBaseNet] = {}
        self.model_configs: Dict[str, ModelConfig] = model_configs

        for model_name in self.model_configs:
            model_path = self.model_configs[model_name]["path"]
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"Model path '{model_path}' for '{model_name}' does not exist."
                )

            # Attempt to load the model to check validity
            try:
                self.models[model_name] = Electra.load_from_checkpoint(
                    model_path, map_location=self.device
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load model '{model_name}' from {model_path}: {e}"
                )

        for model in self.models.values():
            model.freeze()

        # TODO: Later discuss whether this threshold should be independent of metric threshold or not ?
        # if kwargs.get("threshold") is None:
        #     first_metric_key = next(iter(self.train_metrics))  # Get the first key
        #     first_metric = self.train_metrics[first_metric_key]  # Get the metric object
        #     self.threshold = int(first_metric.threshold)  # Access threshold
        # else:
        #     self.threshold = int(kwargs["threshold"])

    @abstractmethod
    def _get_prediction_and_labels(
        self, data: Dict[str, Any], labels: torch.Tensor, output: torch.Tensor
    ) -> (torch.Tensor, torch.Tensor):
        pass


class ChebiEnsemble(_EnsembleBase):

    NAME = "ChebiEnsemble"

    def __init__(self, model_configs: Dict[str, ModelConfig], **kwargs):
        self._validate_model_configs(model_configs)
        super().__init__(model_configs, **kwargs)
        # Add a dummy trainable parameter
        self.dummy_param = torch.nn.Parameter(torch.randn(1, requires_grad=True))

    @classmethod
    def _validate_model_configs(cls, model_configs: Dict[str, ModelConfig]):
        path_set = set()
        required_keys = {"path", "TPV", "FPV"}

        for model_name, config in model_configs.items():
            missing_keys = required_keys - config.keys()

            if missing_keys:
                raise AttributeError(
                    f"Missing keys {missing_keys} in model '{model_name}' configuration."
                )

            model_path = config["path"]

            # if model_path in path_set:
            #     raise ValueError(
            #         f"Duplicate model path detected: '{model_path}'. Each model must have a unique path."
            #     )

            path_set.add(model_path)

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

    def forward(self, data: Dict[str, Tensor], **kwargs: Any) -> Dict[str, Any]:
        predictions = {}
        confidences = {}
        total_logits = torch.zeros(
            data["labels"].shape[0], data["labels"].shape[1], device=self.device
        )

        for name, model in self.models.items():
            output = model(data)
            sigmoid_logits = torch.sigmoid(output["logits"])
            confidences[name] = sigmoid_logits
            predictions[name] = (sigmoid_logits > 0.5).long()
            total_logits += output["logits"]

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

        for model, preds in predictions.items():
            tpv = float(self.model_configs[model]["TPV"])
            npv = float(self.model_configs[model]["FPV"])
            weight = confidences[model] * (tpv * preds + npv * (1 - preds))

            true_scores += weight * preds
            false_scores += weight * (1 - preds)

        return (true_scores > false_scores).long()

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
