import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, Optional, Union

import torch
from lightning.pytorch.core.module import LightningModule

from chebai.preprocessing.structures import XYData

logging.getLogger("pysmiles").setLevel(logging.CRITICAL)

_MODEL_REGISTRY = dict()


class ChebaiBaseNet(LightningModule, ABC):
    """
    Base class for Chebai neural network models inheriting from PyTorch Lightning's LightningModule.

    Args:
        criterion (torch.nn.Module, optional): The loss criterion for the model. Defaults to None.
        out_dim (int, optional): The output dimension of the model. Defaults to None.
        train_metrics (torch.nn.Module, optional): The metrics to be used during training. Defaults to None.
        val_metrics (torch.nn.Module, optional): The metrics to be used during validation. Defaults to None.
        test_metrics (torch.nn.Module, optional): The metrics to be used during testing. Defaults to None.
        pass_loss_kwargs (bool, optional): Whether to pass loss kwargs to the criterion. Defaults to True.
        optimizer_kwargs (Dict[str, Any], optional): Additional keyword arguments for the optimizer. Defaults to None.
        **kwargs: Additional keyword arguments.

    Attributes:
        NAME (str): The name of the model.
    """

    NAME = None

    def __init__(
        self,
        criterion: torch.nn.Module = None,
        out_dim: Optional[int] = None,
        input_dim: Optional[int] = None,
        train_metrics: Optional[torch.nn.Module] = None,
        val_metrics: Optional[torch.nn.Module] = None,
        test_metrics: Optional[torch.nn.Module] = None,
        pass_loss_kwargs: bool = True,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        exclude_hyperparameter_logging: Optional[Iterable[str]] = None,
        **kwargs,
    ):
        super().__init__()
        if exclude_hyperparameter_logging is None:
            exclude_hyperparameter_logging = tuple()
        self.criterion = criterion
        self.save_hyperparameters(
            ignore=[
                "criterion",
                "train_metrics",
                "val_metrics",
                "test_metrics",
                *exclude_hyperparameter_logging,
            ]
        )

        self.out_dim = out_dim
        self.input_dim = input_dim
        assert out_dim is not None, "out_dim must be specified"
        assert input_dim is not None, "input_dim must be specified"

        if optimizer_kwargs:
            self.optimizer_kwargs = optimizer_kwargs
        else:
            self.optimizer_kwargs = dict()
        self.train_metrics = train_metrics
        self.validation_metrics = val_metrics
        self.test_metrics = test_metrics
        self.pass_loss_kwargs = pass_loss_kwargs

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        # avoid errors due to unexpected keys (e.g., if loading checkpoint from a bce model and using it with a
        # different loss)
        if "criterion.base_loss.pos_weight" in checkpoint["state_dict"]:
            del checkpoint["state_dict"]["criterion.base_loss.pos_weight"]
        if "criterion.pos_weight" in checkpoint["state_dict"]:
            del checkpoint["state_dict"]["criterion.pos_weight"]

    def __init_subclass__(cls, **kwargs):
        """
        Automatically registers subclasses in the model registry to prevent duplicates.

        Args:
            **kwargs: Additional keyword arguments.
        """
        if cls.NAME in _MODEL_REGISTRY:
            raise ValueError(f"Model {cls.NAME} does already exist")
        else:
            _MODEL_REGISTRY[cls.NAME] = cls

    def _get_prediction_and_labels(
        self, data: Dict[str, Any], labels: torch.Tensor, output: torch.Tensor
    ) -> (torch.Tensor, torch.Tensor):
        """
        Gets the predictions and labels from the model output.

        Args:
            data (Dict[str, Any]): The processed batch data.
            labels (torch.Tensor): The true labels.
            output (torch.Tensor): The model output.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Predictions and labels.
        """
        return output, labels

    def _process_labels_in_batch(self, batch: XYData) -> torch.Tensor:
        """
        Processes the labels in the batch.

        Args:
            batch (XYData): The input batch of data.

        Returns:
            torch.Tensor: The processed labels.
        """
        return batch.y.float()

    def _process_batch(self, batch: XYData, batch_idx: int) -> Dict[str, Any]:
        """
        Processes the batch data.

        Args:
            batch (XYData): The input batch of data.
            batch_idx (int): The index of the current batch.

        Returns:
            Dict[str, Any]: Processed batch data.
        """
        return dict(
            features=batch.x,
            labels=self._process_labels_in_batch(batch),
            model_kwargs=batch.additional_fields["model_kwargs"],
            loss_kwargs=batch.additional_fields["loss_kwargs"],
            idents=batch.additional_fields["idents"],
        )

    def _process_for_loss(
        self,
        model_output: torch.Tensor,
        labels: torch.Tensor,
        loss_kwargs: Dict[str, Any],
    ) -> (torch.Tensor, torch.Tensor, Dict[str, Any]):
        """
        Processes the data for loss computation.

        Args:
            model_output (torch.Tensor): The model output.
            labels (torch.Tensor): The true labels.
            loss_kwargs (Dict[str, Any]): Additional keyword arguments for the loss function.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]: Model output, labels, and loss kwargs.
        """
        return model_output, labels, loss_kwargs

    def training_step(
        self, batch: XYData, batch_idx: int
    ) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        Defines the training step.

        Args:
            batch (XYData): The input batch of data.
            batch_idx (int): The index of the current batch.

        Returns:
            Dict[str, Union[torch.Tensor, Any]]: The result of the training step.
        """
        return self._execute(
            batch, batch_idx, self.train_metrics, prefix="train_", sync_dist=True
        )

    def validation_step(
        self, batch: XYData, batch_idx: int
    ) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        Defines the validation step.

        Args:
            batch (XYData): The input batch of data.
            batch_idx (int): The index of the current batch.

        Returns:
            Dict[str, Union[torch.Tensor, Any]]: The result of the validation step.
        """
        return self._execute(
            batch, batch_idx, self.validation_metrics, prefix="val_", sync_dist=True
        )

    def test_step(
        self, batch: XYData, batch_idx: int
    ) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        Defines the test step.

        Args:
            batch (XYData): The input batch of data.
            batch_idx (int): The index of the current batch.

        Returns:
            Dict[str, Union[torch.Tensor, Any]]: The result of the test step.
        """
        return self._execute(
            batch, batch_idx, self.test_metrics, prefix="test_", sync_dist=True
        )

    def predict_step(
        self, batch: XYData, batch_idx: int, **kwargs
    ) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        Defines the prediction step.

        Args:
            batch (XYData): The input batch of data.
            batch_idx (int): The index of the current batch.
            **kwargs: Additional keyword arguments.

        Returns:
            Dict[str, Union[torch.Tensor, Any]]: The result of the prediction step.
        """
        return self._execute(batch, batch_idx, self.test_metrics, prefix="", log=False)

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
        pr, tar = self._get_prediction_and_labels(data, labels, model_output)
        d = dict(data=data, labels=labels, output=model_output, preds=pr)
        if log:
            if self.criterion is not None:
                loss_data, loss_labels, loss_kwargs_candidates = self._process_for_loss(
                    model_output, labels, data.get("loss_kwargs", dict())
                )
                loss_kwargs = dict()
                if self.pass_loss_kwargs:
                    loss_kwargs = loss_kwargs_candidates
                loss_kwargs["current_epoch"] = self.trainer.current_epoch
                loss = self.criterion(loss_data, loss_labels, **loss_kwargs)
                if isinstance(loss, tuple):
                    unnamed_loss_index = 1
                    if isinstance(loss[1], dict):
                        unnamed_loss_index = 2
                        for key, value in loss[1].items():
                            self.log(
                                key,
                                value if isinstance(value, int) else value.item(),
                                batch_size=len(batch),
                                on_step=True,
                                on_epoch=True,
                                prog_bar=False,
                                logger=True,
                                sync_dist=sync_dist,
                            )
                    loss_additional = loss[unnamed_loss_index:]
                    for i, loss_add in enumerate(loss_additional):
                        self.log(
                            f"{prefix}loss_{i}",
                            loss_add if isinstance(loss_add, int) else loss_add.item(),
                            batch_size=len(batch),
                            on_step=True,
                            on_epoch=True,
                            prog_bar=False,
                            logger=True,
                            sync_dist=sync_dist,
                        )
                    loss = loss[0]

                d["loss"] = loss
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
                    metric.update(pr, tar)
                self._log_metrics(prefix, metrics, len(batch))
        return d

    def _log_metrics(self, prefix: str, metrics: torch.nn.Module, batch_size: int):
        """
        Logs the metrics for the given prefix.

        Args:
            prefix (str): The prefix to be added to the metric names.
            metrics (torch.nn.Module): A dictionary containing the metrics to be logged.
            batch_size (int): The batch size used for logging.

        Returns:
            None
        """
        # don't use sync_dist=True if the metric is a torchmetrics-metric
        # (see https://github.com/Lightning-AI/pytorch-lightning/discussions/6501#discussioncomment-569757)
        for metric_name, metric in metrics.items():
            m = None  # m = metric.compute()
            if isinstance(m, dict):
                # todo: is this case needed? it requires logging values directly which does not give accurate results
                # with the current metric-setup
                for k, m2 in m.items():
                    self.log(
                        f"{prefix}{metric_name}{k}",
                        m2,
                        batch_size=batch_size,
                        on_step=False,
                        on_epoch=True,
                        prog_bar=True,
                        logger=True,
                    )
            else:
                self.log(
                    f"{prefix}{metric_name}",
                    metric,
                    batch_size=batch_size,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                )

    @abstractmethod
    def forward(self, x: Dict[str, Any]) -> torch.Tensor:
        """
        Defines the forward pass.

        Args:
            x (Dict[str, Any]): The input data.

        Returns:
            torch.Tensor: The model output.
        """
        pass

    def configure_optimizers(self, **kwargs) -> torch.optim.Optimizer:
        """
        Configures the optimizers.

        Args:
            **kwargs: Additional keyword arguments.

        Returns:
            torch.optim.Optimizer: The optimizer.
        """
        return torch.optim.Adamax(self.parameters(), **self.optimizer_kwargs)
