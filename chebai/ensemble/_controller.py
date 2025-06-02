from abc import ABC
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict

import torch
from torch import Tensor

from ._base import EnsembleBase
from ._constants import PRED_OP, WRAPPER_CLS_PATH
from ._utils import load_class
from ._wrappers import BaseWrapper


class _Controller(EnsembleBase, ABC):
    """
    Abstract base controller for ensemble models that handles data loading, collating,
    and inference logic over a collection of models.

    Inherits from:
        EnsembleBase: The base ensemble class with shared ensemble logic.
        ABC: For defining abstract methods.
    """

    def __init__(self, **kwargs: Any):
        """
        Initializes the controller with data loader and collator.

        Args:
            **kwargs (Any): Keyword arguments passed to the EnsembleBase initializer.
        """
        super().__init__(**kwargs)
        self._kwargs = kwargs
        # If an activation condition correponding model is added to queue, removed from this set
        # This is in order to avoid re-adding models that have already been processed
        self._model_key_set: set[str] = set(self._model_configs.keys())

        # Labels from any processed `data.pt` file of any reader
        self._collated_labels: torch.Tensor | None = None

    def _controller(
        self, model_name: str, model_input: list[str] | Path, **kwargs: Any
    ) -> Dict[str, Tensor]:
        """
        Performs inference with the model and extracts predictions and confidence values.

        Args:
            model (ChebaiBaseNet): The model to perform inference with.
            model_props (Dict[str, Tensor]): Dictionary with label mask and trust scores.

        Returns:
            Dict[str, Tensor]: Dictionary containing predictions and confidence scores.
        """
        wrapped_model = self._wrap_model(model_name)
        if self._operation_mode == PRED_OP:
            model_output, model_props = wrapped_model.predict(model_input)
        else:
            model_output, model_props = wrapped_model.evaluate(model_input)
            if (
                self._collated_labels is None
                and wrapped_model.collated_labels is not None
            ):
                self._collated_labels = wrapped_model.collated_labels

        del wrapped_model  # Model can be huge to keep it in memory, delete asap as no longer needed

        pred_conf_dict = self._get_pred_conf_from_model_output(
            model_output, model_props["mask"]
        )
        return {"pred_conf_dict": pred_conf_dict, "model_props": model_props}

    def _get_pred_conf_from_model_output(
        self, model_output: Dict[str, Tensor], model_label_mask: Tensor
    ) -> Dict[str, Tensor]:
        """
        Processes model output to extract binary predictions and confidence scores.

        Args:
            model_output (Dict[str, Tensor]): Dictionary containing logits from the model.
            model_label_mask (Tensor): A boolean mask indicating active labels for the model.

        Returns:
            Dict[str, Tensor]: Dictionary with keys "prediction" and "confidence" containing
                               tensors of the same shape as logits, filled only for active labels.
        """
        sigmoid_logits = torch.sigmoid(model_output["logits"])
        prediction = torch.full(
            (self._total_data_size, self._num_of_labels), -1, dtype=torch.bool
        )
        confidence = torch.full(
            (self._total_data_size, self._num_of_labels), -1, dtype=torch.float
        )
        prediction[:, model_label_mask] = sigmoid_logits > 0.5
        confidence[:, model_label_mask] = 2 * torch.abs(sigmoid_logits - 0.5)
        return {"prediction": prediction, "confidence": confidence}

    def _wrap_model(self, model_name: str) -> BaseWrapper:
        model_config = self._model_configs[model_name]
        wrp_cls = load_class(model_config[WRAPPER_CLS_PATH])
        assert issubclass(wrp_cls, BaseWrapper), ""
        wrapped_model = wrp_cls(
            model_name=model_name,
            model_config=model_config,
            dm_labels=self._dm_labels,
            **self._kwargs
        )

        assert isinstance(wrapped_model, BaseWrapper), ""
        return wrapped_model


class NoActivationCondition(_Controller):
    """
    A controller that queues and activates all models unconditionally.

    This implementation does not filter or select models dynamically.
    """

    def __init__(self, **kwargs: Any):
        """
        Initializes the controller and loads all model names into the processing queue.

        Args:
            **kwargs (Any): Keyword arguments passed to the _Controller initializer.
        """
        super().__init__(**kwargs)
        self._model_queue: Deque[str] = deque(list(self._model_configs.keys()))
