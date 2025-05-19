import os.path
from abc import ABC
from collections import deque
from typing import Any, Deque, Dict

import torch
from torch import Tensor

from chebai.models import ChebaiBaseNet
from chebai.preprocessing.collate import RaggedCollator

from .base import EnsembleBase


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
        self._collator = RaggedCollator()

        self._collated_data = self._load_and_collate_data()
        self.input_dim = len(self._collated_data.x[0])
        self._total_data_size: int = len(self._collated_data)

    def _load_and_collate_data(self) -> Any:
        """
        Loads and collates data using RaggedCollator.

        Returns:
            Collated data object with `.x` and `.y` attributes moved to device.
        """
        data = torch.load(
            os.path.join(self.data_processed_dir_main, self.reader_dir_name, "data.pt"),
            weights_only=False,
            map_location=self._device,
        )
        collated_data = self._collator(data)
        collated_data.x = collated_data.to_x(self._device)
        if collated_data.y is not None:
            collated_data.y = collated_data.to_y(self._device)
        return collated_data

    def _forward_pass(self, model: ChebaiBaseNet) -> Dict[str, Tensor]:
        """
        Runs a forward pass of the given model on the collated data.

        Args:
            model (ChebaiBaseNet): The model to perform inference with.

        Returns:
            Dict[str, Tensor]: Model output dictionary containing logits and other keys.
        """
        processable_data = model._process_batch(self._collated_data, 0)
        del processable_data["loss_kwargs"]
        model_output = model(processable_data, **processable_data["model_kwargs"])
        return model_output

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
        self._model_queue: Deque[str] = deque(list(self.model_configs.keys()))

    def _controller(
        self, model: ChebaiBaseNet, model_props: Dict[str, Tensor], **kwargs: Any
    ) -> Dict[str, Tensor]:
        """
        Performs inference with the model and extracts predictions and confidence values.

        Args:
            model (ChebaiBaseNet): The model to perform inference with.
            model_props (Dict[str, Tensor]): Dictionary with label mask and trust scores.

        Returns:
            Dict[str, Tensor]: Dictionary containing predictions and confidence scores.
        """
        model_output = self._forward_pass(model)
        return self._get_pred_conf_from_model_output(model_output, model_props["mask"])
