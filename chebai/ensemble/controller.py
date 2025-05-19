import os.path
from abc import ABC
from collections import deque
from typing import Deque

import torch

from chebai.models import ChebaiBaseNet
from chebai.preprocessing.collate import RaggedCollator

from .base import EnsembleBase


class _Controller(EnsembleBase, ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._collator = RaggedCollator()

        self._collated_data = self._load_and_collate_data()
        self.input_dim = len(self._collated_data.x[0])
        self._total_data_size: int = len(self._collated_data)

    def _load_and_collate_data(self):
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

    def _forward_pass(self, model: ChebaiBaseNet):
        processable_data = model._process_batch(self._collated_data, 0)
        del processable_data["loss_kwargs"]
        model_output = model(processable_data, **processable_data["model_kwargs"])
        return model_output

    def _get_pred_conf_from_model_output(self, model_output, model_label_mask):
        # Consider logits and confidence only for valid classes
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
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._model_queue: Deque = deque(list(self.model_configs.keys()))

    def _controller(self, model, model_props, **kwargs):
        model_output = self._forward_pass(model)
        return self._get_pred_conf_from_model_output(model_output, model_props["mask"])
