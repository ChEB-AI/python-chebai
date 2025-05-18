import os.path
from abc import ABC

import torch

from chebai.ensemble.base import EnsembleBase
from chebai.models import ChebaiBaseNet
from chebai.preprocessing.collate import RaggedCollator


class _Controller(EnsembleBase, ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._collator = RaggedCollator()

        self._collated_data = self._load_and_collate_data()
        self.total_data_size: int = len(self._collated_data)

    def _load_and_collate_data(self):
        data = torch.load(
            os.path.join(self.data_processed_dir_main, "data.pt"),
            weights_only=False,
            map_location=self.device,
        )
        collated_data = self._collator(data)
        collated_data.x = collated_data.to_x(self.device)
        if collated_data.y is not None:
            collated_data.y = collated_data.to_y(self.device)
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
            (self.total_data_size, self.num_of_labels), -1, dtype=torch.bool
        )
        confidence = torch.full(
            (self.total_data_size, self.num_of_labels), -1, dtype=torch.float
        )
        prediction[:, model_label_mask] = sigmoid_logits > 0.5
        confidence[:, model_label_mask] = 2 * torch.abs(sigmoid_logits - 0.5)
        return {"prediction": prediction, "confidence": confidence}


class NoActivationCondition(_Controller):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._model_queue = list(self.model_configs.keys())

    def _controller(self, model, model_props, **kwargs):
        model_output = self._forward_pass(model)
        return self._get_pred_conf_from_model_output(model_output, model_props["mask"])
