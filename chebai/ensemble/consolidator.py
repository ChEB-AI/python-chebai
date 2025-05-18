from abc import ABC

from .base import EnsembleBase


class WeightedMajorityVoting(EnsembleBase, ABC):
    def _consolidator(
        self, pred_conf_dict, model_props, *, true_scores, false_scores, **kwargs
    ):
        tpv = model_props["tpv_tensor"]
        npv = model_props["fpv_tensor"]
        conf = pred_conf_dict["confidence"]

        # Determine which classes the model provides predictions for
        mask = model_props["mask"]
        weight = conf * (tpv * conf + npv * (1 - conf))

        # Apply mask: Only update scores for valid classes
        true_scores += weight * conf * mask
        false_scores += weight * (1 - conf) * mask

    def _consolidate_on_finish(self, *, true_scores, false_scores):
        # Avoid division by zero: Set valid_counts to 1 where it's zero
        valid_counts = self._num_models_per_label.clamp(min=1)

        # Normalize by valid contributions to prevent bias
        final_preds = (true_scores / valid_counts) > (false_scores / valid_counts)
        return final_preds


class MajorityVoting(EnsembleBase, ABC):
    def _consolidator(
        self, pred_conf_dict, model_props, *, true_scores, false_scores, **kwargs
    ):
        conf = pred_conf_dict["confidence"]

        # Determine which classes the model provides predictions for
        mask = model_props["mask"]
        # Apply mask: Only update scores for valid classes
        true_scores += conf * mask
        false_scores += (1 - conf) * mask

    def _consolidate_on_finish(self, *, true_scores, false_scores):
        return true_scores > false_scores
