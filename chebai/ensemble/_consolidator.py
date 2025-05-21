from abc import ABC
from typing import Any, Dict

from torch import Tensor

from ._base import EnsembleBase


class WeightedMajorityVoting(EnsembleBase, ABC):
    """
    Ensemble consolidator using weighted majority voting.
    Each model's contribution is weighted by a function of confidence,
    true positive value (TPV), and negative predictive value (NPV).
    """

    def _consolidator(
        self,
        pred_conf_dict: Dict[str, Tensor],
        model_props: Dict[str, Tensor],
        *,
        true_scores: Tensor,
        false_scores: Tensor,
        **kwargs: Any
    ) -> None:
        """
        Updates true/false scores based on model predictions using a weighted voting scheme.

        Args:
            pred_conf_dict (Dict[str, Tensor]): Contains model predictions and confidence scores.
            model_props (Dict[str, Tensor]): Contains mask, TPV and NPV tensors for model.
            true_scores (Tensor): Tensor accumulating weighted "true" contributions.
            false_scores (Tensor): Tensor accumulating weighted "false" contributions.
            **kwargs (Any): Additional unused keyword arguments.
        """
        tpv = model_props["tpv_tensor"]
        npv = model_props["fpv_tensor"]
        conf = pred_conf_dict["confidence"]
        mask = model_props["mask"]

        weight = conf * (tpv * conf + npv * (1 - conf))

        # Apply mask: Only update scores for valid classes
        true_scores += weight * conf * mask
        false_scores += weight * (1 - conf) * mask

    def _consolidate_on_finish(
        self, *, true_scores: Tensor, false_scores: Tensor
    ) -> Tensor:
        """
        Finalizes predictions after all models have contributed their scores.

        Args:
            true_scores (Tensor): Accumulated weighted true scores per label.
            false_scores (Tensor): Accumulated weighted false scores per label.

        Returns:
            Tensor: Final binary predictions (True if true_score > false_score).
        """
        # Avoid division by zero: Set valid_counts to 1 where it's zero
        valid_counts = self._num_models_per_label.clamp(min=1)
        # Normalize by valid contributions to prevent bias
        final_preds = (true_scores / valid_counts) > (false_scores / valid_counts)
        return final_preds


class MajorityVoting(EnsembleBase, ABC):
    """
    Ensemble consolidator using simple majority voting.
    Each model contributes equally; confidence is used directly as "vote weight".
    """

    def _consolidator(
        self,
        pred_conf_dict: Dict[str, Tensor],
        model_props: Dict[str, Tensor],
        *,
        true_scores: Tensor,
        false_scores: Tensor,
        **kwargs: Any
    ) -> None:
        """
        Updates true/false scores based on model predictions using unweighted voting.

        Args:
            pred_conf_dict (Dict[str, Tensor]): Contains model predictions and confidence scores.
            model_props (Dict[str, Tensor]): Contains mask tensor for model.
            true_scores (Tensor): Tensor accumulating true contributions.
            false_scores (Tensor): Tensor accumulating false contributions.
            **kwargs (Any): Additional unused keyword arguments.
        """
        conf = pred_conf_dict["confidence"]
        # Apply mask: Only update scores for valid classes
        mask = model_props["mask"]

        true_scores += conf * mask
        false_scores += (1 - conf) * mask

    def _consolidate_on_finish(
        self, *, true_scores: Tensor, false_scores: Tensor
    ) -> Tensor:
        """
        Finalizes predictions after all models have voted.

        Args:
            true_scores (Tensor): Accumulated true votes per label.
            false_scores (Tensor): Accumulated false votes per label.

        Returns:
            Tensor: Final binary predictions (True if true_score > false_score).
        """
        return true_scores > false_scores
