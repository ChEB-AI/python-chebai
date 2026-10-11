import warnings

import torch
import torchmetrics
from sklearn.metrics import roc_auc_score


def custom_reduce_fx(input: torch.Tensor) -> torch.Tensor:
    """
    Custom reduction function for distributed training.

    Args:
        input (torch.Tensor): The input tensor to be reduced.

    Returns:
        torch.Tensor: The reduced tensor.
    """
    print(f"called reduce (device: {input.device})")
    return torch.sum(input, dim=0)


class MacroF1(torchmetrics.Metric):
    """
    Computes the Macro F1 score, which is the unweighted mean of F1 scores for each class.
    This implementation differs from torchmetrics.classification.MultilabelF1Score in the behaviour for undefined
    values (i.e., classes where TP+FN=0). The torchmetrics implementation sets these classes to a default value.
    Here, the mean is only taken over classes which have at least one positive sample.

    Args:
        num_labels (int): Number of classes/labels.
        dist_sync_on_step (bool, optional): Synchronize metric state across processes at each forward
            before returning the value at the step. Default: False.
        threshold (float, optional): Threshold for converting predicted probabilities to binary (0, 1) predictions.
            Default: 0.5.
    """

    def __init__(
        self, num_labels: int, dist_sync_on_step: bool = False, threshold: float = 0.5
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state(
            "true_positives",
            default=torch.zeros(num_labels, dtype=torch.int),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "positive_predictions",
            default=torch.zeros(num_labels, dtype=torch.int),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "positive_labels",
            default=torch.zeros(num_labels, dtype=torch.int),
            dist_reduce_fx="sum",
        )
        self.threshold = threshold

    def update(self, preds: torch.Tensor, labels: torch.Tensor) -> None:
        """
        Update the state (TPs, Positive Predictions, Positive labels) with the current batch of predictions and labels.

        Args:
            preds (torch.Tensor): Predictions from the model.
            labels (torch.Tensor): Ground truth labels.
        """
        tps = torch.sum(
            torch.logical_and(preds > self.threshold, labels.to(torch.bool)),
            dim=0,
        )
        self.true_positives += tps
        self.positive_predictions += torch.sum(preds > self.threshold, dim=0)
        self.positive_labels += torch.sum(labels, dim=0)

    def compute(self) -> torch.Tensor:
        """
        Compute the Macro F1 score.

        Returns:
            torch.Tensor: The computed Macro F1 score.
        """

        # ignore classes without positive labels
        # classes with positive labels, but no positive predictions will get a precision of "nan" (0 divided by 0),
        # which is propagated to the classwise_f1 and then turned into 0
        mask = self.positive_labels != 0
        precision = self.true_positives[mask] / self.positive_predictions[mask]
        recall = self.true_positives[mask] / self.positive_labels[mask]
        classwise_f1 = 2 * precision * recall / (precision + recall)
        # if (precision and recall are 0) or (precision is nan), set f1 to 0
        classwise_f1 = classwise_f1.nan_to_num()
        return torch.mean(classwise_f1)


class BalancedAccuracy(torchmetrics.Metric):
    """
    Computes the Balanced Accuracy, which is the average of true positive rate (TPR) and true negative rate (TNR).
    Useful for imbalanced datasets.
    Balanced Accuracy = (TPR + TNR)/2 = (TP/(TP + FN) + (TN)/(TN + FP))/2

    Args:
        num_labels (int): Number of classes/labels.
        dist_sync_on_step (bool, optional): Synchronize metric state across processes at each forward
            before returning the value at the step. Default: False.
        threshold (float, optional): Threshold for converting predicted probabilities to binary (0, 1) predictions.
            Default: 0.5.
    """

    def __init__(
        self, num_labels: int, dist_sync_on_step: bool = False, threshold: float = 0.5
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state(
            "true_positives",
            default=torch.zeros(num_labels, dtype=torch.int),
            dist_reduce_fx="sum",
        )

        self.add_state(
            "false_positives",
            default=torch.zeros(num_labels, dtype=torch.int),
            dist_reduce_fx="sum",
        )

        self.add_state(
            "true_negatives",
            default=torch.zeros(num_labels, dtype=torch.int),
            dist_reduce_fx="sum",
        )

        self.add_state(
            "false_negatives",
            default=torch.zeros(num_labels, dtype=torch.int),
            dist_reduce_fx="sum",
        )

        self.threshold = threshold

    def update(self, preds: torch.Tensor, labels: torch.Tensor) -> None:
        """
        Update the state (TPs, TNs, FPs, FNs) with the current batch of predictions and labels.

        Args:
            preds (torch.Tensor): Predictions from the model.
            labels (torch.Tensor): Ground truth labels.
        """

        # Size: Batch_size x Num_of_Classes;
        # summing over 1st dimension (dim=0), gives us the True positives per class
        tps = torch.sum(
            torch.logical_and(preds > self.threshold, labels.to(torch.bool)), dim=0
        )
        fps = torch.sum(
            torch.logical_and(preds > self.threshold, ~labels.to(torch.bool)), dim=0
        )
        tns = torch.sum(
            torch.logical_and(preds <= self.threshold, ~labels.to(torch.bool)), dim=0
        )
        fns = torch.sum(
            torch.logical_and(preds <= self.threshold, labels.to(torch.bool)), dim=0
        )

        # Size: Num_of_Classes;
        self.true_positives += tps
        self.false_positives += fps
        self.true_negatives += tns
        self.false_negatives += fns

    def compute(self) -> torch.Tensor:
        """
        Compute the Balanced Accuracy.

        Returns:
            torch.Tensor: The computed Balanced Accuracy.
        """
        tpr = self.true_positives / (self.true_positives + self.false_negatives)
        tnr = self.true_negatives / (self.true_negatives + self.false_positives)
        # Convert the nan values to 0
        tpr = tpr.nan_to_num()
        tnr = tnr.nan_to_num()

        balanced_acc = (tpr + tnr) / 2
        return torch.mean(balanced_acc)


class HiMolMacroAUROC(torchmetrics.Metric):
    """
    Macro-averaged multilabel AUROC that exactly replicates HiMol's eval():
      - missing labels (== ignore_index) are excluded per-task before scoring
      - any task that, after exclusion, has only one class present is
        dropped from BOTH the sum and the divisor of the macro average
        (instead of being scored as 0.0 and diluting the mean, which is
        torchmetrics' default MultilabelAUROC behavior)

    Assumes preds/target are shape (N, num_labels), target values in {0, 1},
    with `ignore_index` marking missing/unlabeled entries.

    References:
        https://github.com/ZangXuan/HiMol/blob/ffdcb247b361a1f85ddb741862cff25e4a3b3341/finetune/optimization.py#L95-L104
    """

    full_state_update = False
    is_differentiable = False
    higher_is_better = True

    def __init__(self, num_labels: int, ignore_index: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.num_labels = num_labels
        self.ignore_index = ignore_index

        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        if preds.shape != target.shape:
            raise ValueError(
                f"preds/target shape mismatch: {preds.shape} vs {target.shape}"
            )
        if preds.ndim != 2 or preds.shape[1] != self.num_labels:
            raise ValueError(
                f"expected shape (N, {self.num_labels}), got {tuple(preds.shape)}"
            )

        self.preds.append(preds.detach().cpu())
        self.target.append(target.detach().cpu())

    def compute(self) -> torch.Tensor:
        preds = torch.cat(self.preds, dim=0).cpu().numpy()
        target = torch.cat(self.target, dim=0).cpu().numpy()

        roc_list = []
        n_skipped = 0

        for i in range(self.num_labels):
            col_target = target[:, i]
            col_preds = preds[:, i]

            valid = col_target != self.ignore_index
            col_target = col_target[valid]
            col_preds = col_preds[valid]

            # need at least one of each class to define AUC
            if (
                len(col_target) == 0
                or (col_target == 0).sum() == 0
                or (col_target == 1).sum() == 0
            ):
                n_skipped += 1
                continue

            roc_list.append(roc_auc_score(col_target, col_preds))

        if n_skipped > 0:
            warnings.warn(
                f"{n_skipped}/{self.num_labels} labels skipped (missing or single-class "
                f"after masking). Macro AUROC computed over {len(roc_list)} labels.",
                stacklevel=2,
            )

        if len(roc_list) == 0:
            return torch.tensor(float("nan"))

        return torch.tensor(sum(roc_list) / len(roc_list))
