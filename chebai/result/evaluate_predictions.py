from typing import Tuple

import numpy as np
import torch
from jsonargparse import CLI
from torchmetrics.functional.classification import multilabel_auroc

from chebai.callbacks.epoch_metrics import MacroF1
from chebai.result.utils import load_results_from_buffer


class EvaluatePredictions:
    def __init__(self, eval_dir: str):
        """
        Initializes the EvaluatePredictions class.

        Args:
            eval_dir (str): Path to the directory containing evaluation files.
        """
        self.eval_dir = eval_dir
        self.metrics = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_labels = None

    @staticmethod
    def validate_eval_dir(label_files: torch.Tensor, pred_files: torch.Tensor) -> None:
        """
        Validates that the number of labels matches the number of predictions,
        ensuring that they have the same shape.

        Args:
            label_files (torch.Tensor): Tensor containing label data.
            pred_files (torch.Tensor): Tensor containing prediction data.

        Raises:
            ValueError: If label and prediction tensors are mismatched in shape.
        """
        if label_files is None or pred_files is None:
            raise ValueError("Both label and prediction tensors must be provided.")

        # Check if the number of labels matches the number of predictions
        if label_files.shape[0] != pred_files.shape[0]:
            raise ValueError(
                "Number of label tensors does not match the number of prediction tensors."
            )

        # Validate that the last dimension matches the expected number of classes
        if label_files.shape[1] != pred_files.shape[1]:
            raise ValueError(
                "Label and prediction tensors must have the same shape in terms of class outputs."
            )

    def evaluate(self) -> None:
        """
        Loads predictions and labels, validates file correspondence, and calculates Multilabel AUROC and Fmax.
        """
        test_preds, test_labels = load_results_from_buffer(self.eval_dir, self.device)
        self.validate_eval_dir(test_labels, test_preds)
        self.num_labels = test_preds.shape[1]

        ml_auroc = multilabel_auroc(
            test_preds, test_labels, num_labels=self.num_labels
        ).item()

        print("Multilabel AUC-ROC:", ml_auroc)

        fmax, threshold = self.calculate_fmax(test_preds, test_labels)
        print(f"F-max : {fmax}, threshold: {threshold}")

    def calculate_fmax(
        self, test_preds: torch.Tensor, test_labels: torch.Tensor
    ) -> Tuple[float, float]:
        """
        Calculates the Fmax metric using the F1 score at various thresholds.

        Args:
            test_preds (torch.Tensor): Predicted scores for the labels.
            test_labels (torch.Tensor): True labels for the evaluation.

        Returns:
            Tuple[float, float]: The maximum F1 score and the corresponding threshold.
        """
        # https://github.com/bio-ontology-research-group/deepgo2/blob/main/deepgo/metrics.py#L51-L52
        thresholds = np.linspace(0, 1, 101)
        fmax = 0.0
        best_threshold = 0.0

        for t in thresholds:
            custom_f1_metric = MacroF1(num_labels=self.num_labels, threshold=t)
            custom_f1_metric.update(test_preds, test_labels)
            custom_f1_metric_score = custom_f1_metric.compute().item()

            # Check if the current score is the best we've seen
            if custom_f1_metric_score > fmax:
                fmax = custom_f1_metric_score
                best_threshold = t

        return fmax, best_threshold


class Main:
    def evaluate(self, eval_dir: str):
        EvaluatePredictions(eval_dir).evaluate()


if __name__ == "__main__":
    # evaluate_predictions.py evaluate <path/to/file>
    CLI(Main)
