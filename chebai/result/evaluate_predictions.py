import torch
from jsonargparse import CLI
from torchmetrics.functional.classification import multilabel_auroc

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
        Loads predictions and labels, validates file correspondence, and calculates Multilabel AUROC.
        """
        test_preds, test_labels = load_results_from_buffer(self.eval_dir, self.device)
        self.validate_eval_dir(test_labels, test_preds)
        self.num_labels = test_preds.shape[1]

        ml_auroc = multilabel_auroc(
            test_preds, test_labels, num_labels=self.num_labels
        ).item()

        print("Multilabel AUC-ROC:", ml_auroc)


class Main:
    def evaluate(self, eval_dir: str):
        EvaluatePredictions(eval_dir).evaluate()


if __name__ == "__main__":
    CLI(Main)
