import os
import random
import unittest

import torch
from torchmetrics.classification import MultilabelF1Score

from chebai.callbacks.epoch_metrics import MacroF1


class TestCustomMacroF1Metric(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.device = "cuda" if torch.cuda.is_available() else "cpu"

    def test_all_predictions_are_1_half_labels_are_1(self) -> None:
        """
        Test custom metric against standard metric for the scenario where all predictions are 1
        but only half of the labels are 1.
        """
        preds = torch.ones((1, 900), dtype=torch.int)
        label = torch.ones((1, 900), dtype=torch.int)

        # Randomly set half of the labels to 0
        mask = [
            [True] * (label.size(1) // 2)
            + [False] * (label.size(1) - (label.size(1) // 2))
        ]
        random.shuffle(mask[0])
        label[torch.tensor(mask)] = 0

        # Get custom and standard metric scores
        macro_f1_custom_score, macro_f1_standard_score = (
            self.__get_custom_and_standard_metric_scores(label.shape[1], preds, label)
        )

        # preds = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        # label = torch.tensor([[1, 1, 0, 0, 1, 1, 0, 0, 1, 0]])
        # tps = [1, 1, 0, 0, 1, 1, 0, 0, 1, 0]
        # positive_predictions = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        # positive_labels = [1, 1, 0, 0, 1, 1, 0, 0, 1, 0]

        # ---------------------- For Standard F1 Macro Metric ---------------------
        # The metric is only proper defined when TP + FP ≠ 0 ∧ TP + FN ≠ 0
        # If this case is encountered for any class/label, the metric for that class/label
        # will be set to 0 and the overall metric may therefore be affected in turn.

        # precision = [1, 1, 0, 0, 1, 1, 0, 0, 1, 0]
        # recall    = [1, 1, 0, 0, 1, 1, 0, 0, 1, 0]
        # classwise_f1 = [2, 2, 0, 0, 2, 2, 0, 0, 2, 0] / [2, 2, 0, 0, 2, 2, 0, 0, 2, 0]
        #              = [1, 1, 0, 0, 1, 1, 0, 0, 1, 0]
        # mean = 5/10 = 0.5

        # ----------------------- For Custom F1 Metric ----------------------------
        # Perform masking as first step to take only class with positive labels
        # mask = [True, True, False, False, True, True, False, False, True, False]
        # precision = [1, 1, 1, 1, 1] / [1, 1, 1, 1, 1] = [1, 1, 1, 1, 1]
        # recall    = [1, 1, 1, 1, 1] / [1, 1, 1, 1, 1] = [1, 1, 1, 1, 1]
        # classwise_f1 = [2, 2, 2, 2, 2] / [2, 2, 2, 2, 2] = [1, 1, 1, 1, 1]
        # mean = 5/5 = 1  (because of masking we're averaging with across positive labels only)
        self.assertAlmostEqual(macro_f1_custom_score, 1, places=4)
        self.assertNotAlmostEqual(
            macro_f1_custom_score, macro_f1_standard_score, places=4
        )

    def test_all_labels_are_1_half_predictions_are_1(self) -> None:
        """
        Test custom metric against standard metric for the scenario where all labels are 1
        but only half of the predictions are 1.
        """
        preds = torch.ones((1, 900), dtype=torch.int)
        label = torch.ones((1, 900), dtype=torch.int)

        # Randomly set half of the predictions to 0
        mask = [
            [True] * (label.size(1) // 2)
            + [False] * (label.size(1) - (label.size(1) // 2))
        ]
        random.shuffle(mask[0])
        preds[torch.tensor(mask)] = 0

        # Get custom and standard metric scores
        macro_f1_custom_score, macro_f1_standard_score = (
            self.__get_custom_and_standard_metric_scores(label.shape[1], preds, label)
        )

        # As we are only taking positive labels for custom metric calculation via masking,
        # and since all labels are positive in this scenario, custom and std metric are same
        self.assertAlmostEqual(macro_f1_custom_score, macro_f1_standard_score, places=4)

    def test_iterative_vs_single_call_approach(self) -> None:
        """
        Test the custom metric implementation in update fashion approach against
        the single call approach.
        """
        preds = torch.tensor([[1, 1, 0, 1], [1, 0, 1, 1], [0, 1, 0, 1]])
        label = torch.tensor([[0, 0, 0, 0], [0, 0, 1, 1], [0, 1, 0, 1]])

        num_labels = label.shape[1]
        iterative_custom_metric = MacroF1(num_labels=num_labels)
        for i in range(label.shape[0]):
            iterative_custom_metric.update(preds[i].unsqueeze(0), label[i].unsqueeze(0))
        iterative_custom_metric_score = iterative_custom_metric.compute().item()

        single_call_custom_metric = MacroF1(num_labels=num_labels)
        single_call_custom_metric_score = single_call_custom_metric(preds, label).item()

        # Assert iterative and single call approaches give the same metric score
        self.assertEqual(iterative_custom_metric_score, single_call_custom_metric_score)

    def test_metric_against_realistic_data(self) -> None:
        """
        Test the custom metric against the standard on realistic data.
        """
        directory_path = os.path.join("tests", "test_data", "CheBIOver100_test")
        abs_path = os.path.join(os.getcwd(), directory_path)
        print(f"Checking data from - {abs_path}")
        num_of_files = len(os.listdir(abs_path)) // 2

        # Load single file to get the number of labels for metric class instantiation
        labels = torch.load(
            f"{directory_path}/labels{0:03d}.pt",
            map_location=torch.device(self.device),
            weights_only=False,
        )
        num_labels = labels.shape[1]
        macro_f1_custom = MacroF1(num_labels=num_labels)
        macro_f1_standard = MultilabelF1Score(num_labels=num_labels, average="macro")

        # Load each file in the directory and update the metrics
        for i in range(num_of_files):
            labels = torch.load(
                f"{directory_path}/labels{i:03d}.pt",
                map_location=torch.device(self.device),
                weights_only=False,
            )
            preds = torch.load(
                f"{directory_path}/preds{i:03d}.pt",
                map_location=torch.device(self.device),
                weights_only=False,
            )
            macro_f1_standard.update(preds, labels)
            macro_f1_custom.update(preds, labels)

        macro_f1_custom_score = macro_f1_custom.compute().item()
        macro_f1_standard_score = macro_f1_standard.compute().item()
        print(
            f"Realistic Data - Custom F1 score: {macro_f1_custom_score}, Std. F1 score: {macro_f1_standard_score}"
        )

        # Assert custom metric score is not equal to standard metric score
        self.assertNotAlmostEqual(
            macro_f1_custom_score, macro_f1_standard_score, places=4
        )

    def test_case_when_few_class_has_no_labels(self) -> None:
        """
        Test custom metric against standard metric for the scenario where some class has no labels.
        """
        preds = torch.tensor([[1, 1, 0, 1], [1, 0, 1, 1], [0, 1, 0, 1]])
        label = torch.tensor([[0, 0, 0, 0], [0, 0, 1, 1], [0, 1, 0, 1]])

        # Get custom and standard metric scores
        macro_f1_custom_score, macro_f1_standard_score = (
            self.__get_custom_and_standard_metric_scores(label.shape[1], preds, label)
        )

        # tps = [0, 1, 1, 2]
        # positive_predictions = [2, 2, 1, 3]
        # positive_labels = [0, 1, 1, 2]

        # ---------------------- For Standard F1 Macro Metric ---------------------
        # The metric is only proper defined when TP + FP ≠ 0 ∧ TP + FN ≠ 0
        # If this case is encountered for any class/label, the metric for that class/label
        # will be set to 0 and the overall metric may therefore be affected in turn.

        # precision = [0, 1, 1, 2] / [2, 2, 1, 3] = [0, 0.5, 1, 0.66666667]
        # recall    = [0, 1, 1, 2] / [0, 1, 1, 2] = [0, 1, 1, 1]
        # classwise_f1 = [0, 1, 2, 1.33333334]/[0, 1.5, 1, 1.66666667] = [0, 0.66666667, 1, 0.8]
        # mean = 2.47/4 = 0.6166666681

        # ----------------------- For Custom F1 Metric ----------------------------
        # Perform masking as first step to take only class with positive labels
        # mask = [False, True, True, True]
        # precision = [1, 1, 2] / [2, 1, 3] = [0.5, 1, 0.66666667]
        # recall    = [1, 1, 2] / [1, 1, 2] = [1, 1, 1]
        # classwise_f1 = [1, 2, 1.33334] / [1.5, 1, 1.67] = [0.66666667, 1, 0.8]
        # mean = 2.47/3 = 0.8222222241 (because of masking we averaging with across positive labels only)

        self.assertAlmostEqual(macro_f1_custom_score, 0.8222222241, places=4)
        self.assertNotAlmostEqual(
            macro_f1_custom_score, macro_f1_standard_score, places=4
        )

    @staticmethod
    def __get_custom_and_standard_metric_scores(
        num_labels: int, preds: torch.Tensor, labels: torch.Tensor
    ) -> tuple:
        """
        Helper method to calculate custom and standard macro F1 scores.

        Args:
            num_labels (int): Number of labels/classes.
            preds (torch.Tensor): Predicted tensor of shape (batch_size, num_labels).
            labels (torch.Tensor): True labels tensor of shape (batch_size, num_labels).

        Returns:
            tuple: A tuple containing two floats:
                - macro_f1_custom_score: Custom macro F1 score.
                - macro_f1_standard_score: Standard macro F1 score.
        """
        # Custom metric score
        macro_f1_custom = MacroF1(num_labels=num_labels)
        macro_f1_custom_score = macro_f1_custom(preds, labels).item()

        # Standard metric score
        macro_f1_standard = MultilabelF1Score(num_labels=num_labels, average="macro")
        macro_f1_standard_score = macro_f1_standard(preds, labels).item()

        return macro_f1_custom_score, macro_f1_standard_score


if __name__ == "__main__":
    unittest.main()
