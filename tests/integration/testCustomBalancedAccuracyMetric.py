import os
import random
import unittest

import torch

from chebai.callbacks.epoch_metrics import BalancedAccuracy


class TestCustomBalancedAccuracyMetric(unittest.TestCase):
    """
    Unit tests for the Custom Balanced Accuracy metric.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """
        Set up class-level variables.
        """
        cls.device = "cuda" if torch.cuda.is_available() else "cpu"

    def test_iterative_vs_single_call_approach(self) -> None:
        """
        Test the custom metric implementation in update fashion approach against
        the single call approach.
        """
        preds = torch.tensor([[1, 1, 0, 1], [1, 0, 1, 1], [0, 1, 0, 1]])
        label = torch.tensor([[0, 0, 0, 0], [0, 0, 1, 1], [0, 1, 0, 1]])

        num_labels = label.shape[1]
        iterative_custom_metric = BalancedAccuracy(num_labels=num_labels)
        for i in range(label.shape[0]):
            iterative_custom_metric.update(preds[i].unsqueeze(0), label[i].unsqueeze(0))
        iterative_custom_metric_score = iterative_custom_metric.compute().item()

        single_call_custom_metric = BalancedAccuracy(num_labels=num_labels)
        single_call_custom_metric_score = single_call_custom_metric(preds, label).item()

        self.assertEqual(iterative_custom_metric_score, single_call_custom_metric_score)

    def test_metric_against_realistic_data(self) -> None:
        """
        Test the custom metric against the standard on realistic data.
        """
        directory_path = os.path.join("tests", "test_data", "CheBIOver100_test")
        abs_path = os.path.join(os.getcwd(), directory_path)
        print(f"Checking data from - {abs_path}")
        num_of_files = len(os.listdir(abs_path)) // 2

        # load single file to get the num of labels for metric class instantiation
        labels = torch.load(
            f"{directory_path}/labels{0:03d}.pt",
            map_location=torch.device(self.device),
            weights_only=False,
        )
        num_labels = labels.shape[1]
        balanced_acc_custom = BalancedAccuracy(num_labels=num_labels)

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
            balanced_acc_custom.update(preds, labels)

        balanced_acc_custom_score = balanced_acc_custom.compute().item()
        print(f"Balanced Accuracy for realistic data: {balanced_acc_custom_score}")

    def test_case_when_few_class_has_no_labels(self) -> None:
        """
        Test custom metric against standard metric for the scenario where some class has no labels.
        """
        preds = torch.tensor([[1, 1, 0, 1], [1, 0, 1, 1], [0, 1, 0, 1]])
        label = torch.tensor([[0, 0, 0, 0], [0, 0, 1, 1], [0, 1, 0, 1]])  # no labels

        # tp = [0, 1, 1, 2], fp = [2, 1, 0, 1], tn = [1, 1, 2, 0], fn = [0, 0, 0, 0]
        # tpr = [0, 1, 1, 2] / ([0, 1, 1, 2] + [0, 0, 0, 0]) = [0, 1, 1, 1]
        # tnr = [1, 1, 2, 0] / ([1, 1, 2, 0] + [2, 1, 0, 1]) = [0.33333, 0.5, 1, 0]
        # balanced_accuracy = ([0, 1, 1, 1] + [0.33333, 0.5, 1, 0]) / 2 = ([0.16666667, 0.75, 1, 0.5]
        # mean bal accuracy = 0.6041666666666666

        balanced_acc_score = self.__get_custom_metric_score(
            preds, label, label.shape[1]
        )

        self.assertAlmostEqual(balanced_acc_score, 0.6041666666, places=4)

    def test_all_predictions_are_1_half_labels_are_1(self) -> None:
        """
        Test custom metric against standard metric for the scenario where all predictions are 1 but only half of
        the labels are 1.
        """
        preds = torch.ones((1, 900), dtype=torch.int)
        label = torch.ones((1, 900), dtype=torch.int)

        mask = [
            [True] * (label.size(1) // 2)
            + [False] * (label.size(1) - (label.size(1) // 2))
        ]
        random.shuffle(mask[0])
        label[torch.tensor(mask)] = 0

        # preds = [1, 1, 1, 1], label = [0, 1, 0, 1]
        # tp = [0, 1, 0, 1], fp = [1, 0, 1, 0], tn = [0, 0, 0, 0], fn = [0, 0, 0, 0]
        # tpr = tp / (tp + fn) = [0, 1, 0, 1] / [0, 1, 0, 1] = [0, 1, 0, 1]
        # tnr = tn / (tn + fp) = [0, 0, 0, 0]
        # balanced accuracy = 1 / 4 = 0.25

        balanced_acc_custom_score = self.__get_custom_metric_score(
            preds, label, label.shape[1]
        )
        self.assertAlmostEqual(balanced_acc_custom_score, 0.25, places=4)

    def test_all_labels_are_1_half_predictions_are_1(self) -> None:
        """
        Test custom metric against standard metric for the scenario where all labels are 1 but only half of
        the predictions are 1.
        """
        preds = torch.ones((1, 900), dtype=torch.int)
        label = torch.ones((1, 900), dtype=torch.int)

        mask = [
            [True] * (label.size(1) // 2)
            + [False] * (label.size(1) - (label.size(1) // 2))
        ]
        random.shuffle(mask[0])
        preds[torch.tensor(mask)] = 0

        # label = [1, 1, 1, 1], pred = [0, 1, 0, 1]
        # tp = [0, 1, 0, 1], fp = [0, 1, 0, 1], tn = [0, 0, 0, 0], fn = [0, 0, 0, 0]
        # tpr = tp / (tp + fn) = [0, 1, 0, 1] / [0, 1, 0, 1] = [0, 1, 0, 1]
        # tnr = tn / (tn + fp) = [0, 0, 0, 0]
        # balanced accuracy = 1 / 4 = 0.25

        balanced_acc_custom_score = self.__get_custom_metric_score(
            preds, label, label.shape[1]
        )
        self.assertAlmostEqual(balanced_acc_custom_score, 0.25, places=4)

    @staticmethod
    def __get_custom_metric_score(
        preds: torch.Tensor, labels: torch.Tensor, num_labels: int
    ) -> float:
        """
        Helper function to compute the custom metric score.

        Args:
        - preds (torch.Tensor): Predictions tensor.
        - labels (torch.Tensor): Labels tensor.
        - num_labels (int): Number of labels/classes.

        Returns:
        - float: Computed custom metric score.
        """
        balanced_acc_custom = BalancedAccuracy(num_labels=num_labels)
        return balanced_acc_custom(preds, labels).item()


if __name__ == "__main__":
    unittest.main()
