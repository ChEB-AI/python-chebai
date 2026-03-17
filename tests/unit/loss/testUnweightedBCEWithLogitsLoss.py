import unittest

import torch
import torch.nn.functional as F

from chebai.loss.bce_weighted import UnWeightedBCEWithLogitsLoss


class TestUnWeightedBCEWithLogitsLoss(unittest.TestCase):
    def test_missing_labels_are_excluded_from_mean_reduction(self) -> None:
        loss_fn = UnWeightedBCEWithLogitsLoss(reduction="mean")
        logits = torch.tensor([[0.0, 0.0]])
        target = torch.tensor([[0.0, 1.0]])
        missing_labels = torch.tensor([[True, False]])

        expected = F.binary_cross_entropy_with_logits(
            logits[:, 1], target[:, 1], reduction="mean"
        )
        actual = loss_fn(logits, target, missing_labels=missing_labels)

        self.assertTrue(
            torch.isclose(actual, expected),
            "Masked label positions must not contribute to BCE mean loss.",
        )

    def test_fully_masked_batch_returns_zero_loss(self) -> None:
        loss_fn = UnWeightedBCEWithLogitsLoss(reduction="mean")
        logits = torch.tensor([[1.0, -2.0]])
        target = torch.tensor([[1.0, 0.0]])
        missing_labels = torch.tensor([[True, True]])

        actual = loss_fn(logits, target, missing_labels=missing_labels)

        self.assertEqual(
            actual.item(),
            0.0,
            "A fully masked batch should produce a neutral zero loss.",
        )

    def test_without_missing_labels_matches_standard_bce(self) -> None:
        loss_fn = UnWeightedBCEWithLogitsLoss(reduction="mean")
        logits = torch.tensor([[0.5, -0.5], [1.0, -1.0]])
        target = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

        expected = F.binary_cross_entropy_with_logits(logits, target, reduction="mean")
        actual = loss_fn(logits, target)

        self.assertTrue(
            torch.isclose(actual, expected),
            "BCE behavior must remain unchanged when no missing_labels mask is provided.",
        )


if __name__ == "__main__":
    unittest.main()
