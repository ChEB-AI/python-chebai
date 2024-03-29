import torch
import torchmetrics


def custom_reduce_fx(input):
    print(f"called reduce (device: {input.device})")
    return torch.sum(input, dim=0)


class MacroF1(torchmetrics.Metric):
    def __init__(self, num_labels, dist_sync_on_step=False, threshold=0.5):
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

    def update(self, preds: torch.Tensor, labels: torch.Tensor):
        tps = torch.sum(
            torch.logical_and(preds > self.threshold, labels.to(torch.bool)), dim=0
        )
        self.true_positives += tps
        self.positive_predictions += torch.sum(preds > self.threshold, dim=0)
        self.positive_labels += torch.sum(labels, dim=0)

    def compute(self):
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
