from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from torch import Tensor
from torchmetrics.classification import (
    MultilabelF1Score,
    MultilabelPrecision,
    MultilabelRecall,
)

from chebai.callbacks.epoch_metrics import BalancedAccuracy, MacroF1
from chebai.result.utils import *


def visualise_f1(logs_path: str) -> None:
    """
    Visualize F1 scores from metrics.csv and save the plot as f1_plot.png.

    Args:
        logs_path: The path to the directory containing metrics.csv.
    """
    df = pd.read_csv(os.path.join(logs_path, "metrics.csv"))
    df_loss = df.melt(
        id_vars="epoch",
        value_vars=[
            "val_ep_macro-f1",
            "val_micro-f1",
            "train_micro-f1",
            "train_ep_macro-f1",
        ],
    )
    lineplt = sns.lineplot(df_loss, x="epoch", y="value", hue="variable")
    plt.savefig(os.path.join(logs_path, "f1_plot.png"))
    plt.show()


def print_metrics(
    preds: Tensor,
    labels: Tensor,
    device: torch.device,
    classes: Optional[List[str]] = None,
    top_k: int = 10,
    markdown_output: bool = False,
) -> None:
    """
    Prints relevant metrics, including micro and macro F1, recall and precision,
    best k classes, and worst classes.

    Args:
        preds: Predicted labels as a tensor.
        labels: True labels as a tensor.
        device: The device to perform computations on.
        classes: Optional list of class names.
        top_k: The number of top classes to display based on F1 score.
        markdown_output: If True, print metrics in markdown format.
    """
    f1_micro = MultilabelF1Score(preds.shape[1], average="micro").to(device=device)
    my_f1_macro = MacroF1(preds.shape[1]).to(device=device)
    my_bal_acc = BalancedAccuracy(preds.shape[1]).to(device=device)

    print(f"Macro-F1: {my_f1_macro(preds, labels):3f}")
    print(f"Micro-F1: {f1_micro(preds, labels):3f}")
    print(f"Balanced Accuracy: {my_bal_acc(preds, labels):3f}")
    precision_macro = MultilabelPrecision(preds.shape[1], average="macro").to(
        device=device
    )
    precision_micro = MultilabelPrecision(preds.shape[1], average="micro").to(
        device=device
    )
    macro_adjust = 1
    recall_macro = MultilabelRecall(preds.shape[1], average="macro").to(device=device)
    recall_micro = MultilabelRecall(preds.shape[1], average="micro").to(device=device)
    print(f"Macro-Precision: {precision_macro(preds, labels) * macro_adjust:3f}")
    print(f"Micro-Precision: {precision_micro(preds, labels):3f}")
    print(f"Macro-Recall: {recall_macro(preds, labels) * macro_adjust:3f}")
    print(f"Micro-Recall: {recall_micro(preds, labels):3f}")
    if markdown_output:
        print(
            f"| Model | Macro-F1 | Micro-F1 | Macro-Precision | Micro-Precision | Macro-Recall | Micro-Recall | Balanced Accuracy |"
        )
        print(f"| --- | --- | --- | --- | --- | --- | --- | --- |")
        print(
            f"| | {my_f1_macro(preds, labels):3f} | {f1_micro(preds, labels):3f} | {precision_macro(preds, labels):3f} | "
            f"{precision_micro(preds, labels):3f} | {recall_macro(preds, labels):3f} | "
            f"{recall_micro(preds, labels):3f} | {my_bal_acc(preds, labels):3f} |"
        )

    classwise_f1_fn = MultilabelF1Score(preds.shape[1], average=None).to(device=device)
    classwise_f1 = classwise_f1_fn(preds, labels)
    best_classwise_f1 = torch.topk(classwise_f1, top_k).indices
    print(f"Top {top_k} classes (F1-score):")
    for i, best in enumerate(best_classwise_f1):
        print(
            f"{i + 1}. {classes[best] if classes is not None else best} - F1: {classwise_f1[best]:3f}"
        )

    zeros = []
    for i, f1 in enumerate(classwise_f1):
        if f1 == 0.0 and torch.sum(labels[:, i]) != 0:
            zeros.append(f"{classes[i] if classes is not None else i}")
    print(
        f'Found {len(zeros)} classes with F1-score == 0 (and non-zero labels): {", ".join(zeros)}'
    )
