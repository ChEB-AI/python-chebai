import torch
from torch import Tensor
from torchmetrics.regression import MeanSquaredError

# from chebai.callbacks.epoch_metrics import BalancedAccuracy, MacroF1
# from chebai.result.utils import *

# def visualise_f1(logs_path: str) -> None:
#     """
#     Visualize F1 scores from metrics.csv and save the plot as f1_plot.png.

#     Args:
#         logs_path: The path to the directory containing metrics.csv.
#     """
#     df = pd.read_csv(os.path.join(logs_path, "metrics.csv"))
#     df_loss = df.melt(
#         id_vars="epoch",
#         value_vars=[
#             "val_ep_macro-f1",
#             "val_micro-f1",
#             "train_micro-f1",
#             "train_ep_macro-f1",
#         ],
#     )
#     lineplt = sns.lineplot(df_loss, x="epoch", y="value", hue="variable")
#     plt.savefig(os.path.join(logs_path, "f1_plot.png"))
#     plt.show()


def metrics_regression(
    preds: Tensor,
    labels: Tensor,
    device: torch.device,
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
    mse = MeanSquaredError()
    mse = mse.to(labels.device)

    rmse = MeanSquaredError(squared=False)
    rmse = rmse.to(labels.device)

    return (mse(preds, labels), rmse(preds, labels))

    # print(f"Micro-F1: {f1_micro(preds, labels):3f}")
    # print(f"Balanced Accuracy: {my_bal_acc(preds, labels):3f}")

    # if markdown_output:
    #     print(
    #         f"| Model | MSE | RMSE | Macro-Precision | Micro-Precision | Macro-Recall | Micro-Recall | Balanced Accuracy"
    #     )
    #     print(f"| --- | --- | --- | --- | --- | --- | --- | --- |")
    #     print(
    #         f"| Elektra | {my_f1_macro(preds, labels):3f} | {f1_micro(preds, labels):3f} | {precision_macro(preds, labels):3f} | "
    #         f"{precision_micro(preds, labels):3f} | {recall_macro(preds, labels):3f} | "
    #         f"{recall_micro(preds, labels):3f} | {my_bal_acc(preds, labels):3f} |"
    #     )
