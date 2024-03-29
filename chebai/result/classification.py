import os

from torchmetrics.classification import (
    MultilabelF1Score,
    MultilabelPrecision,
    MultilabelRecall,
)
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import tqdm

from chebai.callbacks.epoch_metrics import MacroF1
from chebai.models import ChebaiBaseNet
from chebai.preprocessing.datasets import XYBaseDataModule


def visualise_f1(logs_path):
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


def evaluate_model(
    model: ChebaiBaseNet,
    data_module: XYBaseDataModule,
    filename=None,
    buffer_dir=None,
    batch_size: int = 32,
):
    """Runs model on test set of data_module (or, if filename is not None, on data set found in that file).
    If buffer_dir is set, results will be saved in buffer_dir. Returns tensors with predictions and labels.
    """
    model.eval()
    collate = data_module.reader.COLLATER()

    data_list = data_module.load_processed_data("test", filename)
    data_list = data_list[: data_module.data_limit]
    preds_list = []
    labels_list = []
    if buffer_dir is not None:
        os.makedirs(buffer_dir, exist_ok=True)
    save_ind = 0
    save_batch_size = 4
    n_saved = 0

    for i in tqdm.tqdm(range(0, len(data_list), batch_size)):
        collated = collate(data_list[i : min(i + batch_size, len(data_list) - 1)])
        collated.x = collated.to_x(model.device)
        collated.y = collated.to_y(model.device)
        processable_data = model._process_batch(collated, 0)
        model_output = model(processable_data, **processable_data["model_kwargs"])
        preds, labels = model._get_prediction_and_labels(
            processable_data, processable_data["labels"], model_output
        )
        preds_list.append(preds)
        labels_list.append(labels)
        if buffer_dir is not None:
            n_saved += 1
            if n_saved >= save_batch_size:
                torch.save(
                    torch.cat(preds_list),
                    os.path.join(buffer_dir, f"preds{save_ind:03d}.pt"),
                )
                torch.save(
                    torch.cat(labels_list),
                    os.path.join(buffer_dir, f"labels{save_ind:03d}.pt"),
                )
                preds_list = []
                labels_list = []
                save_ind += 1
                n_saved = 0

    if buffer_dir is None:
        test_preds = torch.cat(preds_list)
        test_labels = torch.cat(labels_list)

        return test_preds, test_labels


def load_results_from_buffer(buffer_dir, device):
    """Load results stored in evaluate_model()"""
    preds_list = []
    labels_list = []

    i = 0
    filename = f"preds{i:03d}.pt"
    while os.path.isfile(os.path.join(buffer_dir, filename)):
        preds_list.append(
            torch.load(
                os.path.join(buffer_dir, filename),
                map_location=torch.device(device),
            )
        )
        i += 1
        filename = f"preds{i:03d}.pt"

    i = 0
    filename = f"labels{i:03d}.pt"
    while os.path.isfile(os.path.join(buffer_dir, filename)):
        labels_list.append(
            torch.load(
                os.path.join(buffer_dir, filename),
                map_location=torch.device(device),
            )
        )
        i += 1
        filename = f"labels{i:03d}.pt"

    test_preds = torch.cat(preds_list)
    test_labels = torch.cat(labels_list)

    return test_preds, test_labels


def print_metrics(preds, labels, device, classes=None, top_k=10, markdown_output=False):
    """Prints relevant metrics, including micro and macro F1, recall and precision, best k classes and worst classes."""
    f1_micro = MultilabelF1Score(preds.shape[1], average="micro").to(device=device)
    my_f1_macro = MacroF1(preds.shape[1]).to(device=device)

    print(f"Macro-F1: {my_f1_macro(preds, labels):3f}")
    print(f"Micro-F1: {f1_micro(preds, labels):3f}")
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
            f"| Model | Macro-F1 | Micro-F1 | Macro-Precision | Micro-Precision | Macro-Recall | Micro-Recall |"
        )
        print(f"| --- | --- | --- | --- | --- | --- | --- |")
        print(
            f"| | {my_f1_macro(preds, labels):3f} | {f1_micro(preds, labels):3f} | {precision_macro(preds, labels):3f} | "
            f"{precision_micro(preds, labels):3f} | {recall_macro(preds, labels):3f} | "
            f"{recall_micro(preds, labels):3f} |"
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
