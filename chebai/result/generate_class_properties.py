import json
from pathlib import Path
from typing import Literal

import torchmetrics
from jsonargparse import CLI
from torchmetrics.classification import MultilabelConfusionMatrix, MultilabelF1Score

from chebai.preprocessing.datasets.base import XYBaseDataModule
from chebai.result.utils import (
    load_data_instance,
    load_model_for_inference,
    parse_config_file,
)


class ClassesPropertiesGenerator:
    """
    Computes PPV (Positive Predictive Value) and NPV (Negative Predictive Value) and counts the number of
    true positives (TP), false positives (FP), true negatives (TN), and false negatives (FN)
    for each class in a multi-label classification problem using a PyTorch Lightning model.
    """

    @staticmethod
    def load_class_labels(path: Path) -> list[str]:
        """
        Load a list of class names from a .json or .txt file.

        Args:
            path: Path to the class labels file (txt or json).

        Returns:
            A list of class names, one per line.
        """
        path = Path(path)
        with path.open() as f:
            return [line.strip() for line in f if line.strip()]

    @staticmethod
    def compute_classwise_scores(
        metrics_obj_dict: dict[str, torchmetrics.Metric],
        class_names: list[str],
    ) -> dict[str, dict[str, float]]:
        """
        Compute per-class evaluation metrics for a multi-label classification task.

        This method uses torchmetrics objects (MultilabelConfusionMatrix, F1 scores, etc.)
        to compute the following metrics for each class:
            - PPV (Positive Predictive Value or Precision)
            - NPV (Negative Predictive Value)
            - True Positives (TP)
            - False Positives (FP)
            - True Negatives (TN)
            - False Negatives (FN)
            - F1 score

        Args:
            metrics_obj_dict: Dictionary containing pre-updated torchmetrics.Metric objects:
                {
                    "cm": MultilabelConfusionMatrix,
                    "micro-f1": MultilabelF1Score (average=None)
                }
            class_names: List of class names in the same order as class indices.

        Returns:
            Dictionary mapping each class name to a sub-dictionary of computed metrics:
            {
                "class_name_1": {
                    "PPV": float,
                    "NPV": float,
                    "TN": int,
                    "FP": int,
                    "FN": int,
                    "TP": int,
                    "f1": float,
                },
                ...
            }
        """
        cm_tensor = metrics_obj_dict["cm"].compute()  # Shape: (num_classes, 2, 2)
        f1_tensor = metrics_obj_dict["f1"].compute()  # shape: (num_classes,)

        assert len(class_names) == cm_tensor.shape[0] == f1_tensor.shape[0], (
            f"Mismatch between number of class names ({len(class_names)}) and metric tensor sizes: "
            f"confusion matrix has {cm_tensor.shape[0]}, "
            f"F1 has {f1_tensor.shape[0]}, "
        )

        results: dict[str, dict[str, float]] = {}
        for idx, cls_name in enumerate(class_names):
            tn = cm_tensor[idx][0][0].item()
            fp = cm_tensor[idx][0][1].item()
            fn = cm_tensor[idx][1][0].item()
            tp = cm_tensor[idx][1][1].item()

            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0  # Precision
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0  # Negative predictive value

            # positive_raw = [p.item() for i, p in enumerate(raw_preds[:, idx]) if true_np[i, idx]]
            # negative_raw = [p.item() for i, p in enumerate(raw_preds[:, idx]) if not true_np[i, idx]]

            f1 = f1_tensor[idx]

            results[cls_name] = {
                "PPV": round(ppv, 4),
                "NPV": round(npv, 4),
                "TN": int(tn),
                "FP": int(fp),
                "FN": int(fn),
                "TP": int(tp),
                "f1": round(f1.item(), 4),
                # "positive_preds": positive_raw,
                # "negative_preds": negative_raw,
            }
        return results

    def generate_props(
        self,
        data_partition: Literal["train", "val", "test"],
        model_ckpt_path: str,
        model_config_file_path: str,
        data_config_file_path: str,
        output_path: str | None = None,
        apply_id_filter: str | None = None,
    ) -> None:
        """
        Run inference on validation set, compute TPV/NPV  per class, and save to JSON.

        Args:
            data_partition: Partition of the dataset to use to generate class properties.
            model_ckpt_path: Path to the PyTorch Lightning checkpoint file.
            model_config_file_path: Path to yaml config file of the model.
            data_config_file_path: Path to yaml config file of the data.
            output_path: Optional path where to write the JSON metrics file.
                         Defaults to '<processed_dir_main>/classes.json'.
            apply_id_filter: Optional path to a (data.pt) file containing IDs to filter the dataset. This is useful for comparing datasets with different ids.
        """
        data_cls_path, data_cls_kwargs = parse_config_file(data_config_file_path)
        data_module: XYBaseDataModule = load_data_instance(
            data_cls_path, data_cls_kwargs
        )
        data_module.apply_id_filter = apply_id_filter

        splits_file_path = Path(data_module.processed_dir_main, "splits.csv")
        if data_module.splits_file_path is None:
            if not splits_file_path.exists():
                raise RuntimeError(
                    "Either the data module should be initialized with a `splits_file_path`, "
                    f"or the file `{splits_file_path}` must exists.\n"
                    "This is to prevent the data module from dynamically generating the splits."
                )

            print(
                f"`splits_file_path` is not provided as an initialization parameter to the data module\n"
                f"Using splits from the file {splits_file_path}"
            )
            data_module.splits_file_path = splits_file_path

        model_class_path, model_kwargs = parse_config_file(model_config_file_path)
        model = load_model_for_inference(
            model_ckpt_path, model_class_path, model_kwargs
        )

        if data_partition == "train":
            data_loader = data_module.train_dataloader()
        elif data_partition == "val":
            data_loader = data_module.val_dataloader()
        elif data_partition == "test":
            data_loader = data_module.test_dataloader()
        else:
            raise ValueError(f"Unknown data partition: {data_partition}")
        print(f"Running inference on {data_partition} data...")

        if data_module.apply_label_filter is not None:
            classes_file = data_module.apply_label_filter
        else:
            classes_file = Path(data_module.processed_dir_main) / "classes.txt"
        class_names = self.load_class_labels(classes_file)
        num_classes = len(class_names)
        metrics_obj_dict: dict[str, torchmetrics.Metric] = {
            "cm": MultilabelConfusionMatrix(num_labels=num_classes).to(
                device=model.device
            ),
            "f1": MultilabelF1Score(num_labels=num_classes, average=None).to(
                device=model.device
            ),
        }

        for batch_idx, batch in enumerate(data_loader):
            batch = batch.to(device=model.device)
            data = model._process_batch(batch, batch_idx=batch_idx)
            labels = data["labels"].to(device=model.device)
            data["features"][0].to(device=model.device)
            model_output = model(data, **data.get("model_kwargs", {}))
            preds, targets = model._get_prediction_and_labels(
                data, labels, model_output
            )
            for metric_obj in metrics_obj_dict.values():
                metric_obj.update(preds, targets)

        print("Computing metrics...")
        if output_path is None:
            output_file = (
                Path(data_module.processed_dir_main) / f"classes_{data_partition}.json"
            )
        else:
            output_file = Path(output_path)

        metrics = self.compute_classwise_scores(metrics_obj_dict, class_names)

        with output_file.open("w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved metrics to {output_file}")


class Main:
    """
    CLI wrapper for ClassesPropertiesGenerator.
    """

    def generate(
        self,
        data_partition: Literal["train", "val", "test"],
        model_ckpt_path: str,
        model_config_file_path: str,
        data_config_file_path: str,
        output_path: str | None = None,
        apply_id_filter: str | None = None,
    ) -> None:
        """
        CLI command to generate JSON with metrics on validation set.

        Args:
            data_partition: Partition of dataset to use to generate class properties.
            model_ckpt_path: Path to the PyTorch Lightning checkpoint file.
            model_config_file_path: Path to yaml config file of the model.
            data_config_file_path: Path to yaml config file of the data.
            output_path: Optional path where to write the JSON metrics file.
                         Defaults to '<processed_dir_main>/classes.json'.
        """
        assert data_partition in [
            "train",
            "val",
            "test",
        ], (
            f"Given data partition invalid: {data_partition}, Choose one of the value among `train`, `val`, `test` "
        )
        generator = ClassesPropertiesGenerator()
        generator.generate_props(
            data_partition,
            model_ckpt_path,
            model_config_file_path,
            data_config_file_path,
            output_path,
            apply_id_filter=apply_id_filter,
        )


if __name__ == "__main__":
    # Usage:
    # generate_classes_properties.py generate \
    # --data_partition "val" \
    # --model_ckpt_path "model/ckpt/path" \
    # --model_config_file_path "model/config/file/path" \
    # --data_config_file_path "data/config/file/path" \
    # --output_path "output/file/path" # Optional
    CLI(Main, as_positional=False)
