import json
from pathlib import Path

import torch
from jsonargparse import CLI
from sklearn.metrics import multilabel_confusion_matrix

from chebai.preprocessing.datasets.base import XYBaseDataModule

from .._utils import load_data_instance, load_model_for_inference, parse_config_file


class ClassesPropertiesGenerator:
    """
    Computes PPV (Positive Predictive Value) and NPV (Negative Predictive Value)
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
    def compute_tpv_npv(
        y_true: list[torch.Tensor],
        y_pred: list[torch.Tensor],
        class_names: list[str],
    ) -> dict[str, dict[str, float]]:
        """
        Compute TPV (precision) and NPV for each class in a multi-label setting.

        Args:
            y_true: List of binary ground-truth label tensors, one tensor per sample.
            y_pred: List of binary prediction tensors, one tensor per sample.
            class_names: Ordered list of class names corresponding to class indices.

        Returns:
            Dictionary mapping each class name to its TPV and NPV metrics:
            {
                "class_name": {"PPV": float, "NPV": float},
                ...
            }
        """
        # Stack per-sample tensors into (n_samples, n_classes) numpy arrays
        true_np = torch.stack(y_true).cpu().numpy().astype(int)
        pred_np = torch.stack(y_pred).cpu().numpy().astype(int)

        # Compute confusion matrix for each class
        cm = multilabel_confusion_matrix(true_np, pred_np)

        results: dict[str, dict[str, float]] = {}
        for idx, cls_name in enumerate(class_names):
            tn, fp, fn, tp = cm[idx].ravel()
            tpv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
            results[cls_name] = {"PPV": round(tpv, 4), "NPV": round(npv, 4)}
        return results

    def generate_props(
        self,
        model_ckpt_path: str,
        model_config_file_path: str,
        data_config_file_path: str,
        output_path: str | None = None,
    ) -> None:
        """
        Run inference on validation set, compute TPV/NPV per class, and save to JSON.

        Args:
            model_ckpt_path: Path to the PyTorch Lightning checkpoint file.
            model_config_file_path: Path to yaml config file of the model.
            data_config_file_path: Path to yaml config file of the data.
            output_path: Optional path where to write the JSON metrics file.
                         Defaults to '<processed_dir_main>/classes.json'.
        """
        print("Extracting validation data for computation...")

        data_cls_path, data_cls_kwargs = parse_config_file(data_config_file_path)
        data_module: XYBaseDataModule = load_data_instance(
            data_cls_path, data_cls_kwargs
        )

        model_class_path, model_kwargs = parse_config_file(model_config_file_path)
        model = load_model_for_inference(
            model_ckpt_path, model_class_path, model_kwargs
        )

        val_loader = data_module.val_dataloader()
        print("Running inference on validation data...")

        y_true, y_pred = [], []
        for batch_idx, batch in enumerate(val_loader):
            data = model._process_batch(  # pylint: disable=W0212
                batch, batch_idx=batch_idx
            )
            labels = data["labels"]
            outputs = model(data, **data.get("model_kwargs", {}))
            logits = outputs["logits"]
            preds = torch.sigmoid(logits) > 0.5
            y_pred.extend(preds)
            y_true.extend(labels)

        print("Computing TPV and NPV metrics...")
        classes_file = Path(data_module.processed_dir_main) / "classes.txt"
        if output_path is None:
            output_file = Path(data_module.processed_dir_main) / "classes.json"
        else:
            output_file = Path(output_path)

        class_names = self.load_class_labels(classes_file)
        metrics = self.compute_tpv_npv(y_true, y_pred, class_names)

        with output_file.open("w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved TPV/NPV metrics to {output_file}")


class Main:
    """
    CLI wrapper for ClassesPropertiesGenerator.
    """

    def generate(
        self,
        model_ckpt_path: str,
        model_config_file_path: str,
        data_config_file_path: str,
        output_path: str | None = None,
    ) -> None:
        """
        CLI command to generate TPV/NPV JSON.

        Args:
            model_ckpt_path: Path to the PyTorch Lightning checkpoint file.
            model_config_file_path: Path to yaml config file of the model.
            data_config_file_path: Path to yaml config file of the data.
            output_path: Optional path where to write the JSON metrics file.
                         Defaults to '<processed_dir_main>/classes.json'.
        """
        generator = ClassesPropertiesGenerator()
        generator.generate_props(
            model_ckpt_path,
            model_config_file_path,
            data_config_file_path,
            output_path,
        )


if __name__ == "__main__":
    # _generate_classes_props_json.py generate \
    # --model_ckpt_path "model/ckpt/path" \
    # --model_config_file_path "model/config/file/path" \
    # --data_config_file_path "data/config/file/path" \
    # --output_path "output/file/path" # Optional
    CLI(Main, as_positional=False)
