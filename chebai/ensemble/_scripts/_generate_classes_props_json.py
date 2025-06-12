"""Generate TPV/NPV JSON for multi-class classification models."""

import json
from pathlib import Path

import pandas as pd
import torch
from jsonargparse import CLI
from sklearn.metrics import multilabel_confusion_matrix
from torch.utils.data import DataLoader

from chebai.ensemble._utils import load_class
from chebai.models.base import ChebaiBaseNet
from chebai.preprocessing.collate import Collator


class ClassesPropertiesGenerator:
    """
    Computes TPV (Precision/ True Predictive Value) and NPV (Negative Predictive Value)
    for each class in a multi-class classification problem using a PyTorch Lightning model.
    """

    @staticmethod
    def load_class_labels(path: str) -> list[str]:
        """
        Load a list of class names from a .json or .txt file.

        Args:
            path (str): Path to class labels file.

        Returns:
            list[str]: List of class names.
        """
        with open(path) as f:
            return [line.strip() for line in f if line.strip()]

    @staticmethod
    def compute_tpv_npv(
        y_true: list[torch.Tensor], y_pred: list[torch.Tensor], class_names: list[str]
    ) -> dict[str, dict[str, float]]:
        """
        Compute TPV and NPV for each class in a multi-label classification problem.

        Args:
            y_true (list[Tensor]): List of binary ground truth label tensors per sample.
            y_pred (list[Tensor]): List of binary prediction tensors per sample.
            class_names (list[str]): List of class names corresponding to class indices.

        Returns:
            dict[str, dict[str, float]]: Dictionary with class names as keys and TPV/NPV as values.
        """
        # Convert list of tensors to a single binary matrix
        y_true_tensor = torch.stack(y_true).cpu().numpy().astype(int)
        y_pred_tensor = torch.stack(y_pred).cpu().numpy().astype(int)

        cm = multilabel_confusion_matrix(y_true_tensor, y_pred_tensor)

        metrics = {}
        for i, cls in enumerate(class_names):
            tn, fp, fn, tp = cm[i].ravel()

            PPV = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            NPV = tn / (tn + fn) if (tn + fn) > 0 else 0.0

            metrics[cls] = {"PPV": round(TPV, 4), "NPV": round(NPV, 4)}

        return metrics

    def generate_props(
        self,
        model_path: str,
        model_class_path: str,
        splits_path: str,
        data_path: str,
        classes_file_path: str,
        collator_class_path: str,
        output_path: str,
        batch_size: int = 32,
    ) -> None:
        """
        Main method to compute TPV/NPV from validation data and save as JSON.

        Args:
            model_path (str): Path to the PyTorch Lightning model checkpoint.
            model_class_path (str): Full path to the model class to load.
            splits_path (str): CSV file with 'id' and 'split' columns.
            data_path (str): processed `data.pt` file path.
            classes_file_path (str): Path to file containing class names `classes.txt`.
            collator_class_path (str): Full path to the collator class.
            output_path (str): Output path for the saving JSON file.
            batch_size (int): Batch size for inference.
        """
        print("Extracting validation data for computation...")
        splits_df = pd.read_csv(splits_path)
        validation_ids = set(splits_df[splits_df["split"] == "validation"]["id"])
        data_df = pd.DataFrame(torch.load(data_path, weights_only=False))
        val_df = data_df[data_df["ident"].isin(validation_ids)]

        # Load model
        print(f"Loading model from {model_path} ...")
        model_cls = load_class(model_class_path)
        if not issubclass(model_cls, ChebaiBaseNet):
            raise TypeError("Loaded model is not a valid LightningModule.")
        model = model_cls.load_from_checkpoint(model_path, input_dim=3)
        model.freeze()
        model.eval()

        # Load collator
        collator_cls = load_class(collator_class_path)
        if not issubclass(collator_cls, Collator):
            raise TypeError(f"{collator_cls} must be subclass of Collator")
        collator = collator_cls()

        val_loader = DataLoader(
            val_df.to_dict(orient="records"),
            collate_fn=collator,
            batch_size=batch_size,
            shuffle=False,
        )

        print("Running inference on validation data...")
        y_true, y_pred = [], []
        for batch_idx, batch in enumerate(val_loader):
            data = model._process_batch(batch, batch_idx=batch_idx)
            labels = data["labels"]
            model_output = model(data, **data.get("model_kwargs", dict()))
            sigmoid_logits = torch.sigmoid(model_output["logits"])
            preds = sigmoid_logits > 0.5
            y_pred.extend(preds)
            y_true.extend(labels)

        # Compute and save metrics
        print("Computing TPV and NPV metrics...")
        classes_file_path = Path(classes_file_path)
        if output_path is None:
            output_path = classes_file_path.parent / "classes.json"
        class_names = self.load_class_labels(classes_file_path)
        metrics = self.compute_tpv_npv(y_true, y_pred, class_names)
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved TPV/NPV metrics to {output_path}")


class Main:
    """
    Command-line interface wrapper for the ClassesPropertiesGenerator.
    """

    def generate(
        self,
        model_path: str,
        splits_path: str,
        data_path: str,
        classes_file_path: str,
        model_class_path: str,
        collator_class_path: str = "chebai.preprocessing.collate.RaggedCollator",
        batch_size: int = 32,
        output_path: str = None,  # Default path will be the directory of classes_file_path
    ) -> None:
        """
        Entry point for CLI use.

        Args:
            model_path (str): Path to the PyTorch Lightning model checkpoint.
            model_class_path (str): Full path to the model class to load.
            splits_path (str): CSV file with 'id' and 'split' columns.
            data_path (str): processed `data.pt` file path.
            classes_file_path (str): Path to file containing class names `classes.txt`.
            collator_class_path (str): Full path to the collator class.
            output_path (str): Output path for the saving JSON file.
            batch_size (int): Batch size for inference.
        """
        generator = ClassesPropertiesGenerator()
        generator.generate_props(
            model_path=model_path,
            model_class_path=model_class_path,
            splits_path=splits_path,
            data_path=data_path,
            classes_file_path=classes_file_path,
            collator_class_path=collator_class_path,
            output_path=output_path,
            batch_size=batch_size,
        )


if __name__ == "__main__":
    # _generate_classes_props_json.py generate \
    # --model_path "model/ckpt/path" \
    # --splits_path "splits/file/path" \
    # --data_path "data.pt/file/path" \
    # --classes_file_path "classes/file/path" \
    # --model_class_path "model.class.path" \
    # --collator_class_path "collator.class.path" \
    # --batch_size 32 \  # Optional, default is 32
    # --output_path "output/file/path" # Optional, default will be the directory of classes_file_path
    CLI(Main, as_positional=False)
