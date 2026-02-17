"""
Docstring for chebai.preprocessing.migration.migrate_checkpoints

This script migrates lightning checkpoints created before python-chebai
version 1.2.1 to be compatible with the new version.

The main change is the addition of a new key "classification_labels" in the checkpoint,
which is required for the new version of python-chebai from version 1.2.1 onwards.

For more details, see the pull request: https://github.com/ChEB-AI/python-chebai/pulls
"""

import sys

import torch


def add_class_labels_to_checkpoint(input_path, classes_file_path):
    with open(classes_file_path, "r") as f:
        class_labels = [line.strip() for line in f.readlines()]

    assert len(class_labels) > 0, "The classes file is empty."

    # 1. Load the checkpoint
    checkpoint = torch.load(
        input_path, map_location=torch.device("cpu"), weights_only=False
    )

    if "classification_labels" in checkpoint:
        print(
            "Warning: 'classification_labels' key already exists in the checkpoint and will be overwritten."
        )

    # 2. Add your custom key/value pair
    checkpoint["classification_labels"] = class_labels

    # 3. Save the modified checkpoint
    output_path = input_path.replace(".ckpt", "_modified.ckpt")
    torch.save(checkpoint, output_path)
    print(f"Successfully added classification_labels and saved to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python modify_checkpoints.py <input_checkpoint> <classes_file>")
        sys.exit(1)

    input_ckpt = sys.argv[1]
    classes_file = sys.argv[2]

    add_class_labels_to_checkpoint(
        input_path=input_ckpt, classes_file_path=classes_file
    )
