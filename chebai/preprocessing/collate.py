from typing import Dict, List, Tuple, Union

import torch
from torch.nn.utils.rnn import pad_sequence

from chebai.preprocessing.structures import XYData


class Collator:
    """Base class for collating data samples into a batch."""

    def __init__(self, **kwargs):
        pass

    def __call__(self, data: List[Dict]) -> XYData:
        """Collate a list of data samples into a batch.

        Args:
            data (List[Dict]): List of data samples.

        Returns:
            XYData: Batched data.
        """
        raise NotImplementedError


class DefaultCollator(Collator):
    """Default collator that extracts features and labels."""

    def __call__(self, data: List[Dict]) -> XYData:
        """Collate data samples by extracting features and labels.

        Args:
            data (List[Dict]): List of data samples.

        Returns:
            XYData: Batched data.
        """
        x, y = zip(*((d["features"], d["labels"]) for d in data))
        return XYData(x, y)


class RaggedCollator(Collator):
    """
    Collator for handling ragged data samples, designed to support scenarios where some labels may be missing (None).

    This class is specifically designed for preparing batches of "ragged" data, where the samples may have varying sizes,
    such as molecular representations or variable-length protein sequences. Additionally, it supports cases where some
    of the data samples might be partially labeled, which is useful for certain loss functions that allow training
    with incomplete or fuzzy data (e.g., fuzzy loss).

    During batching, the class pads the data samples to a uniform length, applies appropriate masks to differentiate
    between valid and padded elements, and ensures that label misalignment is handled by filtering out unlabelled
    data points. The indices of valid labels are stored in the `non_null_labels` field, which can be used later for
    metrics computation such as F1-score or MSE, especially in cases where some data points lack labels.

    Reference: https://github.com/ChEB-AI/python-chebai/pull/48#issuecomment-2324393829
    """

    def __call__(self, data: List[Union[Dict, Tuple]]) -> XYData:
        """
        Collate ragged data samples (i.e., samples of unequal size, such as molecular sequences) into a batch.

        Handles both fully and partially labeled data, where some samples may have `None` as their label. The indices
        of non-null labels are stored in the `non_null_labels` field, which is used to filter out predictions for
        unlabeled data during evaluation (e.g., F1, MSE). For models supporting partially labeled data, this method
        ensures alignment between features and labels.

        Args:
            data (List[Union[Dict, Tuple]]): List of ragged data samples. Each sample can be a dictionary or tuple
            with 'features', 'labels', and 'ident'.

        Returns:
            XYData: A batch of padded sequences and labels, including masks for valid positions and indices of
            non-null labels for metric computation.
        """
        model_kwargs: Dict = dict()
        # Indices of non-null labels are stored in key `non_null_labels` of loss_kwargs.
        loss_kwargs: Dict = dict()

        if isinstance(data[0], tuple):
            # For legacy data
            x, y, idents = zip(*data)
        else:
            x, y, idents = zip(
                *((d["features"], d["labels"], d.get("ident")) for d in data)
            )
        if any(x is not None for x in y):
            # If any label is not None: (None, None, `1`, None)
            if any(x is None for x in y):
                # If any label is None: (`None`, `None`, 1, `None`)
                non_null_labels = [i for i, r in enumerate(y) if r is not None]
                y = self.process_label_rows(
                    tuple(ye for i, ye in enumerate(y) if i in non_null_labels)
                )
                loss_kwargs["non_null_labels"] = non_null_labels
            else:
                # If all labels are not None: (`0`, `2`, `1`, `3`)
                y = self.process_label_rows(y)
        else:
            # If all labels are None : (`None`, `None`, `None`, `None`)
            y = None
            loss_kwargs["non_null_labels"] = []

        # Calculate the lengths of each sequence, create a binary mask for valid (non-padded) positions
        lens = torch.tensor(list(map(len, x)))
        model_kwargs["mask"] = torch.arange(max(lens))[None, :] < lens[:, None]
        model_kwargs["lens"] = lens

        return XYData(
            pad_sequence([torch.tensor(a) for a in x], batch_first=True),
            y,
            model_kwargs=model_kwargs,
            loss_kwargs=loss_kwargs,
            idents=idents,
        )

    def process_label_rows(self, labels: Tuple) -> torch.Tensor:
        """
        Process label rows by padding sequences to ensure uniform shape across the batch.

        This method pads the label rows, converting sequences of labels of different lengths into a uniform tensor.
        It ensures that `None` values in the labels are handled by substituting them with a default value(e.g.,`False`).

        Args:
            labels (Tuple): Tuple of label rows.

        Returns:
            torch.Tensor: Padded label sequences.
        """
        return pad_sequence(
            [
                torch.tensor([v if v is not None else False for v in row])
                for row in labels
            ],
            batch_first=True,
        )
