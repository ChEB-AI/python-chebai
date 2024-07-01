from torch.nn.utils.rnn import pad_sequence
import torch
from typing import List, Tuple, Union, Dict

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
    """Collator for handling ragged data samples."""

    def __call__(self, data: List[Union[Dict, Tuple]]) -> XYData:
        """Collate ragged data samples (i.e., samples of unequal size such as string representations of molecules) into
        a batch.

        Args:
            data (List[Union[Dict, Tuple]]): List of ragged data samples.

        Returns:
            XYData: Batched data with appropriate padding and masks.
        """
        model_kwargs: Dict = dict()
        loss_kwargs: Dict = dict()

        if isinstance(data[0], tuple):
            # For legacy data
            x, y, idents = zip(*data)
        else:
            x, y, idents = zip(
                *((d["features"], d["labels"], d.get("ident")) for d in data)
            )
        if any(x is not None for x in y):
            if any(x is None for x in y):
                non_null_labels = [i for i, r in enumerate(y) if r is not None]
                y = self.process_label_rows(
                    tuple(ye for i, ye in enumerate(y) if i in non_null_labels)
                )
                loss_kwargs["non_null_labels"] = non_null_labels
            else:
                y = self.process_label_rows(y)
        else:
            y = None
            loss_kwargs["non_null_labels"] = []

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
        """Process label rows by padding sequences.

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
