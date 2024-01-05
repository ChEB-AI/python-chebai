from torch.nn.utils.rnn import pad_sequence
import torch

from chebai.preprocessing.structures import XYData


class Collater:
    def __init__(self, **kwargs):
        pass

    def __call__(self, data) -> XYData:
        raise NotImplementedError


class DefaultCollater(Collater):
    def __call__(self, data):
        x, y = zip(*((d["features"], d["labels"]) for d in data))
        return XYData(x, y)


class RaggedCollater(Collater):
    def __call__(self, data):
        model_kwargs = dict()
        loss_kwargs = dict()
        if isinstance(data[0], tuple):
            # For legacy data
            x, y, idents = zip(*data)
        else:
            x, y, idents = zip(
                *((d["features"], d["labels"], d.get("ident")) for d in data)
            )
        if any(x is not None for x in y):
            target_mask_candidates = [
                [v is not None for v in row] for row in y if row is not None
            ]
            if any(map(any, target_mask_candidates)):
                loss_kwargs["target_mask"] = torch.tensor(target_mask_candidates)
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

    def process_label_rows(self, labels):
        return pad_sequence(
            [
                torch.tensor([v if v is not None else False for v in row])
                for row in labels
            ],
            batch_first=True,
        )
