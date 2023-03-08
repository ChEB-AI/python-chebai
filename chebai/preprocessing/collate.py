from torch.nn.utils.rnn import pad_sequence
import torch

from chebai.preprocessing.structures import XYData


class Collater:
    def __init__(self, **kwargs):
        pass

    def __call__(self, data):
        raise NotImplementedError


class DefaultCollater(Collater):
    def __call__(self, data):
        x, y = zip(*((d["features"], d["labels"]) for d in data))
        return XYData(x, y)


class RaggedCollater(Collater):
    def __call__(self, data):
        if isinstance(data[0], tuple):
            # For legacy data
            x, y, _ = zip(*data)
        else:
            x, y = zip(*((d["features"], d["labels"]) for d in data))
        if not all(x is None for x in y):
            is_not_none = ~torch.tensor([[v is None for v in row] for row in y])
            y = pad_sequence(
                [
                    torch.tensor([v if v is not None else False for v in row])
                    for row in y
                ],
                batch_first=True,
            )
        else:
            is_not_none = None
            y = None
        lens = torch.tensor(list(map(len, x)))
        mask = torch.arange(max(lens))[None, :] < lens[:, None]
        return XYData(
            pad_sequence([torch.tensor(a) for a in x], batch_first=True),
            y,
            additional_fields=dict(lens=lens, mask=mask, target_mask=is_not_none),
        )
