import torch
from chebai.models.electra import extract_class_hierarchy
import os
import csv
import pickle

IMPLICATION_CACHE_FILE = "chebi.cache"


class ImplicationLoss(torch.nn.Module):
    def __init__(
        self, path_to_chebi, path_to_label_names, base_loss: torch.nn.Module = None
    ):
        super().__init__()
        self.base_loss = base_loss
        label_names = _load_label_names(path_to_label_names)
        hierarchy = _load_implications(path_to_chebi)
        implication_filter = _build_implication_filter(label_names, hierarchy)
        self.implication_filter_l = implication_filter[:, 0]
        self.implication_filter_r = implication_filter[:, 1]

    def forward(self, input, target, **kwargs):
        if target is not None:
            base_loss = self.base_loss(input, target.float())
        else:
            base_loss = 0
        pred = torch.sigmoid(input)
        l = pred[:, self.implication_filter_l]
        r = pred[:, self.implication_filter_r]
        # implication_loss = torch.sqrt(torch.mean(torch.sum(l*(1-r), dim=-1), dim=0))
        implication_loss = self._calculate_implication_loss(l, r)
        return base_loss + implication_loss

    def _calculate_implication_loss(self, l, r):
        capped_difference = torch.relu(l - r)
        return torch.mean(torch.sum((torch.softmax(capped_difference, dim=-1)*capped_difference), dim=-1), dim=0)


class DisjointLoss(ImplicationLoss):
    def __init__(
        self,
        path_to_chebi,
        path_to_label_names,
        path_to_disjointedness,
        base_loss: torch.nn.Module = None,
    ):
        super().__init__(path_to_chebi, path_to_label_names, base_loss)
        label_names = _load_label_names(path_to_label_names)
        hierarchy = _load_implications(path_to_chebi)
        self.disjoint_filter_l, self.disjoint_filter_r = _build_disjointness_filter(
            path_to_disjointedness, label_names, hierarchy
        )

    def forward(self, input, target, **kwargs):
        loss = super().forward(input, target, **kwargs)
        pred = torch.sigmoid(input)
        l = pred[:, self.disjoint_filter_l]
        r = pred[:, self.disjoint_filter_r]
        disjointness_loss = self._calculate_implication_loss(l, 1-r)
        return loss + disjointness_loss


def _load_label_names(path_to_label_names):
    with open(path_to_label_names) as fin:
        label_names = [int(line.strip()) for line in fin]
    return label_names


def _load_implications(path_to_chebi, implication_cache=IMPLICATION_CACHE_FILE):
    if os.path.isfile(implication_cache):
        with open(implication_cache, "rb") as fin:
            hierarchy = pickle.load(fin)
    else:
        hierarchy = extract_class_hierarchy(path_to_chebi)
        with open(implication_cache, "wb") as fout:
            pickle.dump(hierarchy, fout)
    return hierarchy


def _build_implication_filter(label_names, hierarchy):
    return torch.tensor(
        [
            (i1, i2)
            for i1, l1 in enumerate(label_names)
            for i2, l2 in enumerate(label_names)
            if l2 in hierarchy.pred[l1]
        ]
    )


def _build_disjointness_filter(path_to_disjointedness, label_names, hierarchy):
    disjoints = set()
    label_dict = dict(map(reversed, enumerate(label_names)))

    with open(path_to_disjointedness, "rt") as fin:
        reader = csv.reader(fin)
        for l1_raw, r1_raw in reader:
            l1 = int(l1_raw)
            r1 = int(r1_raw)
            disjoints.update(
                {
                    (label_dict[l2], label_dict[r2])
                    for r2 in hierarchy.succ[r1]
                    if r2 in label_names
                    for l2 in hierarchy.succ[l1]
                    if l2 in label_names and l2 < r2
                }
            )

    dis_filter = torch.tensor(list(disjoints))
    return dis_filter[:, 0], dis_filter[:, 1]
