import csv
import os
import pickle

import torch
from typing import Literal

from chebai.preprocessing.datasets.chebi import _ChEBIDataExtractor


class ImplicationLoss(torch.nn.Module):
    def __init__(
        self,
        path_to_chebi,
        path_to_label_names,
        data_extractor: _ChEBIDataExtractor,
        base_loss: torch.nn.Module = None,
        aggregation="",
        tnorm: Literal["product", "lukasiewicz"] = "product",
    ):
        super().__init__()
        self.data_extractor = data_extractor
        self.base_loss = base_loss
        self.implication_cache_file = f"implications_{self.data_extractor.name}.cache"
        label_names = _load_label_names(path_to_label_names)
        hierarchy = self._load_implications(path_to_chebi)
        implication_filter = _build_implication_filter(label_names, hierarchy)
        self.implication_filter_l = implication_filter[:, 0]
        self.implication_filter_r = implication_filter[:, 1]
        self.tnorm = tnorm

    def forward(self, input, target, **kwargs):
        nnl = kwargs.pop("non_null_labels", None)
        if nnl:
            labeled_input = input[nnl]
        else:
            labeled_input = input
        if target is not None:
            base_loss = self.base_loss(labeled_input, target.float())
        else:
            base_loss = 0
        pred = torch.sigmoid(input)
        l = pred[:, self.implication_filter_l]
        r = pred[:, self.implication_filter_r]
        # implication_loss = torch.sqrt(torch.mean(torch.sum(l*(1-r), dim=-1), dim=0))
        implication_loss = self._calculate_implication_loss(l, r)
        return base_loss + 0.01 * implication_loss

    def _calculate_implication_loss(self, l, r):
        if self.tnorm == "product":
            individual_loss = l * (1 - r)
        else:  # lukasiewicz
            individual_loss = torch.relu(l - r)
        return torch.mean(
            torch.sum(individual_loss, dim=-1),
            dim=0,
        )

    def _load_implications(self, path_to_chebi):
        if os.path.isfile(self.implication_cache_file):
            with open(self.implication_cache_file, "rb") as fin:
                hierarchy = pickle.load(fin)
        else:
            hierarchy = self.data_extractor.extract_class_hierarchy(path_to_chebi)
            with open(self.implication_cache_file, "wb") as fout:
                pickle.dump(hierarchy, fout)
        return hierarchy


class DisjointLoss(ImplicationLoss):
    def __init__(
        self,
        path_to_chebi,
        path_to_label_names,
        path_to_disjointedness,
        data_extractor: _ChEBIDataExtractor,
        base_loss: torch.nn.Module = None,
        tnorm: Literal["product", "lukasiewicz"] = "product",
    ):
        super().__init__(
            path_to_chebi, path_to_label_names, data_extractor, base_loss, tnorm
        )
        label_names = _load_label_names(path_to_label_names)
        hierarchy = self._load_implications(path_to_chebi)
        self.disjoint_filter_l, self.disjoint_filter_r = _build_disjointness_filter(
            path_to_disjointedness, label_names, hierarchy
        )

    def forward(self, input, target, **kwargs):
        loss = super().forward(input, target, **kwargs)
        pred = torch.sigmoid(input)
        l = pred[:, self.disjoint_filter_l]
        r = pred[:, self.disjoint_filter_r]
        disjointness_loss = self._calculate_implication_loss(l, 1 - r)
        return loss + disjointness_loss


def _load_label_names(path_to_label_names):
    with open(path_to_label_names) as fin:
        label_names = [int(line.strip()) for line in fin]
    return label_names


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
