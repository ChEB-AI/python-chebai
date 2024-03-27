import csv
import os
import pickle

import torch
from typing import Literal

from chebai.preprocessing.datasets.chebi import _ChEBIDataExtractor, ChEBIOver100


class ImplicationLoss(torch.nn.Module):
    def __init__(
        self,
        data_extractor: _ChEBIDataExtractor,
        base_loss: torch.nn.Module = None,
        tnorm: Literal["product", "lukasiewicz", "xu19"] = "product",
        impl_loss_weight=0.1,  # weight of implication loss in relation to base_loss
        pos_scalar=1,
    ):
        super().__init__()
        self.data_extractor = data_extractor
        self.base_loss = base_loss
        self.implication_cache_file = f"implications_{self.data_extractor.name}.cache"
        self.label_names = _load_label_names(
            os.path.join(data_extractor.raw_dir, "classes.txt")
        )
        self.hierarchy = self._load_implications(
            os.path.join(data_extractor.raw_dir, "chebi.obo")
        )
        implication_filter = _build_implication_filter(self.label_names, self.hierarchy)
        self.implication_filter_l = implication_filter[:, 0]
        self.implication_filter_r = implication_filter[:, 1]
        self.tnorm = tnorm
        self.impl_weight = impl_loss_weight
        self.pos_scalar = pos_scalar

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

        return (
            base_loss + self.impl_weight * implication_loss,
            base_loss,
            implication_loss,
        )

    def _calculate_implication_loss(self, l, r):
        if self.pos_scalar != 1:
            l = torch.pow(l, 1 / self.pos_scalar)
            r = torch.pow(r, self.pos_scalar)
        assert not l.isnan().any()
        assert not r.isnan().any()
        if self.tnorm == "product":
            individual_loss = l * (1 - r)
        elif self.tnorm == "xu19":
            individual_loss = -torch.log(1 - l * (1 - r))
        elif self.tnorm == "lukasiewicz":
            individual_loss = torch.relu(l - r)
        else:
            raise NotImplementedError(f"Unknown tnorm {self.tnorm}")

        assert not individual_loss.isnan().any()
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
        path_to_disjointness,
        data_extractor: _ChEBIDataExtractor,
        base_loss: torch.nn.Module = None,
        disjoint_loss_weight=100,
        **kwargs,
    ):
        super().__init__(data_extractor, base_loss, **kwargs)
        self.disjoint_filter_l, self.disjoint_filter_r = _build_disjointness_filter(
            path_to_disjointness, self.label_names, self.hierarchy
        )
        self.disjoint_weight = disjoint_loss_weight

    def forward(self, input, target, **kwargs):
        loss, base_loss, impl_loss = super().forward(input, target, **kwargs)
        pred = torch.sigmoid(input)
        l = pred[:, self.disjoint_filter_l]
        r = pred[:, self.disjoint_filter_r]
        disjointness_loss = self._calculate_implication_loss(l, 1 - r)
        return (
            loss + self.disjoint_weight * disjointness_loss,
            base_loss,
            impl_loss,
            disjointness_loss,
        )


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


def _build_disjointness_filter(path_to_disjointness, label_names, hierarchy):
    disjoints = set()
    label_dict = dict(map(reversed, enumerate(label_names)))

    with open(path_to_disjointness, "rt") as fin:
        reader = csv.reader(fin)
        for l1_raw, r1_raw in reader:
            l1 = int(l1_raw)
            r1 = int(r1_raw)
            if l1 == 36233 and r1 == 63353:
                # ignore disaccharide-disaccharide derivative disjointness axiom
                continue
            disjoints.update(
                {
                    (label_dict[l2], label_dict[r2])
                    for r2 in list(hierarchy.succ[r1]) + [r1]
                    if r2 in label_names
                    for l2 in list(hierarchy.succ[l1]) + [l1]
                    if l2 in label_names
                }
            )

    dis_filter = torch.tensor(list(disjoints))
    return dis_filter[:, 0], dis_filter[:, 1]


if __name__ == "__main__":
    loss = DisjointLoss(
        os.path.join("data", "disjoint.csv"), ChEBIOver100(chebi_version=227)
    )
