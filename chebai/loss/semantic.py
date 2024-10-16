import csv
import math
import os
import pickle
from typing import List, Literal, Union

import torch

from chebai.loss.bce_weighted import BCEWeighted
from chebai.preprocessing.datasets.chebi import ChEBIOver100, _ChEBIDataExtractor
from chebai.preprocessing.datasets.pubchem import LabeledUnlabeledMixed


class ImplicationLoss(torch.nn.Module):
    """
    Implication Loss module.

    Args:
        data_extractor (Union[_ChEBIDataExtractor, LabeledUnlabeledMixed]): Data extractor for labels.
        base_loss (torch.nn.Module, optional): Base loss function. Defaults to None.
        tnorm (Literal["product", "lukasiewicz", "xu19"], optional): T-norm type. Defaults to "product".
        impl_loss_weight (float, optional): Weight of implication loss relative to base loss. Defaults to 0.1.
        pos_scalar (int, optional): Positive scalar exponent. Defaults to 1.
        pos_epsilon (float, optional): Epsilon value for numerical stability. Defaults to 0.01.
        multiply_by_softmax (bool, optional): Whether to multiply by softmax. Defaults to False.
    """

    def __init__(
        self,
        data_extractor: Union[_ChEBIDataExtractor, LabeledUnlabeledMixed],
        base_loss: torch.nn.Module = None,
        tnorm: Literal["product", "lukasiewicz", "xu19"] = "product",
        impl_loss_weight: float = 0.1,
        pos_scalar: int = 1,
        pos_epsilon: float = 0.01,
        multiply_by_softmax: bool = False,
    ):
        super().__init__()
        # automatically choose labeled subset for implication filter in case of mixed dataset
        if isinstance(data_extractor, LabeledUnlabeledMixed):
            data_extractor = data_extractor.labeled
        self.data_extractor = data_extractor
        # propagate data_extractor to base loss
        if isinstance(base_loss, BCEWeighted):
            base_loss.data_extractor = self.data_extractor
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
        self.eps = pos_epsilon
        self.multiply_by_softmax = multiply_by_softmax

    def forward(self, input: torch.Tensor, target: torch.Tensor, **kwargs) -> tuple:
        """
        Forward pass of the implication loss module.

        Args:
            input (torch.Tensor): Input tensor.
            target (torch.Tensor): Target tensor.
            **kwargs: Additional arguments.

        Returns:
            tuple: Tuple containing total loss, base loss, and implication loss.
        """
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

    def _calculate_implication_loss(
        self, l: torch.Tensor, r: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate implication loss based on T-norm and other parameters.

        Args:
            l (torch.Tensor): Left part of implication.
            r (torch.Tensor): Right part of implication.

        Returns:
            torch.Tensor: Calculated implication loss.
        """
        assert not l.isnan().any()
        assert not r.isnan().any()
        if self.pos_scalar != 1:
            l = (
                torch.pow(l + self.eps, 1 / self.pos_scalar)
                - math.pow(self.eps, 1 / self.pos_scalar)
            ) / (
                math.pow(1 + self.eps, 1 / self.pos_scalar)
                - math.pow(self.eps, 1 / self.pos_scalar)
            )
            one_min_r = torch.pow(1 - r, self.pos_scalar)
        else:
            one_min_r = 1 - r
        if self.tnorm == "product":
            individual_loss = l * one_min_r
        elif self.tnorm == "xu19":
            individual_loss = -torch.log(1 - l * one_min_r)
        elif self.tnorm == "lukasiewicz":
            individual_loss = torch.relu(l + one_min_r - 1)
        else:
            raise NotImplementedError(f"Unknown tnorm {self.tnorm}")

        if self.multiply_by_softmax:
            individual_loss = individual_loss * individual_loss.softmax(dim=-1)
        return torch.mean(
            torch.sum(individual_loss, dim=-1),
            dim=0,
        )

    def _load_implications(self, path_to_chebi: str) -> dict:
        """
        Load class hierarchy implications.

        Args:
            path_to_chebi (str): Path to the ChEBI ontology file.

        Returns:
            dict: Loaded hierarchy of implications.
        """
        if os.path.isfile(self.implication_cache_file):
            with open(self.implication_cache_file, "rb") as fin:
                hierarchy = pickle.load(fin)
        else:
            hierarchy = self.data_extractor.extract_class_hierarchy(path_to_chebi)
            with open(self.implication_cache_file, "wb") as fout:
                pickle.dump(hierarchy, fout)
        return hierarchy


class DisjointLoss(ImplicationLoss):
    """
    Disjoint Loss module, extending ImplicationLoss.

    Args:
        path_to_disjointness (str): Path to the disjointness data file (a csv file containing pairs of disjoint classes)
        data_extractor (Union[_ChEBIDataExtractor, LabeledUnlabeledMixed]): Data extractor for labels.
        base_loss (torch.nn.Module, optional): Base loss function. Defaults to None.
        disjoint_loss_weight (float, optional): Weight of disjointness loss. Defaults to 100.
        **kwargs: Additional arguments.
    """

    def __init__(
        self,
        path_to_disjointness: str,
        data_extractor: Union[_ChEBIDataExtractor, LabeledUnlabeledMixed],
        base_loss: torch.nn.Module = None,
        disjoint_loss_weight: float = 100,
        **kwargs,
    ):
        super().__init__(data_extractor, base_loss, **kwargs)
        self.disjoint_filter_l, self.disjoint_filter_r = _build_disjointness_filter(
            path_to_disjointness, self.label_names, self.hierarchy
        )
        self.disjoint_weight = disjoint_loss_weight

    def forward(self, input: torch.Tensor, target: torch.Tensor, **kwargs) -> tuple:
        """
        Forward pass of the disjoint loss module.

        Args:
            input (torch.Tensor): Input tensor.
            target (torch.Tensor): Target tensor.
            **kwargs: Additional arguments.

        Returns:
            tuple: Tuple containing total loss, base loss, implication loss, and disjointness loss.
        """
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


def _load_label_names(path_to_label_names: str) -> List:
    """
    Load label names from a file.

    Args:
        path_to_label_names (str): Path to the label names file.

    Returns:
        list: List of label names.
    """
    with open(path_to_label_names) as fin:
        label_names = [int(line.strip()) for line in fin]
    return label_names


def _build_implication_filter(label_names: List, hierarchy: dict) -> torch.Tensor:
    """
    Build implication filter based on label names and hierarchy.

    Args:
        label_names (list): List of label names.
        hierarchy (dict): Hierarchy of implications.

    Returns:
        torch.Tensor: Tensor representing implication filter.
    """
    return torch.tensor(
        [
            (i1, i2)
            for i1, l1 in enumerate(label_names)
            for i2, l2 in enumerate(label_names)
            if l2 in hierarchy.pred[l1]
        ]
    )


def _build_disjointness_filter(
    path_to_disjointness: str, label_names: List, hierarchy: dict
) -> tuple:
    """
    Build disjointness filter based on disjointness data and hierarchy.

    Args:
        path_to_disjointness (str): Path to the disjointness data file.
        label_names (list): List of label names.
        hierarchy (dict): Hierarchy of implications.

    Returns:
        tuple: Tuple containing tensors representing disjointness filter.
    """
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
