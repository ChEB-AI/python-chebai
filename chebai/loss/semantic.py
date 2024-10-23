import csv
import math
import os
import pickle
from typing import List, Literal, Union

import torch

from chebai.loss.bce_weighted import BCEWeighted
from chebai.preprocessing.datasets import XYBaseDataModule
from chebai.preprocessing.datasets.chebi import ChEBIOver100, _ChEBIDataExtractor
from chebai.preprocessing.datasets.pubchem import LabeledUnlabeledMixed


def _filter_by_ground_truth(individual_loss, target, filter_l, filter_r):
    # mask of ground truth labels for implications, shape (batch_size, num_implications/num_disjointnesses)
    target_l = target[:, filter_l]
    target_r = target[:, filter_r]

    # filter individual loss: for an implication A->B, only apply loss to A if A is labeled false,
    # only apply loss to B if B is labeled true
    applicable_individual_loss_l = individual_loss * (1 - target_l)
    applicable_individual_loss_r = individual_loss * target_r
    class_loss = torch.zeros(target.shape, device=target.device)
    # for each class, sum up the losses for all implication antecedents and consequents
    for cls in range(class_loss.shape[1]):
        class_loss[:, cls] = applicable_individual_loss_l[:, filter_l == cls].sum(dim=1)
        class_loss[:, cls] += applicable_individual_loss_r[:, filter_r == cls].sum(
            dim=1
        )
    return class_loss


class ImplicationLoss(torch.nn.Module):
    """
    Implication Loss module.

    Args:
        data_extractor _ChEBIDataExtractor: Data extractor for labels.
        base_loss (torch.nn.Module, optional): Base loss function. Defaults to None.
        fuzzy_implication (Literal["product", "lukasiewicz", "xu19"], optional): T-norm type. Defaults to "product".
        impl_loss_weight (float, optional): Weight of implication loss relative to base loss. Defaults to 0.1.
        pos_scalar (int, optional): Positive scalar exponent. Defaults to 1.
        pos_epsilon (float, optional): Epsilon value for numerical stability. Defaults to 0.01.
        multiply_by_softmax (bool, optional): Whether to multiply by softmax. Defaults to False.
        use_sigmoidal_implication (bool, optional): Whether to use the sigmoidal fuzzy implication based on the
        specified fuzzy_implication (as defined by van Krieken et al., 2022: Analyzing Differentiable Fuzzy Logic
        Operators). Defaults to False.
    """

    def __init__(
        self,
        data_extractor: XYBaseDataModule,
        base_loss: torch.nn.Module = None,
        fuzzy_implication: Literal[
            "reichenbach",
            "rc",
            "lukasiewicz",
            "lk",
            "xu19",
            "kleene_dienes",
            "kd",
            "goedel",
            "g",
            "reverse-goedel",
            "rg",
        ] = "reichenbach",
        impl_loss_weight: float = 0.1,
        pos_scalar: Union[int, float] = 1,
        pos_epsilon: float = 0.01,
        multiply_by_softmax: bool = False,
        use_sigmoidal_implication: bool = False,
        weight_epoch_dependent: Union[bool | tuple[int, int]] = False,
        start_at_epoch: int = 0,
    ):
        super().__init__()
        # automatically choose labeled subset for implication filter in case of mixed dataset
        if isinstance(data_extractor, LabeledUnlabeledMixed):
            data_extractor = data_extractor.labeled
        assert isinstance(data_extractor, _ChEBIDataExtractor)
        self.data_extractor = data_extractor
        # propagate data_extractor to base loss
        if isinstance(base_loss, BCEWeighted):
            base_loss.data_extractor = self.data_extractor
            base_loss.reduction = (
                "none"  # needed to multiply fuzzy loss with base loss for each sample
            )
        self.base_loss = base_loss
        self.implication_cache_file = f"implications_{self.data_extractor.name}.cache"
        self.label_names = _load_label_names(
            os.path.join(data_extractor.processed_dir_main, "classes.txt")
        )
        self.hierarchy = self._load_implications(
            os.path.join(data_extractor.raw_dir, "chebi.obo")
        )
        implication_filter = _build_implication_filter(self.label_names, self.hierarchy)
        self.implication_filter_l = implication_filter[:, 0]
        self.implication_filter_r = implication_filter[:, 1]
        self.fuzzy_implication = fuzzy_implication
        self.impl_weight = impl_loss_weight
        self.pos_scalar = pos_scalar
        self.eps = pos_epsilon
        self.multiply_by_softmax = multiply_by_softmax
        self.use_sigmoidal_implication = use_sigmoidal_implication
        self.weight_epoch_dependent = weight_epoch_dependent
        self.start_at_epoch = start_at_epoch

    def _calculate_unaggregated_fuzzy_loss(
        self, pred, target: torch.Tensor, weight, filter_l, filter_r, **kwargs
    ):
        l = pred[:, filter_l]
        r = pred[:, filter_r]
        individual_loss = self._calculate_implication_loss(l, r, target)
        implication_loss = _filter_by_ground_truth(
            individual_loss,
            target,
            filter_l,
            filter_r,
        )
        unweighted_mean = implication_loss.mean()
        implication_loss_weighted = implication_loss
        if "current_epoch" in kwargs and self.weight_epoch_dependent:
            sigmoid_center = (
                self.weight_epoch_dependent[0]
                if isinstance(self.weight_epoch_dependent, tuple)
                else 50
            )
            sigmoid_spread = (
                self.weight_epoch_dependent[1]
                if isinstance(self.weight_epoch_dependent, tuple)
                else 10
            )
            # sigmoid function centered around epoch 50
            implication_loss_weighted = implication_loss_weighted / (
                1
                + math.exp(-(kwargs["current_epoch"] - sigmoid_center) / sigmoid_spread)
            )
        implication_loss_weighted *= weight
        weighted_mean = implication_loss_weighted.mean()

        return implication_loss_weighted, unweighted_mean, weighted_mean

    def _calculate_unaggregated_base_loss(self, input, target, **kwargs):
        nnl = kwargs.pop("non_null_labels", None)
        labeled_input = input[nnl] if nnl else input

        if target is not None and self.base_loss is not None:
            return self.base_loss(labeled_input, target.float())
        else:
            return torch.zeros(input.shape, device=input.device)

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
        base_loss = self._calculate_unaggregated_base_loss(input, target, **kwargs)
        loss_components = {"base_loss": base_loss.mean()}

        if "current_epoch" in kwargs and self.start_at_epoch > kwargs["current_epoch"]:
            return base_loss.mean(), loss_components

        pred = torch.sigmoid(input)
        fuzzy_loss, unweighted_fuzzy_mean, weighted_fuzzy_mean = (
            self._calculate_unaggregated_fuzzy_loss(
                pred,
                target,
                self.impl_weight,
                self.implication_filter_l,
                self.implication_filter_r,
                **kwargs,
            )
        )
        loss_components["unweighted_fuzzy_loss"] = unweighted_fuzzy_mean
        loss_components["weighted_fuzzy_loss"] = weighted_fuzzy_mean
        if self.base_loss is None or target is None:
            total_loss = self.impl_weight * fuzzy_loss
        else:
            total_loss = base_loss * (1 + self.impl_weight * fuzzy_loss)
        return total_loss.mean(), loss_components

    def _calculate_implication_loss(
        self, l: torch.Tensor, r: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate implication loss based on T-norm and other parameters.

        Args:
            l (torch.Tensor): Left part of implication.
            r (torch.Tensor): Right part of implication.

        Returns:
            torch.Tensor: Calculated implication loss.
        """
        assert not l.isnan().any(), (
            f"l contains NaN values - l.shape: {l.shape}, l.isnan().sum(): {l.isnan().sum()}, "
            f"l: {l}"
        )
        assert not r.isnan().any(), (
            f"r contains NaN values - r.shape: {r.shape}, r.isnan().sum(): {r.isnan().sum()}, "
            f"r: {r}"
        )
        if self.pos_scalar != 1:
            l = (
                torch.pow(l + self.eps, 1 / self.pos_scalar)
                - math.pow(self.eps, 1 / self.pos_scalar)
            ) / (
                math.pow(1 + self.eps, 1 / self.pos_scalar)
                - math.pow(self.eps, 1 / self.pos_scalar)
            )
            one_min_r = (
                torch.pow(1 - r + self.eps, 1 / self.pos_scalar)
                - math.pow(self.eps, 1 / self.pos_scalar)
            ) / (
                math.pow(1 + self.eps, 1 / self.pos_scalar)
                - math.pow(self.eps, 1 / self.pos_scalar)
            )
        else:
            one_min_r = 1 - r
        # for each implication I, calculate 1 - I(l, 1-one_min_r)
        # for S-implications, this is equivalent to the t-norm
        if self.fuzzy_implication in ["reichenbach", "rc"]:
            individual_loss = l * one_min_r
        # xu19 (from Xu et al., 2019: Semantic loss) is not a fuzzy implication, but behaves similar to the Reichenbach
        # implication
        elif self.fuzzy_implication == "xu19":
            individual_loss = -torch.log(1 - l * one_min_r)
        elif self.fuzzy_implication in ["lukasiewicz", "lk"]:
            individual_loss = torch.relu(l + one_min_r - 1)
        elif self.fuzzy_implication in ["kleene_dienes", "kd"]:
            individual_loss = torch.min(l, 1 - r)
        elif self.fuzzy_implication in ["goedel", "g"]:
            individual_loss = torch.where(l <= r, 0, one_min_r)
        elif self.fuzzy_implication in ["reverse-goedel", "rg"]:
            individual_loss = torch.where(l <= r, 0, l)
        else:
            raise NotImplementedError(
                f"Unknown fuzzy implication {self.fuzzy_implication}"
            )

        if self.use_sigmoidal_implication:
            # formula by van Krieken, 2022, applied to fuzzy implication with default parameters: b_0 = 0.5, s = 9
            # parts that only depend on b_0 and s are pre-calculated
            implication = 1 - individual_loss
            sigmoidal_implication = 0.01123379 * (
                91.0171 * torch.sigmoid(9 * (implication - 0.5)) - 1
            )
            individual_loss = 1 - sigmoidal_implication

        if self.multiply_by_softmax:
            individual_loss = individual_loss * individual_loss.softmax(dim=-1)

        return individual_loss

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
        base_loss = self._calculate_unaggregated_base_loss(input, target, **kwargs)
        loss_components = {"base_loss": base_loss.mean()}

        if "current_epoch" in kwargs and self.start_at_epoch > kwargs["current_epoch"]:
            return base_loss.mean(), loss_components

        pred = torch.sigmoid(input)
        impl_loss, unweighted_impl_mean, weighted_impl_mean = (
            self._calculate_unaggregated_fuzzy_loss(
                pred,
                target,
                self.impl_weight,
                self.implication_filter_l,
                self.implication_filter_r,
                **kwargs,
            )
        )
        loss_components["unweighted_implication_loss"] = unweighted_impl_mean
        loss_components["weighted_implication_loss"] = weighted_impl_mean

        disj_loss, unweighted_disj_mean, weighted_disj_mean = (
            self._calculate_unaggregated_fuzzy_loss(
                pred,
                target,
                self.disjoint_weight,
                self.disjoint_filter_l,
                self.disjoint_filter_r,
                **kwargs,
            )
        )
        loss_components["unweighted_disjointness_loss"] = unweighted_disj_mean
        loss_components["weighted_disjointness_loss"] = weighted_disj_mean

        if self.base_loss is None or target is None:
            total_loss = self.impl_weight * impl_loss + self.disjoint_weight * disj_loss
        else:
            total_loss = base_loss * (
                1 + self.impl_weight * impl_loss + self.disjoint_weight * disj_loss
            )
        return total_loss.mean(), loss_components


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
        os.path.join("data", "disjoint.csv"), ChEBIOver100(chebi_version=231)
    )
    l = loss(torch.randn(10, 997), torch.randn(10, 997))

    loss_with_base = DisjointLoss(
        os.path.join("data", "disjoint.csv"),
        ChEBIOver100(chebi_version=231),
        base_loss=BCEWeighted(beta=0.99),
    )
    lb = loss_with_base(torch.randn(10, 997), torch.randn(10, 997))
    print(l)
    print(lb)
