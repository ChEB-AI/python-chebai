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
        weight_epoch_dependent (Union[bool, tuple[int, int]], optional): Whether to weight the implication loss
        depending on the current epoch with the sigmoid function sigmoid((epoch-c)/s). If True, c=50 and s=10,
        otherwise, a tuple of integers (c,s) can be supplied. Defaults to False.
        start_at_epoch (int, optional): Epoch at which to start applying the loss. Defaults to 0.
        violations_per_cls_aggregator (Literal["sum", "max"], optional): How to aggregate violations for each class.
        If a class is involved in several implications / disjointnesses, the loss value for this class will be
        aggregated with this method. Defaults to "sum".
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
            "binary",
            "b",
        ] = "reichenbach",
        impl_loss_weight: float = 0.1,
        pos_scalar: Union[int, float] = 1,
        pos_epsilon: float = 0.01,
        multiply_by_softmax: bool = False,
        use_sigmoidal_implication: bool = False,
        weight_epoch_dependent: Union[bool | tuple[int, int]] = False,
        start_at_epoch: int = 0,
        violations_per_cls_aggregator: Literal[
            "sum", "max", "mean", "log-sum", "log-max", "log-mean"
        ] = "sum",
        multiply_with_base_loss: bool = True,
        no_grads: bool = False,
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
        implication_filter_dense = _build_dense_filter(
            _build_implication_filter(self.label_names, self.hierarchy),
            len(self.label_names),
        )
        self.implication_filter_l = implication_filter_dense
        self.implication_filter_r = self.implication_filter_l.transpose(0, 1)
        self.fuzzy_implication = fuzzy_implication
        self.impl_weight = impl_loss_weight
        self.pos_scalar = pos_scalar
        self.eps = pos_epsilon
        self.multiply_by_softmax = multiply_by_softmax
        self.use_sigmoidal_implication = use_sigmoidal_implication
        self.weight_epoch_dependent = weight_epoch_dependent
        self.start_at_epoch = start_at_epoch
        self.violations_per_cls_aggregator = violations_per_cls_aggregator
        self.multiply_with_base_loss = multiply_with_base_loss
        self.no_grads = no_grads

    def _calculate_unaggregated_fuzzy_loss(
        self,
        pred,
        target: torch.Tensor,
        weight,
        filter_l,
        filter_r,
        mode="impl",
        **kwargs,
    ):
        # for each batch, get all pairwise losses: [a1, a2, a3] -> [[a1*a1, a1*a2, a1*a3],[a2*a1,...],[a3*a1,...]]
        preds_expanded1 = pred.unsqueeze(1).expand(-1, pred.shape[1], -1)
        preds_expanded2 = pred.unsqueeze(2).expand(-1, -1, pred.shape[1])
        # filter by implication relations and labels

        label_filter = target.unsqueeze(2).expand(-1, -1, pred.shape[1])
        filter_l = filter_l.to(pred.device).unsqueeze(0).expand(pred.shape[0], -1, -1)
        filter_r = filter_r.to(pred.device).unsqueeze(0).expand(pred.shape[0], -1, -1)
        if mode == "impl":
            all_implications = self._calculate_implication_loss(
                preds_expanded2, preds_expanded1
            )
        else:
            all_implications = self._calculate_implication_loss(
                preds_expanded2, 1 - preds_expanded1
            )
        loss_impl_l = all_implications * filter_l * (1 - label_filter)
        if mode == "impl":
            loss_impl_r = all_implications.transpose(1, 2) * filter_r * label_filter
            loss_impl_sum = loss_impl_l + loss_impl_r
        else:
            loss_impl_sum = loss_impl_l

        if self.violations_per_cls_aggregator.startswith("log-"):
            loss_impl_sum = -torch.log(1 - loss_impl_sum)
            violations_per_cls_aggregator = self.violations_per_cls_aggregator[4:]
        else:
            violations_per_cls_aggregator = self.violations_per_cls_aggregator
        if violations_per_cls_aggregator == "sum":
            loss_by_cls = loss_impl_sum.sum(dim=-1)
        elif violations_per_cls_aggregator == "max":
            loss_by_cls = loss_impl_sum.max(dim=-1).values
        elif violations_per_cls_aggregator == "mean":
            loss_by_cls = loss_impl_sum.mean(dim=-1)
        else:
            raise NotImplementedError(
                f"Unknown violations_per_cls_aggregator {self.violations_per_cls_aggregator}"
            )

        unweighted_mean = loss_by_cls.mean()
        implication_loss_weighted = loss_by_cls
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
        if self.no_grads:
            fuzzy_loss = fuzzy_loss.detach()
        loss_components["unweighted_fuzzy_loss"] = unweighted_fuzzy_mean
        loss_components["weighted_fuzzy_loss"] = weighted_fuzzy_mean
        if self.base_loss is None or target is None:
            total_loss = self.impl_weight * fuzzy_loss
        elif self.multiply_with_base_loss:
            total_loss = base_loss * (1 + self.impl_weight * fuzzy_loss)
        else:
            total_loss = base_loss + self.impl_weight * fuzzy_loss
        return total_loss.mean(), loss_components

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
        elif self.fuzzy_implication in ["binary", "b"]:
            individual_loss = torch.where(l <= r, 0, 1).to(dtype=l.dtype)
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
        if self.no_grads:
            impl_loss = impl_loss.detach()
        loss_components["unweighted_implication_loss"] = unweighted_impl_mean
        loss_components["weighted_implication_loss"] = weighted_impl_mean

        disj_loss, unweighted_disj_mean, weighted_disj_mean = (
            self._calculate_unaggregated_fuzzy_loss(
                pred,
                target,
                self.disjoint_weight,
                self.disjoint_filter_l,
                self.disjoint_filter_r,
                mode="disj",
                **kwargs,
            )
        )
        if self.no_grads:
            disj_loss = disj_loss.detach()
        loss_components["unweighted_disjointness_loss"] = unweighted_disj_mean
        loss_components["weighted_disjointness_loss"] = weighted_disj_mean

        if self.base_loss is None or target is None:
            total_loss = self.impl_weight * impl_loss + self.disjoint_weight * disj_loss
        elif self.multiply_with_base_loss:
            total_loss = base_loss * (
                1 + self.impl_weight * impl_loss + self.disjoint_weight * disj_loss
            )
        else:
            total_loss = (
                base_loss
                + self.impl_weight * impl_loss
                + self.disjoint_weight * disj_loss
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
    Build implication filter based on label names and hierarchy. Results in list of pairs (A,B) for each implication
    A->B (including indirect implications).

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


def _build_dense_filter(sparse_filter: torch.Tensor, n_labels: int) -> torch.Tensor:
    res = torch.zeros((n_labels, n_labels), dtype=torch.bool)
    for l, r in sparse_filter:
        res[l, r] = True
    return res


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
    dense = _build_dense_filter(dis_filter, len(label_names))
    dense_r = dense.transpose(0, 1)
    return dense, dense_r


if __name__ == "__main__":
    loss = DisjointLoss(
        os.path.join("data", "disjoint.csv"),
        ChEBIOver100(chebi_version=231),
        base_loss=BCEWeighted(),
        impl_loss_weight=1,
        disjoint_loss_weight=1,
    )
    random_preds = torch.randn(10, 997)
    random_labels = torch.randint(0, 2, (10, 997))
    for agg in ["sum", "max", "mean", "log-mean"]:
        loss.violations_per_cls_aggregator = agg
        l = loss(random_preds, random_labels)
        print(f"Loss with {agg} aggregation for random input:", l)

    # simplified example for ontology with 4 classes, A -> B, B -> C, D -> C, B and D disjoint
    loss.implication_filter_l = torch.tensor(
        [[0, 1, 1, 0], [0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 1, 0]]
    )
    loss.implication_filter_r = loss.implication_filter_l.transpose(0, 1)
    loss.disjoint_filter_l = torch.tensor(
        [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 1, 0, 0]]
    )
    loss.disjoint_filter_r = loss.disjoint_filter_l.transpose(0, 1)
    # expected result: first sample: moderately high loss for B disj D, otherwise low, second sample: high loss for A -> B (applied to A), otherwise low
    preds = torch.tensor([[0.1, 0.3, 0.7, 0.4], [0.5, 0.2, 0.9, 0.1]])
    labels = [[0, 1, 1, 0], [0, 0, 1, 1]]
    for agg in ["sum", "max", "mean", "log-mean"]:
        loss.violations_per_cls_aggregator = agg
        l = loss(preds, torch.tensor(labels))
        print(f"Loss with {agg} aggregation for simple input:", l)
