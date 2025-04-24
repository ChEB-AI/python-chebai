import csv
import math
import os
import pickle
from typing import Iterable, List, Literal, Union

import torch

from chebai.loss.bce_weighted import BCEWeighted
from chebai.preprocessing.datasets import XYBaseDataModule
from chebai.preprocessing.datasets.chebi import (
    ChEBIOver100,
    ChEBIOver100Parthood,
    _ChEBIDataExtractor,
)
from chebai.preprocessing.datasets.pubchem import LabeledUnlabeledMixed


class FuzzyLossTerm:

    def __init__(self, weight: float = 1):
        self.weight = weight

    @property
    def name(self) -> str:
        raise NotImplementedError()

    def get_axiom_filter(
        self, data_extractor: Union[_ChEBIDataExtractor, LabeledUnlabeledMixed]
    ):
        raise NotImplementedError()

    def process_preds_l(self, preds, **kwargs):
        return preds

    def process_preds_r(self, preds, **kwargs):
        return preds


class ImplicationLossTerm(FuzzyLossTerm):
    def __init__(self, weight: float = 0.1):
        super().__init__(weight)

    @property
    def name(self) -> str:
        return "implication"

    def get_axiom_filter(
        self, data_extractor: Union[_ChEBIDataExtractor, LabeledUnlabeledMixed]
    ):
        label_names = _load_label_names(
            os.path.join(data_extractor.processed_dir_main, "classes.txt")
        )
        hierarchy = _load_implications(
            os.path.join(data_extractor.raw_dir, "chebi.obo"), data_extractor
        )
        implication_filter_dense = _build_dense_filter(
            _build_implication_filter(label_names, hierarchy),
            len(label_names),
            len(label_names),
        )
        implication_filter_l = implication_filter_dense
        implication_filter_r = implication_filter_l.transpose(0, 1)
        return implication_filter_l, implication_filter_r


class DisjointLossTerm(FuzzyLossTerm):

    def __init__(self, path_to_disjointness: str, weight: float = 100):
        super().__init__(weight)
        self.path_to_disjointness = path_to_disjointness

    @property
    def name(self) -> str:
        return "disjointness"

    def get_axiom_filter(self, data_extractor):
        label_names = _load_label_names(
            os.path.join(data_extractor.processed_dir_main, "classes.txt")
        )
        hierarchy = _load_implications(
            os.path.join(data_extractor.raw_dir, "chebi.obo"), data_extractor
        )

        return _build_disjointness_filter(
            self.path_to_disjointness, label_names, hierarchy
        )

    def process_preds_r(self, preds, **kwargs):
        # for disjointness, we need to use 1-pred
        return 1 - preds


class ParthoodLossTerm(FuzzyLossTerm):
    """
    A ParthoodLossTerm uses parthood relations between molecule classes and functional groups.
    For instance, ChEBI has an axiom that _thiocarbonyl compound (CHEBI:50492) has part some thiocarbonyl group (CHEBI:30256)_
    If we predict a molecule as a thiocarbonyl compound that has **no** thiocarbonyl group, it gets a loss for that.

    As a subsumption: instead of $A subseteq B$ (which gets a loss for A=1 and B=0), we can use $A subseteq has part some G$.
    """

    def __init__(self, path_to_parthoods: str, path_to_parts: str, weight: float = 100):
        super().__init__(weight)
        self.path_to_parthoods = path_to_parthoods
        self.path_to_parts = path_to_parts

    @property
    def name(self) -> str:
        return "parthood"

    def get_axiom_filter(self, data_extractor):
        label_names = _load_label_names(
            os.path.join(data_extractor.processed_dir_main, "classes.txt")
        )
        part_names = _load_label_names(self.path_to_parts)

        with open(self.path_to_parthoods, "r") as f:
            cls_group_pairs = [[int(r.strip()) for r in row.split(",")] for row in f]
        cls_group_indices = [
            [label_names.index(r[0]), part_names.index(r[1])] for r in cls_group_pairs
        ]

        print(
            f"Using {len(cls_group_indices)} parthood axioms for loss, starting with {cls_group_indices[:5]}"
        )

        dense = _build_dense_filter(
            cls_group_indices, len(label_names), len(part_names)
        )
        dense_r = dense.transpose(0, 1)
        return dense, dense_r

    def process_preds_r(self, preds, **kwargs):
        assert (
            "parthoods" in kwargs
        ), "Parthood loss requires that information about the parts of molecules are given in the loss_kwargs"
        # for disjointness, we need to use 1-pred
        return kwargs["parthoods"]


class FuzzyLoss(torch.nn.Module):
    """
    Implication Loss module.

    Args:
        data_extractor _ChEBIDataExtractor: Data extractor for labels.
        base_loss (torch.nn.Module, optional): Base loss function. Defaults to None.
        fuzzy_implication (Literal["product", "lukasiewicz", "xu19"], optional): T-norm type. Defaults to "product".
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
        fuzzy_terms: List[FuzzyLossTerm] = None,
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
        pos_scalar: Union[int, float] = 1,
        pos_epsilon: float = 0.01,
        multiply_by_softmax: bool = False,
        use_sigmoidal_implication: bool = False,
        weight_epoch_dependent: Union[bool | tuple[int, int]] = False,
        start_at_epoch: int = 0,
        violations_per_cls_aggregator: Literal[
            "sum", "max", "mean", "log-sum", "log-max", "log-mean"
        ] = "sum",
        multiply_with_base_loss: bool = False,
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

        self.fuzzy_implication = fuzzy_implication
        self.pos_scalar = pos_scalar
        self.eps = pos_epsilon
        self.multiply_by_softmax = multiply_by_softmax
        self.use_sigmoidal_implication = use_sigmoidal_implication
        self.weight_epoch_dependent = weight_epoch_dependent
        self.start_at_epoch = start_at_epoch
        self.violations_per_cls_aggregator = violations_per_cls_aggregator
        self.multiply_with_base_loss = multiply_with_base_loss
        self.no_grads = no_grads

        self.fuzzy_terms = fuzzy_terms
        self.filters_l = {}
        self.filters_r = {}
        for fuzzy_term in fuzzy_terms:
            filter_l, filter_r = fuzzy_term.get_axiom_filter(data_extractor)
            self.filters_l[fuzzy_term.name] = filter_l
            self.filters_r[fuzzy_term.name] = filter_r

    def _calculate_unaggregated_fuzzy_loss(
        self,
        pred_left,
        pred_right,
        weight,
        filter_l,
        filter_r,
        **kwargs,
    ):
        """Calculates fuzzy loss for I(pred_left, pred_right)"""
        # for each batch, get all pairwise losses: [a1, a2, a3] -> [[a1*a1, a1*a2, a1*a3],[a2*a1,...],[a3*a1,...]]
        preds_expanded1 = pred_right.unsqueeze(1).expand(-1, pred_left.shape[1], -1)
        preds_expanded2 = pred_left.unsqueeze(2).expand(-1, -1, pred_right.shape[1])
        # filter by implication relations and labels

        filter_l = (
            filter_l.to(pred_left.device)
            .unsqueeze(0)
            .expand(pred_left.shape[0], -1, -1)
        )
        # filter_r = (
        #    filter_r.to(pred_right.device)
        #    .unsqueeze(0)
        #    .expand(pred_right.shape[0], -1, -1)
        # )
        all_implications = self._calculate_implication_loss(
            preds_expanded2, preds_expanded1
        )

        loss_impl_l = all_implications * filter_l
        # loss_impl_r = all_implications.transpose(1, 2) * filter_r
        loss_impl_sum = loss_impl_l  # + loss_impl_r

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
            return self.base_loss(labeled_input.float(), target.float())
        else:
            return torch.zeros(input.shape, device=input.device)

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
        term_loss_list = []
        for fuzzy_term in self.fuzzy_terms:
            pred_l = fuzzy_term.process_preds_l(pred, **kwargs)
            pred_r = fuzzy_term.process_preds_r(pred, **kwargs)
            term_loss, unweighted_term_mean, weighted_term_mean = (
                self._calculate_unaggregated_fuzzy_loss(
                    pred_l,
                    pred_r,
                    fuzzy_term.weight,
                    self.filters_l[fuzzy_term.name],
                    self.filters_r[fuzzy_term.name],
                    **kwargs,
                )
            )

            if self.no_grads:
                term_loss = term_loss.detach()
            term_loss_list.append(term_loss)
            loss_components[f"unweighted_{fuzzy_term.name}_loss"] = unweighted_term_mean
            loss_components[f"weighted_{fuzzy_term.name}_loss"] = weighted_term_mean

        if self.base_loss is None or target is None:
            total_loss = sum(term_loss_list)
        elif self.multiply_with_base_loss:
            total_loss = base_loss * sum(term_loss_list + [1])
        else:
            total_loss = sum([base_loss] + term_loss_list)

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


def _load_implications(path_to_chebi: str, data_extractor) -> dict:
    implication_cache_file = f"implications_{data_extractor.name}.cache"

    """
    Load class hierarchy implications.

    Args:
        path_to_chebi (str): Path to the ChEBI ontology file.

    Returns:
        dict: Loaded hierarchy of implications.
    """
    if os.path.isfile(implication_cache_file):
        with open(implication_cache_file, "rb") as fin:
            hierarchy = pickle.load(fin)
    else:
        hierarchy = data_extractor._extract_class_hierarchy(path_to_chebi)
        with open(implication_cache_file, "wb") as fout:
            pickle.dump(hierarchy, fout)
    return hierarchy


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


def _build_dense_filter(sparse_filter: Iterable, l_size: int, r_size) -> torch.Tensor:
    res = torch.zeros((l_size, r_size), dtype=torch.bool)
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
    dense = _build_dense_filter(dis_filter, len(label_names), len(label_names))
    dense_r = dense.transpose(0, 1)
    return dense, dense_r


if __name__ == "__main__":
    loss = FuzzyLoss(
        ChEBIOver100Parthood(
            chebi_version=231,
            splits_file_path=os.path.join(
                "data", "chebi_v231", "ChEBI100", "fuzzy_loss_splits.csv"
            ),
        ),
        fuzzy_terms=[
            # ImplicationLossTerm(weight=1),
            # DisjointLossTerm(os.path.join("data", "disjoint.csv"), weight=1),
            ParthoodLossTerm(
                os.path.join("parthood", "label_group_pairs.csv"),
                os.path.join("parthood", "groups2841.txt"),
                weight=1,
            )
        ],
        base_loss=BCEWeighted(),
    )
    processed = loss.data_extractor.load_processed_data(filename="data_parthoods.pt")
    parthoods = torch.stack(
        [processed[i]["additional_kwargs"]["parthoods"] for i in range(10)]
    )
    random_preds = torch.randn(10, 997)
    random_labels = torch.randint(0, 2, (10, 997))
    for agg in ["sum", "max", "mean", "log-mean"]:
        loss.violations_per_cls_aggregator = agg
        l = loss(random_preds, random_labels, parthoods=parthoods)
        print(f"Loss with {agg} aggregation for random input:", l)

    loss.filters_l["parthood"] = torch.tensor([[0, 0, 1], [1, 1, 0]])
    loss.filters_r["parthood"] = loss.filters_l["parthood"].transpose(0, 1)
    preds = torch.tensor([[0, 100000], [100000, 0]])
    labels = [[0, 1], [1, 0]]
    parthoods = torch.tensor([[1, 1, 0], [0, 0, 1]])
    for agg in ["sum", "max", "mean", "log-mean"]:
        loss.violations_per_cls_aggregator = agg
        l = loss(preds, torch.tensor(labels), parthoods=parthoods)
        print(f"Loss with {agg} aggregation for simple input:", l)
