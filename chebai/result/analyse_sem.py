import gc
import os
import traceback
from datetime import datetime
from typing import List, LiteralString, Optional, Tuple

import pandas as pd
import torch
import wandb
from torchmetrics.functional.classification import (
    multilabel_auroc,
    multilabel_average_precision,
    multilabel_f1_score,
)

from chebai.loss.semantic import DisjointLoss
from chebai.models import Electra
from chebai.preprocessing.datasets.base import _DynamicDataset
from chebai.preprocessing.datasets.chebi import ChEBIOver100
from chebai.result.utils import (
    evaluate_model,
    get_checkpoint_from_wandb,
    load_results_from_buffer,
)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def binary(left, right):
    return torch.logical_and(left > 0.5, right > 0.5)


def strict(left, right):
    return left + right > 1


def weak(left, right):
    return left + right > 1.01


def product(left, right):
    return left * right


def lukasiewicz(left, right):
    return torch.relu(left + right - 1)


def apply_metric(metric, left, right):
    return torch.sum(metric(left, right), dim=0)


ALL_CONSISTENCY_METRICS = [product, lukasiewicz, weak, strict, binary]


def _filter_to_one_hot(preds, idx_filter):
    """Takes list of indices (e.g. [1, 3, 0]) and returns a one-hot filter with these indices
    (e.g. [[0,1,0,0], [0,0,0,1], [1,0,0,0]])"""
    res = torch.zeros((len(idx_filter), preds.shape[1]), dtype=torch.bool)
    for i, idx in enumerate(idx_filter):
        res[i][idx] = True
    return res


def _sort_results_by_label(n_labels, results, filter):
    by_label = torch.zeros(n_labels, device=DEVICE, dtype=torch.int)
    for r, filter_l in zip(results, filter):
        by_label[filter_l] += r
    return by_label


def get_best_epoch(run):
    files = run.files()
    best_ep = None
    best_micro_f1 = 0
    for file in files:
        if file.name.startswith("checkpoints/best_epoch"):
            micro_f1 = float(file.name.split("=")[-1][:-5])
            if micro_f1 > best_micro_f1 or best_ep is None:
                best_ep = int(file.name.split("=")[1].split("_")[0])
                best_micro_f1 = micro_f1
    if best_ep is None:
        raise Exception(f"Could not find any 'best' checkpoint for run {run.id}")
    else:
        print(f"Best epoch for run {run.id}: {best_ep}")
    return best_ep


def download_model_from_wandb(
    run_id, base_dir=os.path.join("logs", "downloaded_ckpts")
):
    api = wandb.Api()
    run = api.run(f"chebai/chebai/{run_id}")
    epoch = get_best_epoch(run)
    return (
        get_checkpoint_from_wandb(epoch, run, root=base_dir),
        epoch,
    )


def load_preds_labels(
    ckpt_path: LiteralString, data_module, data_subset_key="test", buffer_dir=None
):
    if buffer_dir is None:
        buffer_dir = os.path.join(
            "results_buffer",
            *ckpt_path.split(os.path.sep)[-2:],
            f"{data_module.__class__.__name__}_{data_subset_key}",
        )
    model = Electra.load_from_checkpoint(ckpt_path, map_location="cuda:0", strict=False)
    print(
        f"Calculating predictions on {data_module.__class__.__name__} ({data_subset_key})..."
    )
    evaluate_model(
        model,
        data_module,
        buffer_dir=buffer_dir,
        # for chebi, use kinds, otherwise use file names
        filename=(
            data_subset_key if not isinstance(buffer_dir, _DynamicDataset) else None
        ),
        kind=data_subset_key,
        skip_existing_preds=True,
        batch_size=1,
    )
    return load_results_from_buffer(buffer_dir, device=torch.device("cpu"))


def get_label_names(data_module):
    if os.path.exists(os.path.join(data_module.processed_dir_main, "classes.txt")):
        with open(os.path.join(data_module.processed_dir_main, "classes.txt")) as fin:
            return [line.strip() for line in fin]
    print(
        f"Failed to retrieve label names, {os.path.join(data_module.processed_dir_main, 'classes.txt')} not found"
    )
    return None


def get_chebi_graph(data_module, label_names):
    if os.path.exists(os.path.join(data_module.raw_dir, "chebi.obo")):
        chebi_graph = data_module._extract_class_hierarchy(
            os.path.join(data_module.raw_dir, "chebi.obo")
        )
        if label_names is None:
            return chebi_graph
        return chebi_graph.subgraph([int(n) for n in label_names])
    print(
        f"Failed to retrieve ChEBI graph, {os.path.join(data_module.raw_dir, 'chebi.obo')} not found"
    )
    return None


def get_disjoint_groups(disjoint_files):
    if disjoint_files is None:
        disjoint_files = os.path.join("data", "chebi-disjoints.owl")
    disjoint_pairs, disjoint_groups = [], []
    for file in disjoint_files:
        if file.split(".")[-1] == "csv":
            disjoint_pairs += pd.read_csv(file, header=None).values.tolist()
        elif file.split(".")[-1] == "owl":
            with open(file, "r") as f:
                plaintext = f.read()
                segments = plaintext.split("<")
                disjoint_pairs = []
                left = None
                for seg in segments:
                    if seg.startswith("rdf:Description ") or seg.startswith(
                        "owl:Class"
                    ):
                        left = int(seg.split('rdf:about="&obo;CHEBI_')[1].split('"')[0])
                    elif seg.startswith("owl:disjointWith"):
                        right = int(
                            seg.split('rdf:resource="&obo;CHEBI_')[1].split('"')[0]
                        )
                        disjoint_pairs.append([left, right])

                disjoint_groups = []
                for seg in plaintext.split("<rdf:Description>"):
                    if "owl;AllDisjointClasses" in seg:
                        classes = seg.split('rdf:about="&obo;CHEBI_')[1:]
                        classes = [int(c.split('"')[0]) for c in classes]
                        disjoint_groups.append(classes)
        else:
            raise NotImplementedError(
                "Unsupported disjoint file format: " + file.split(".")[-1]
            )

    disjoint_all = disjoint_pairs + disjoint_groups
    # one disjointness is commented out in the owl-file
    # (the correct way would be to parse the owl file and notice the comment symbols, but for this case, it should work)
    if [22729, 51880] in disjoint_all:
        disjoint_all.remove([22729, 51880])
    # print(f"Found {len(disjoint_all)} disjoint groups")
    return disjoint_all


class PredictionSmoother:
    """Removes implication and disjointness violations from predictions"""

    def __init__(self, dataset, label_names=None, disjoint_files=None):
        self.chebi_graph = get_chebi_graph(dataset, None)
        self.set_label_names(label_names)
        self.disjoint_groups = get_disjoint_groups(disjoint_files)

    def set_label_names(self, label_names):
        if label_names is not None:
            self.label_names = [int(label) for label in label_names]
            chebi_subgraph = self.chebi_graph.subgraph(self.label_names)
            self.label_successors = torch.zeros(
                (len(self.label_names), len(self.label_names)), dtype=torch.bool
            )
            for i, label in enumerate(self.label_names):
                self.label_successors[i, i] = 1
                for p in chebi_subgraph.successors(label):
                    if p in self.label_names:
                        self.label_successors[i, self.label_names.index(p)] = 1
            self.label_successors = self.label_successors.unsqueeze(0)

    def __call__(self, preds):
        preds_sum_orig = torch.sum(preds)
        # step 1: apply implications: for each class, set prediction to max of itself and all successors
        preds = preds.unsqueeze(1)
        preds_masked_succ = torch.where(self.label_successors, preds, 0)
        preds = preds_masked_succ.max(dim=2).values
        if torch.sum(preds) != preds_sum_orig:
            print(f"Preds change (step 1): {torch.sum(preds) - preds_sum_orig}")
        preds_sum_orig = torch.sum(preds)
        # step 2: eliminate disjointness violations: for group of disjoint classes, set all except max to 0.49 (if it is not already lower)
        preds_bounded = torch.min(preds, torch.ones_like(preds) * 0.49)
        for disj_group in self.disjoint_groups:
            disj_group = [
                self.label_names.index(g) for g in disj_group if g in self.label_names
            ]
            if len(disj_group) > 1:
                old_preds = preds[:, disj_group]
                disj_max = torch.max(preds[:, disj_group], dim=1)
                for i, row in enumerate(preds):
                    for l_ in range(len(preds[i])):
                        if l_ in disj_group and l_ != disj_group[disj_max.indices[i]]:
                            preds[i, l_] = preds_bounded[i, l_]
                samples_changed = 0
                for i, row in enumerate(preds[:, disj_group]):
                    if any(r != o for r, o in zip(row, old_preds[i])):
                        samples_changed += 1
                if samples_changed != 0:
                    print(
                        f"disjointness group {[self.label_names[d] for d in disj_group]} changed {samples_changed} samples"
                    )
        if torch.sum(preds) != preds_sum_orig:
            print(f"Preds change (step 2): {torch.sum(preds) - preds_sum_orig}")
        preds_sum_orig = torch.sum(preds)
        # step 3: disjointness violation removal may have caused new implication inconsistencies -> set each prediction to min of predecessors
        preds = preds.unsqueeze(1)
        preds_masked_predec = torch.where(
            torch.transpose(self.label_successors, 1, 2), preds, 1
        )
        preds = preds_masked_predec.min(dim=2).values
        if torch.sum(preds) != preds_sum_orig:
            print(f"Preds change (step 3): {torch.sum(preds) - preds_sum_orig}")
        return preds


def _filter_to_dense(filter):
    filter_dense = []
    for i in range(filter.shape[0]):
        for j in range(filter.shape[1]):
            if filter[i, j] > 0:
                filter_dense.append([i, j])
    return torch.tensor(filter_dense)


def build_prediction_filter(data_module_labeled=None):
    if data_module_labeled is None:
        data_module_labeled = ChEBIOver100(chebi_version=231)
    # prepare filters
    print("Loading implication / disjointness filters...")
    dl = DisjointLoss(
        path_to_disjointness=os.path.join("data", "disjoint.csv"),
        data_extractor=data_module_labeled,
    )
    impl = _filter_to_dense(dl.implication_filter_l)
    disj = _filter_to_dense(dl.disjoint_filter_l)

    return [
        (impl[:, 0], impl[:, 1], "impl"),
        (disj[:, 0], disj[:, 1], "disj"),
    ]


def run_consistency_metrics(
    preds,
    consistency_filters,
    data_module_labeled=None,  # use labels from this dataset for violations
    violation_metrics=None,
    verbose_violation_output=False,
    save_details_to=None,
):
    """Calculates all semantic metrics for given predictions (and supervised metrics if labels are provided)"""
    if violation_metrics is None:
        violation_metrics = ALL_CONSISTENCY_METRICS
    if data_module_labeled is None:
        data_module_labeled = ChEBIOver100(chebi_version=231)
    if save_details_to is not None:
        os.makedirs(save_details_to, exist_ok=True)

    preds.to("cpu")

    n_labels = preds.size(1)
    print(f"Found {preds.shape[0]} predictions ({n_labels} classes)")

    results = {}

    for dl_filter_l, dl_filter_r, filter_type in consistency_filters:
        l_preds = preds[:, dl_filter_l]
        r_preds = preds[:, dl_filter_r]
        for i, metric in enumerate(violation_metrics):
            if metric.__name__ not in results:
                results[metric.__name__] = {}
            print(f"Calculating metrics {metric.__name__} on {filter_type}")

            metric_results = {}
            metric_results["tps"] = torch.sum(
                torch.stack(
                    [
                        apply_metric(
                            metric,
                            l_preds[i : i + 1000],
                            (
                                r_preds[i : i + 1000]
                                if filter_type == "impl"
                                else 1 - r_preds[i : i + 1000]
                            ),
                        )
                        for i in range(0, r_preds.shape[0], 1000)
                    ]
                ),
                dim=0,
            )
            metric_results["fns"] = torch.sum(
                torch.stack(
                    [
                        apply_metric(
                            metric,
                            l_preds[i : i + 1000],
                            (
                                1 - r_preds[i : i + 1000]
                                if filter_type == "impl"
                                else r_preds[i : i + 1000]
                            ),
                        )
                        for i in range(0, r_preds.shape[0], 1000)
                    ]
                ),
                dim=0,
            )
            if verbose_violation_output:
                label_names = get_label_names(data_module_labeled)
                print(
                    f"Found {torch.sum(metric_results['fns'])} {filter_type}-violations"
                )
                # for k, fn_cls in enumerate(metric_results['fns']):
                #    if fn_cls > 0:
                #        print(f"\tThereof, {fn_cls.item()} belong to class {label_names[k]}")
                if torch.sum(metric_results["fns"]) != 0:
                    fns = metric(
                        l_preds, 1 - r_preds if filter_type == "impl" else r_preds
                    )
                    print(fns.shape)
                    for k, row in enumerate(fns):
                        if torch.sum(row) != 0:
                            print(f"{torch.sum(row)} violations for entity {k}")
                            for j, violation in enumerate(row):
                                if violation > 0:
                                    print(
                                        f"\tviolated ({label_names[dl_filter_l[j]]} -> {preds[k, dl_filter_l[j]]:.3f}"
                                        f", {label_names[dl_filter_r[j]]} -> {preds[k, dl_filter_r[j]]:.3f})"
                                    )

            m_l_agg = {}
            for key, value in metric_results.items():
                m_l_agg[key] = _sort_results_by_label(
                    n_labels,
                    value,
                    dl_filter_l,
                )
            m_r_agg = {}
            for key, value in metric_results.items():
                m_r_agg[key] = _sort_results_by_label(
                    n_labels,
                    value,
                    dl_filter_r,
                )

            if save_details_to is not None:
                with open(
                    os.path.join(
                        save_details_to, f"{metric.__name__}_{filter_type}_all.csv"
                    ),
                    "w+",
                ) as f:
                    f.write("left,right,tps,fns\n")
                    for left, right, tps, fns in zip(
                        dl_filter_l,
                        dl_filter_r,
                        metric_results["tps"],
                        metric_results["fns"],
                    ):
                        f.write(f"{left},{right},{tps},{fns}\n")
                with open(
                    os.path.join(
                        save_details_to, f"{metric.__name__}_{filter_type}_l.csv"
                    ),
                    "w+",
                ) as f:
                    f.write("left,tps,fns\n")
                    for left in range(n_labels):
                        f.write(
                            f"{left},{m_l_agg['tps'][left].item()},{m_l_agg['fns'][left].item()}\n"
                        )
                with open(
                    os.path.join(
                        save_details_to, f"{metric.__name__}_{filter_type}_r.csv"
                    ),
                    "w+",
                ) as f:
                    f.write("right,tps,fns\n")
                    for right in range(n_labels):
                        f.write(
                            f"{right},{m_r_agg['tps'][right].item()},{m_r_agg['fns'][right].item()}\n"
                        )
                print(
                    f"Saved unaggregated consistency metrics ({metric.__name__}, {filter_type}) to {save_details_to}"
                )

            fns_sum = torch.sum(metric_results["fns"]).item()
            results[metric.__name__][f"micro-fnr-{filter_type}"] = (
                0
                if fns_sum == 0
                else (
                    torch.sum(metric_results["fns"])
                    / (
                        torch.sum(metric_results["tps"])
                        + torch.sum(metric_results["fns"])
                    )
                ).item()
            )
            macro_fnr_l = m_l_agg["fns"] / (m_l_agg["tps"] + m_l_agg["fns"])
            results[metric.__name__][f"lmacro-fnr-{filter_type}"] = (
                0
                if fns_sum == 0
                else torch.mean(macro_fnr_l[~macro_fnr_l.isnan()]).item()
            )
            macro_fnr_r = m_r_agg["fns"] / (m_r_agg["tps"] + m_r_agg["fns"])
            results[metric.__name__][f"rmacro-fnr-{filter_type}"] = (
                0
                if fns_sum == 0
                else torch.mean(macro_fnr_r[~macro_fnr_r.isnan()]).item()
            )
            results[metric.__name__][f"fn-sum-{filter_type}"] = torch.sum(
                metric_results["fns"]
            ).item()
            results[metric.__name__][f"tp-sum-{filter_type}"] = torch.sum(
                metric_results["tps"]
            ).item()

            del metric_results
            del m_l_agg
            del m_r_agg

            gc.collect()
        del l_preds
        del r_preds
        gc.collect()

    return results


def run_supervised_metrics(preds, labels, save_details_to=None):
    # calculate supervised metrics
    results = {}
    if labels is not None:
        results["micro-f1"] = multilabel_f1_score(
            preds, labels, num_labels=preds.size(1), average="micro"
        ).item()
        results["macro-f1"] = multilabel_f1_score(
            preds, labels, num_labels=preds.size(1), average="macro"
        ).item()
        results["micro-roc-auc"] = multilabel_auroc(
            preds, labels, num_labels=preds.size(1), average="micro"
        ).item()
        results["macro-roc-auc"] = multilabel_auroc(
            preds, labels, num_labels=preds.size(1), average="macro"
        ).item()

        results["micro-ap"] = multilabel_average_precision(
            preds, labels, num_labels=preds.size(1), average="micro"
        ).item()
        results["macro-ap"] = multilabel_average_precision(
            preds, labels, num_labels=preds.size(1), average="macro"
        ).item()

        if save_details_to is not None:
            f1_by_label = multilabel_f1_score(
                preds, labels, num_labels=preds.size(1), average=None
            )
            roc_by_label = multilabel_auroc(
                preds, labels, num_labels=preds.size(1), average=None
            )
            ap_by_label = multilabel_average_precision(
                preds, labels, num_labels=preds.size(1), average=None
            )
            with open(os.path.join(save_details_to, "supervised.csv"), "w+") as f:
                f.write("label,f1,roc-auc,ap\n")
                for right in range(preds.size(1)):
                    f.write(
                        f"{right},{f1_by_label[right].item()},{roc_by_label[right].item()},{ap_by_label[right].item()}\n"
                    )
            print(f"Saved class-wise supervised metrics to {save_details_to}")

    del preds
    del labels
    gc.collect()
    return results


# run predictions / metrics calculations for semantic loss paper runs (NeSy 2024 submission)
def run_semloss_eval():
    # runs from wandb
    non_wandb_runs = []
    api = wandb.Api()
    runs = api.runs("chebai/chebai", filters={"tags": "eval_semloss_paper"})
    print(f"Found {len(runs)} tagged wandb runs")
    # ids_wandb = [run.id for run in runs]

    # ids used in the NeSy submission
    prod = ["tk15yznc", "uke62a8m", "w0h3zr5s"]
    xu19 = ["5ko8knb4", "061fd85t", "r50ioujs"]
    prod_mixed = ["hk8555ff", "e0lxw8py", "lig23cmg"]
    luka = ["0c0s48nh", "lfg384bp", "qeghvubh"]
    baseline = ["i4wtz1k4", "zd020wkv", "rc1q3t49"]
    prodk2 = ["ng3usn0p", "rp0wwzjv", "8fma1q7r"]
    ids = baseline + prod + prodk2 + xu19 + luka + prod_mixed
    # ids = ids_wandb
    run_all(
        ids,
        non_wandb_runs,
        prediction_datasets=[(ChEBIOver100(chebi_version=231), "test")],
        consistency_metrics=[binary],
    )


def run_all(
    wandb_ids=None,
    local_ckpts: List[Tuple] = None,
    consistency_metrics: Optional[List[callable]] = None,
    prediction_datasets: List[Tuple] = None,
    remove_violations: bool = False,
    results_dir="_fuzzy_loss_eval",
    check_consistency_on=None,
    verbose_violation_output=False,
):
    if wandb_ids is None:
        wandb_ids = []
    if local_ckpts is None:
        local_ckpts = []
    if consistency_metrics is None:
        consistency_metrics = ALL_CONSISTENCY_METRICS
    if prediction_datasets is None:
        prediction_datasets = [
            (ChEBIOver100(chebi_version=231), "test"),
        ]
    if check_consistency_on is None:
        check_consistency_on = ChEBIOver100(chebi_version=231)

    if remove_violations:
        smooth_preds = PredictionSmoother(check_consistency_on)
    else:
        smooth_preds = lambda x: x  # noqa: E731

    timestamp = datetime.now().strftime("%y%m%d-%H%M%S")
    prediction_filters = build_prediction_filter(check_consistency_on)

    results_path_consistency = os.path.join(
        results_dir,
        f"consistency_metrics_{timestamp}{'_violations_removed' if remove_violations else ''}.csv",
    )
    consistency_keys = [
        "micro-fnr-impl",
        "lmacro-fnr-impl",
        "rmacro-fnr-impl",
        "fn-sum-impl",
        "tp-sum-impl",
        "micro-fnr-disj",
        "lmacro-fnr-disj",
        "rmacro-fnr-disj",
        "fn-sum-disj",
        "tp-sum-disj",
    ]
    with open(results_path_consistency, "x") as f:
        f.write(
            "run-id,epoch,datamodule,data_key,metric,"
            + ",".join(consistency_keys)
            + "\n"
        )
    results_path_supervised = os.path.join(
        results_dir,
        f"supervised_metrics_{timestamp}{'_violations_removed' if remove_violations else ''}.csv",
    )
    supervised_keys = [
        "micro-f1",
        "macro-f1",
        "micro-roc-auc",
        "macro-roc-auc",
        "micro-ap",
        "macro-ap",
    ]
    with open(results_path_supervised, "x") as f:
        f.write("run-id,epoch,datamodule,data_key," + ",".join(supervised_keys) + "\n")

    ckpts = [(run_name, ep, None) for run_name, ep in local_ckpts] + [
        (None, None, wandb_id) for wandb_id in wandb_ids
    ]

    for run_name, epoch, wandb_id in ckpts:
        try:
            ckpt_dir = os.path.join("logs", "downloaded_ckpts")
            # for wandb runs, use short id as name, otherwise use ckpt dir name
            if wandb_id is not None:
                run_name = wandb_id
                ckpt_path, epoch = download_model_from_wandb(run_name, ckpt_dir)
            else:
                ckpt_path = None
                for file in os.listdir(os.path.join(ckpt_dir, run_name)):
                    if f"epoch={epoch}_" in file or f"epoch={epoch}." in file:
                        ckpt_path = os.path.join(os.path.join(ckpt_dir, run_name, file))
                assert (
                    ckpt_path is not None
                ), f"Failed to find checkpoint for epoch {epoch} in {os.path.join(ckpt_dir, run_name)}"
            print(f"Starting run {run_name} (epoch {epoch})")

            for dataset, dataset_key in prediction_datasets:
                # copy data from legacy buffer dir if possible
                old_buffer_dir = os.path.join(
                    "results_buffer",
                    *ckpt_path.split(os.path.sep)[-2:],
                    f"{dataset.__class__.__name__}_{dataset_key}",
                )
                buffer_dir = os.path.join(
                    "results_buffer",
                    run_name,
                    f"epoch={epoch}",
                    f"{dataset.__class__.__name__}_{dataset_key}",
                )
                print("Checking for buffer dir", old_buffer_dir)
                if os.path.isdir(old_buffer_dir):
                    from distutils.dir_util import copy_tree, remove_tree

                    os.makedirs(buffer_dir, exist_ok=True)
                    copy_tree(old_buffer_dir, buffer_dir)
                    remove_tree(old_buffer_dir, dry_run=True)
                    print(f"Moved buffer from {old_buffer_dir} to {buffer_dir}")
                print(f"Using buffer_dir {buffer_dir}")
                preds, labels = load_preds_labels(
                    ckpt_path, dataset, dataset_key, buffer_dir
                )
                # identity function if remove_violations is False
                smooth_preds(preds)

                details_path = None  # os.path.join(
                #    results_dir,
                #    f"{run_name}_ep{epoch}_{dataset.__class__.__name__}_{dataset_key}",
                # )
                metrics_dict = run_consistency_metrics(
                    preds,
                    prediction_filters,
                    check_consistency_on,
                    consistency_metrics,
                    verbose_violation_output,
                    save_details_to=details_path,
                )
                with open(results_path_consistency, "a") as f:
                    for metric in metrics_dict:
                        values = metrics_dict[metric]
                        f.write(
                            f"{run_name},{epoch},{dataset.__class__.__name__},{dataset_key},{metric},"
                            f"{','.join([str(values[k]) for k in consistency_keys])}\n"
                        )
                print(
                    f"Consistency metrics have been written to {results_path_consistency}"
                )
                if labels is not None:
                    metrics_dict = run_supervised_metrics(
                        preds, labels, save_details_to=details_path
                    )
                    with open(results_path_supervised, "a") as f:
                        f.write(
                            f"{run_name},{epoch},{dataset.__class__.__name__},{dataset_key},"
                            f"{','.join([str(metrics_dict[k]) for k in supervised_keys])}\n"
                        )
                    print(
                        f"Supervised metrics have been written to {results_path_supervised}"
                    )
        except Exception as e:
            print(
                f"Error during run {wandb_id if wandb_id is not None else run_name}: {e}"
            )
            print(traceback.format_exc())


# follow-up to NeSy submission
def run_fuzzy_loss(tag="fuzzy_loss", skip_first_n=0):
    api = wandb.Api()
    runs = api.runs("chebai/chebai", filters={"tags": tag})
    print(f"Found {len(runs)} wandb runs tagged with '{tag}'")
    ids = [run.id for run in runs]
    chebi100 = ChEBIOver100(
        chebi_version=231,
        splits_file_path=os.path.join(
            "data", "chebi_v231", "ChEBI100", "fuzzy_loss_splits.csv"
        ),
    )
    local_ckpts = [][skip_first_n:]
    # pubchem_kmeans = PubChemKMeans()
    run_all(
        ids[max(0, skip_first_n - len(local_ckpts)) :],  # ids,
        local_ckpts,
        consistency_metrics=[binary],
        check_consistency_on=chebi100,
        prediction_datasets=[
            (chebi100, "test"),
            # (pubchem_kmeans, "cluster1_cutoff2k.pt"),
            # (pubchem_kmeans, "cluster2.pt"),
            # (pubchem_kmeans, "ten_from_each_cluster.pt"),
            # (pubchem_kmeans, "chebi_close.pt"),
        ],
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 2:
        run_fuzzy_loss(sys.argv[1], int(sys.argv[2]))
    elif len(sys.argv) > 1:
        run_fuzzy_loss(sys.argv[1])
    else:
        run_fuzzy_loss()
