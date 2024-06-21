import pandas as pd
import sys
import traceback
from datetime import datetime
from chebai.loss.semantic import DisjointLoss
from chebai.preprocessing.datasets.chebi import ChEBIOver100
from chebai.preprocessing.datasets.pubchem import Hazardous
import os
import torch
from torchmetrics.functional.classification import multilabel_auroc
from torchmetrics.functional.classification import multilabel_f1_score
import wandb
import gc
from typing import List, Union
from utils import *

DEVICE = "cpu"  # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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


def _filter_to_one_hot(preds, idx_filter):
    """Takes list of indices (e.g. [1, 3, 0]) and returns a one-hot filter with these indices
    (e.g. [[0,1,0,0], [0,0,0,1], [1,0,0,0]])"""
    res = torch.zeros((len(idx_filter), preds.shape[1]), dtype=torch.bool)
    for i, idx in enumerate(idx_filter):
        res[i][idx] = True
    return res


def _sort_results_by_label(n_labels, results, filter):
    by_label = torch.zeros(n_labels, device=DEVICE)
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
        raise Exception(f"Could not find any 'best' checkpoint for run {run.name}")
    else:
        print(f"Best epoch for run {run.name}: {best_ep}")
    return best_ep


def load_preds_labels_from_wandb(
    run,
    epoch,
    chebi_version,
    test_on_data_cls=ChEBIOver100,  # use data from this class
    kind="test",  # specify segment of test_on_data_cls
):
    data_module = test_on_data_cls(chebi_version=chebi_version)

    buffer_dir = os.path.join(
        "results_buffer",
        f"{run.name}_ep{epoch}",
        f"{data_module.__class__.__name__}_{kind}",
    )

    model = get_checkpoint_from_wandb(epoch, run, map_device_to="cuda:0")
    print(f"Calculating predictions...")
    evaluate_model(
        model,
        data_module,
        buffer_dir=buffer_dir,
        filename=f"{kind}.pt",
        skip_existing_preds=True,
    )
    preds, labels = load_results_from_buffer(buffer_dir, device=DEVICE)
    del model
    gc.collect()

    return preds, labels


def load_preds_labels_from_nonwandb(
    name, epoch, chebi_version, test_on_data_cls=ChEBIOver100, kind="test"
):
    data_module = test_on_data_cls(chebi_version=chebi_version)

    buffer_dir = os.path.join(
        "results_buffer",
        f"{name}_ep{epoch}",
        f"{data_module.__class__.__name__}_{kind}",
    )
    ckpt_path = None
    for file in os.listdir(os.path.join("logs", "downloaded_ckpts", name)):
        if file.startswith(f"best_epoch={epoch}"):
            ckpt_path = os.path.join(
                os.path.join("logs", "downloaded_ckpts", name, file)
            )
    assert (
        ckpt_path is not None
    ), f"Could not find ckpt for epoch {epoch} in directory {os.path.join('logs', 'downloaded_ckpts', name)}"
    model = Electra.load_from_checkpoint(ckpt_path, map_location="cuda:0", strict=False)
    print(f"Calculating predictions...")
    evaluate_model(
        model,
        data_module,
        buffer_dir=buffer_dir,
        filename=f"{kind}.pt",
        skip_existing_preds=True,
    )
    preds, labels = load_results_from_buffer(buffer_dir, device=DEVICE)
    del model
    gc.collect()

    return preds, labels


def get_label_names(data_module):
    if os.path.exists(os.path.join(data_module.raw_dir, "classes.txt")):
        with open(os.path.join(data_module.raw_dir, "classes.txt")) as fin:
            return [int(line.strip()) for line in fin]
    return None


def get_chebi_graph(data_module, label_names):
    if os.path.exists(os.path.join(data_module.raw_dir, "chebi.obo")):
        chebi_graph = data_module.extract_class_hierarchy(
            os.path.join(data_module.raw_dir, "chebi.obo")
        )
        return chebi_graph.subgraph(label_names)
    return None


def get_disjoint_groups():
    disjoints_owl_file = os.path.join("data", "chebi-disjoints.owl")
    with open(disjoints_owl_file, "r") as f:
        plaintext = f.read()
        segments = plaintext.split("<")
        disjoint_pairs = []
        left = None
        for seg in segments:
            if seg.startswith("rdf:Description ") or seg.startswith("owl:Class"):
                left = int(seg.split('rdf:about="&obo;CHEBI_')[1].split('"')[0])
            elif seg.startswith("owl:disjointWith"):
                right = int(seg.split('rdf:resource="&obo;CHEBI_')[1].split('"')[0])
                disjoint_pairs.append([left, right])

        disjoint_groups = []
        for seg in plaintext.split("<rdf:Description>"):
            if "owl;AllDisjointClasses" in seg:
                classes = seg.split('rdf:about="&obo;CHEBI_')[1:]
                classes = [int(c.split('"')[0]) for c in classes]
                disjoint_groups.append(classes)
    disjoint_all = disjoint_pairs + disjoint_groups
    # one disjointness is commented out in the owl-file
    # (the correct way would be to parse the owl file and notice the comment symbols, but for this case, it should work)
    disjoint_all.remove([22729, 51880])
    print(f"Found {len(disjoint_all)} disjoint groups")
    return disjoint_all


def smooth_preds(preds, label_names, chebi_graph, disjoint_groups):
    preds_sum_orig = torch.sum(preds)
    print(f"Preds sum: {preds_sum_orig}")
    # eliminate implication violations by setting each prediction to maximum of its successors
    for i, label in enumerate(label_names):
        succs = [label_names.index(p) for p in chebi_graph.successors(label)] + [i]
        if len(succs) > 0:
            preds[:, i] = torch.max(preds[:, succs], dim=1).values
    print(f"Preds change (step 1): {torch.sum(preds) - preds_sum_orig}")
    preds_sum_orig = torch.sum(preds)
    # step 2: eliminate disjointness violations: for group of disjoint classes, set all except max to 0.49 (if it is not already lower)
    preds_bounded = torch.min(preds, torch.ones_like(preds) * 0.49)
    for disj_group in disjoint_groups:
        disj_group = [label_names.index(g) for g in disj_group if g in label_names]
        if len(disj_group) > 1:
            old_preds = preds[:, disj_group]
            disj_max = torch.max(preds[:, disj_group], dim=1)
            for i, row in enumerate(preds):
                for l in range(len(preds[i])):
                    if l in disj_group and l != disj_group[disj_max.indices[i]]:
                        preds[i, l] = preds_bounded[i, l]
            samples_changed = 0
            for i, row in enumerate(preds[:, disj_group]):
                if any(r != o for r, o in zip(row, old_preds[i])):
                    samples_changed += 1
            if samples_changed != 0:
                print(
                    f"disjointness group {[label_names[d] for d in disj_group]} changed {samples_changed} samples"
                )
    print(
        f"Preds change after disjointness (step 2): {torch.sum(preds) - preds_sum_orig}"
    )
    preds_sum_orig = torch.sum(preds)
    # step 3: disjointness violation removal may have caused new implication inconsistencies -> set each prediction to min of predecessors
    for i, label in enumerate(label_names):
        predecessors = [i] + [
            label_names.index(p) for p in chebi_graph.predecessors(label)
        ]
        lowest_predecessors = torch.min(preds[:, predecessors], dim=1)
        preds[:, i] = lowest_predecessors.values
        for idx_idx, idx in enumerate(lowest_predecessors.indices):
            if idx > 0:
                print(
                    f"class {label}: changed prediction of sample {idx_idx} to value of class "
                    f"{label_names[predecessors[idx]]} ({preds[idx_idx, i].item():.2f})"
                )
        if torch.sum(preds) != preds_sum_orig:
            print(
                f"Preds change (step 3) for {label}: {torch.sum(preds) - preds_sum_orig}"
            )
            preds_sum_orig = torch.sum(preds)
    return preds


def analyse_run(
    preds,
    labels,
    df_hyperparams,  # parameters that are the independent of the semantic loss function used
    labeled_data_cls=ChEBIOver100,  # use labels from this dataset for violations
    chebi_version=231,
    results_path=os.path.join("_semantic", "eval_results.csv"),
    violation_metrics: Union[str, List[callable]] = "all",
    verbose_violation_output=False,
):
    """Calculates all semantic metrics for given predictions (and supervised metrics if labels are provided),
    saves results to csv"""
    if violation_metrics == "all":
        violation_metrics = [product, lukasiewicz, weak, strict, binary]
    data_module_labeled = labeled_data_cls(chebi_version=chebi_version)
    n_labels = preds.size(1)
    print(f"Found {preds.shape[0]} predictions ({n_labels} classes)")

    df_new = []

    # prepare filters
    print(f"Loading & rescaling implication / disjointness filters...")
    dl = DisjointLoss(
        path_to_disjointness=os.path.join("data", "disjoint.csv"),
        data_extractor=data_module_labeled,
    )
    for dl_filter_l, dl_filter_r, filter_type in [
        (dl.implication_filter_l, dl.implication_filter_r, "impl"),
        (dl.disjoint_filter_l, dl.disjoint_filter_r, "disj"),
    ]:
        # prepare predictions
        n_loss_terms = dl_filter_l.shape[0]
        preds_exp = preds.unsqueeze(2).expand((-1, -1, n_loss_terms)).swapaxes(1, 2)
        l_preds = preds_exp[:, _filter_to_one_hot(preds, dl_filter_l)]
        r_preds = preds_exp[:, _filter_to_one_hot(preds, dl_filter_r)]
        del preds_exp
        gc.collect()

        for i, metric in enumerate(violation_metrics):
            if filter_type == "impl":
                df_new.append(df_hyperparams.copy())
                df_new[-1]["metric"] = metric.__name__
            print(
                f"Calculating metric {metric.__name__ if metric is not None else 'supervised'} on {filter_type}"
            )

            m = {}
            m["tps"] = apply_metric(
                metric, l_preds, r_preds if filter_type == "impl" else 1 - r_preds
            )
            m["fns"] = apply_metric(
                metric, l_preds, 1 - r_preds if filter_type == "impl" else r_preds
            )
            if verbose_violation_output:
                label_names = get_label_names(data_module_labeled)
                print(f"Found {torch.sum(m['fns'])} {filter_type}-violations")
                # for k, fn_cls in enumerate(m['fns']):
                #    if fn_cls > 0:
                #        print(f"\tThereof, {fn_cls.item()} belong to class {label_names[k]}")
                if torch.sum(m["fns"]) != 0:
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

            m_cls = {}
            for key, value in m.items():
                m_cls[key] = _sort_results_by_label(
                    n_labels,
                    value,
                    (dl_filter_l),
                )

            df_new[i][f"micro-sem-recall-{filter_type}"] = (
                torch.sum(m["tps"]) / (torch.sum(m[f"tps"]) + torch.sum(m[f"fns"]))
            ).item()
            macro_recall = m_cls[f"tps"] / (m_cls[f"tps"] + m_cls[f"fns"])
            df_new[i][f"macro-sem-recall-{filter_type}"] = torch.mean(
                macro_recall[~macro_recall.isnan()]
            ).item()
            df_new[i][f"fn-sum-{filter_type}"] = torch.sum(m["fns"]).item()
            df_new[i][f"tp-sum-{filter_type}"] = torch.sum(m["tps"]).item()

            del m
            del m_cls

            gc.collect()
            df_new[i] = pd.DataFrame(df_new[i], index=[0])
        del l_preds
        del r_preds
        gc.collect()

    # calculate supervised metrics
    if labels is not None:
        df_supervised = df_hyperparams.copy()
        df_supervised["micro-f1"] = multilabel_f1_score(
            preds, labels, num_labels=preds.size(1), average="micro"
        ).item()
        df_supervised["macro-f1"] = multilabel_f1_score(
            preds, labels, num_labels=preds.size(1), average="macro"
        ).item()
        df_supervised["micro-roc-auc"] = multilabel_auroc(
            preds, labels, num_labels=preds.size(1), average="micro"
        ).item()
        df_supervised["macro-roc-auc"] = multilabel_auroc(
            preds, labels, num_labels=preds.size(1), average="macro"
        ).item()

        df_new.append(pd.DataFrame(df_supervised, index=[0]))

    if os.path.exists(results_path):
        df_previous = pd.read_csv(results_path)
    else:
        df_previous = None
    if df_previous is not None:
        df_new = [df_previous] + df_new
        del df_previous

    df_new = pd.concat(df_new, ignore_index=True)
    print(f"Saving results to {results_path}")
    df_new.to_csv(results_path, index=False)

    del df_new
    del preds
    del labels
    del dl
    gc.collect()


def run_all(
    run_ids,
    datasets=None,
    chebi_version=231,
    skip_analyse=False,
    skip_preds=False,
    nonwandb_runs=None,
    violation_metrics="all",
    remove_violations=False,
):
    # evaluate a list of runs on Hazardous and ChEBIOver100 datasets
    if datasets is None:
        datasets = [(Hazardous, "all"), (ChEBIOver100, "test")]
    timestamp = datetime.now().strftime("%y%m%d-%H%M")
    results_path = os.path.join(
        "_semloss_eval",
        f"semloss_results_pc-dis-200k_{timestamp}{'_violations_removed' if remove_violations else ''}.csv",
    )
    label_names = get_label_names(ChEBIOver100(chebi_version=chebi_version))
    chebi_graph = get_chebi_graph(
        ChEBIOver100(chebi_version=chebi_version), label_names
    )
    disjoint_groups = get_disjoint_groups()

    api = wandb.Api()
    for run_id in run_ids:
        try:
            run = api.run(f"chebai/chebai/{run_id}")
            epoch = get_best_epoch(run)
            for test_on, kind in datasets:
                df = {
                    "run-id": run_id,
                    "epoch": int(epoch),
                    "kind": kind,
                    "data_module": test_on.__name__,
                    "chebi_version": chebi_version,
                }
                buffer_dir_smoothed = os.path.join(
                    "results_buffer",
                    "smoothed3step",
                    f"{run.name}_ep{epoch}",
                    f"{test_on.__name__}_{kind}",
                )
                if remove_violations and os.path.exists(
                    os.path.join(buffer_dir_smoothed, "preds000.pt")
                ):
                    preds = torch.load(
                        os.path.join(buffer_dir_smoothed, "preds000.pt"), DEVICE
                    )
                    labels = None
                else:
                    if not skip_preds:
                        preds, labels = load_preds_labels_from_wandb(
                            run, epoch, chebi_version, test_on, kind
                        )
                    else:
                        buffer_dir = os.path.join(
                            "results_buffer",
                            f"{run.name}_ep{epoch}",
                            f"{test_on.__name__}_{kind}",
                        )
                        preds, labels = load_results_from_buffer(
                            buffer_dir, device=DEVICE
                        )
                        assert (
                            preds is not None
                        ), f"Did not find predictions in dir {buffer_dir}"
                        if remove_violations:
                            preds = smooth_preds(
                                preds, label_names, chebi_graph, disjoint_groups
                            )
                            buffer_dir_smoothed = os.path.join(
                                "results_buffer",
                                "smoothed3step",
                                f"{run.name}_ep{epoch}",
                                f"{test_on.__name__}_{kind}",
                            )
                            os.makedirs(buffer_dir_smoothed, exist_ok=True)
                            torch.save(
                                preds, os.path.join(buffer_dir_smoothed, "preds000.pt")
                            )
                if not skip_analyse:
                    print(
                        f"Calculating metrics for run {run.name} on {test_on.__name__} ({kind})"
                    )
                    analyse_run(
                        preds,
                        labels,
                        df_hyperparams=df,
                        chebi_version=chebi_version,
                        results_path=results_path,
                        violation_metrics=violation_metrics,
                        verbose_violation_output=True,
                    )
        except Exception as e:
            print(f"Failed for run {run_id}: {e}")
            print(traceback.format_exc())

    if nonwandb_runs:
        for run_name, epoch in nonwandb_runs:
            try:
                for test_on, kind in datasets:
                    df = {
                        "run-id": run_name,
                        "epoch": int(epoch),
                        "kind": kind,
                        "data_module": test_on.__name__,
                        "chebi_version": chebi_version,
                    }
                    if not skip_preds:
                        preds, labels = load_preds_labels_from_nonwandb(
                            run_name, epoch, chebi_version, test_on, kind
                        )
                    else:
                        buffer_dir = os.path.join(
                            "results_buffer",
                            f"{run_name}_ep{epoch}",
                            f"{test_on.__name__}_{kind}",
                        )
                        preds, labels = load_results_from_buffer(
                            buffer_dir, device=DEVICE
                        )
                        assert (
                            preds is not None
                        ), f"Did not find predictions in dir {buffer_dir}"
                        if remove_violations:
                            preds = smooth_preds(
                                preds, label_names, chebi_graph, disjoint_groups
                            )
                    if not skip_analyse:
                        print(
                            f"Calculating metrics for run {run_name} on {test_on.__name__} ({kind})"
                        )
                        analyse_run(
                            preds,
                            labels,
                            df_hyperparams=df,
                            chebi_version=chebi_version,
                            results_path=results_path,
                            violation_metrics=violation_metrics,
                        )
            except Exception as e:
                print(f"Failed for run {run_name}: {e}")
                print(traceback.format_exc())


# run predictions / metrics calculations for semantic loss paper runs (NeSy 2024 submission)
def run_semloss_eval(mode="eval"):
    non_wandb_runs = []
    if mode == "preds":
        api = wandb.Api()
        runs = api.runs("chebai/chebai", filters={"tags": "eval_semloss_paper"})
        print(f"Found {len(runs)} tagged wandb runs")
        ids = [run.id for run in runs]
        run_all(ids, skip_analyse=True, nonwandb_runs=non_wandb_runs)

    if mode == "eval":
        prod = [
            "tk15yznc",
            "uke62a8m",
            "w0h3zr5s",
        ]
        xu19 = [
            "5ko8knb4",
            "061fd85t",
            "r50ioujs",
        ]
        prod_mixed = [
            "hk8555ff",
            "e0lxw8py",
            "lig23cmg",
        ]
        luka = [
            "0c0s48nh",
            "lfg384bp",
            "qeghvubh",
        ]
        baseline = ["i4wtz1k4", "zd020wkv", "rc1q3t49"]
        prodk2 = ["ng3usn0p", "rp0wwzjv", "8fma1q7r"]
        ids = baseline + prod + prodk2 + xu19 + luka + prod_mixed
        run_all(
            ids,
            skip_preds=True,
            nonwandb_runs=non_wandb_runs,
            datasets=[(ChEBIOver100, "test")],
            violation_metrics=[binary],
            remove_violations=True,
        )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_semloss_eval(sys.argv[1])
    else:
        run_semloss_eval()
