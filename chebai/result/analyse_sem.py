import pandas as pd
import sys

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
    best_val_loss = 0
    for file in files:
        if file.name.startswith("checkpoints/best_epoch"):
            val_loss = float(file.name.split("=")[2].split("_")[0])
            if val_loss < best_val_loss or best_ep is None:
                best_ep = int(file.name.split("=")[1].split("_")[0])
                best_val_loss = val_loss
    if best_ep is None:
        raise Exception("Could not find any 'best' checkpoint")
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

    model = get_checkpoint_from_wandb(epoch, run)
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


def analyse_run(
    preds,
    labels,
    df_hyperparams,  # parameters that are the independent of the semantic loss function used
    labeled_data_cls=ChEBIOver100,  # use labels from this dataset for violations
    chebi_version=231,
    results_path=os.path.join("_semantic", "eval_results.csv"),
):
    """Calculates all semantic metrics for given predictions (and supervised metrics if labels are provided),
    saves results to csv"""
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
        print(f"Calculating on {filter_type} loss")
        # prepare predictions
        n_loss_terms = dl_filter_l.shape[0]
        preds_exp = preds.unsqueeze(2).expand((-1, -1, n_loss_terms)).swapaxes(1, 2)
        l_preds = preds_exp[:, _filter_to_one_hot(preds, dl_filter_l)]
        r_preds = preds_exp[:, _filter_to_one_hot(preds, dl_filter_r)]
        del preds_exp
        gc.collect()

        for i, metric in enumerate([product, lukasiewicz, weak, strict, binary]):
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


def run_all(run_ids, datasets=None, chebi_version=231):
    # evaluate a list of runs on Hazardous and ChEBIOver100 datasets
    if datasets is None:
        datasets = [(Hazardous, "all"), (ChEBIOver100, "test")]
    timestamp = datetime.now().strftime("%y%m%d-%H%M")
    results_path = os.path.join(
        "_semloss_eval", f"semloss_results_pc-dis-200k_{timestamp}.csv"
    )

    for run_id in run_ids:
        api = wandb.Api()
        run = api.run(f"chebai/chebai/{run_id}")
        epoch = get_best_epoch(run)
        for test_on, kind in datasets:
            df = {
                "run-id": run_id,
                "epoch": int(epoch),
                "kind": kind,
                "data_module": test_on.__class__.__name__,
                "chebi_version": chebi_version,
            }
            preds, labels = load_preds_labels_from_wandb(
                run, epoch, chebi_version, test_on, kind
            )
            analyse_run(
                preds,
                labels,
                df_hyperparams=df,
                chebi_version=chebi_version,
                results_path=results_path,
            )