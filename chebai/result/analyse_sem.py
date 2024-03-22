import pandas as pd
import sys

from torch.fx.experimental.migrate_gradual_types.constraint import Prod

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


def _filter_to_one_hot(preds, filter):
    res = torch.zeros((len(filter), preds.shape[1]), dtype=torch.bool)
    for i, idx in enumerate(filter):
        res[i][idx] = True
    return res


def _sort_results_by_label(n_labels, results, filter):
    by_label = torch.zeros(n_labels, device=DEVICE)
    for r, filter_l in zip(results, filter):
        by_label[filter_l] += r
    return by_label


def analyse_run(
    run_id,
    epoch,
    chebi_version=231,
    labeled_data_cls=ChEBIOver100,
    test_on_data_cls=ChEBIOver100,
    kind="test",
    results_path=os.path.join("_semloss_eval", "semloss_results_pc-dis-200k.csv"),
    metric=None,
):
    data_module_labeled = labeled_data_cls(chebi_version=chebi_version)
    data_module = test_on_data_cls(chebi_version=chebi_version)
    api = wandb.Api()
    run = api.run(f"chebai/chebai/{run_id}")
    buffer_dir = os.path.join(
        "results_buffer",
        f"{run.name}_ep{epoch}",
        f"{data_module.__class__.__name__}_{kind}",
    )

    preds, labels = load_results_from_buffer(buffer_dir, device=DEVICE)
    if preds is None:
        model = get_checkpoint(epoch, run)
        print(f"Calculating predictions...")
        evaluate_model(
            model,
            data_module,
            buffer_dir=buffer_dir,
            filename=f"{kind}.pt",
            skip_existing_preds=True,
        )
        preds, labels = load_results_from_buffer(buffer_dir, device=DEVICE)

    print(f"Found {preds.shape[0]} predictions ({preds.shape[1]} classes)")

    # prepare filters
    print(f"Loading & rescaling implication / disjointness filters...")
    dl = DisjointLoss(
        path_to_disjointness=os.path.join("data", "disjoint.csv"),
        data_extractor=data_module_labeled,
    )
    implication_filter_l = _filter_to_one_hot(preds, dl.implication_filter_l)
    implication_filter_r = _filter_to_one_hot(preds, dl.implication_filter_r)
    disjoint_filter_l = _filter_to_one_hot(preds, dl.disjoint_filter_l)
    disjoint_filter_r = _filter_to_one_hot(preds, dl.disjoint_filter_r)

    print(f"Rescaling predictions...")
    # prepare predictions
    n_implications = dl.implication_filter_l.shape[0]
    n_disjointness = dl.disjoint_filter_l.shape[0]
    preds_exp = preds.unsqueeze(2).expand((-1, -1, max(n_implications, n_disjointness)))
    preds_exp = preds_exp.swapaxes(1, 2)
    l_impl_preds = preds_exp[:, :n_implications, :][:, implication_filter_l]
    r_impl_preds = preds_exp[:, :n_implications, :][:, implication_filter_r]
    l_disj_preds = preds_exp[:, :n_disjointness, :][:, disjoint_filter_l]
    r_disj_preds = preds_exp[:, :n_disjointness, :][:, disjoint_filter_r]

    df_previous = pd.read_csv(results_path)
    i = 0
    df_new = []

    for metric in [metric]:
        print(
            f"Calculating metric {metric.__name__ if metric is not None else 'supervised'}"
        )
        i += 1
        df = {}
        df["run-id"] = run_id
        df["epoch"] = int(epoch)
        df["kind"] = kind
        df["data_module"] = data_module.__class__.__name__
        df["chebi_version"] = int(data_module.chebi_version)
        if metric is None:
            df["micro-f1"] = multilabel_f1_score(
                preds, labels, num_labels=preds.size(1), average="micro"
            ).item()
            df["macro-f1"] = multilabel_f1_score(
                preds, labels, num_labels=preds.size(1), average="macro"
            ).item()
            df["micro-roc-auc"] = multilabel_auroc(
                preds, labels, num_labels=preds.size(1), average="micro"
            ).item()
            df["macro-roc-auc"] = multilabel_auroc(
                preds, labels, num_labels=preds.size(1), average="macro"
            ).item()

        else:
            m = {}
            m["impl_tps"] = apply_metric(metric, l_impl_preds, r_impl_preds)
            m["impl_fns"] = apply_metric(metric, l_impl_preds, 1 - r_impl_preds)
            m["disj_tps"] = apply_metric(metric, l_disj_preds, 1 - r_disj_preds)
            m["disj_fns"] = apply_metric(metric, l_disj_preds, r_disj_preds)
            m_cls = {}
            for key, value in m.items():
                m_cls[key] = _sort_results_by_label(
                    preds.shape[1],
                    value,
                    (
                        dl.implication_filter_l
                        if key.startswith("impl")
                        else dl.disjoint_filter_l
                    ),
                )

            df["metric"] = metric.__name__
            for imdi in ["impl", "disj"]:
                df[f"micro-sem-recall-{imdi}"] = (
                    torch.sum(m[f"{imdi}_tps"])
                    / (torch.sum(m[f"{imdi}_tps"]) + torch.sum(m[f"{imdi}_fns"]))
                ).item()
                macro_recall = m_cls[f"{imdi}_tps"] / (
                    m_cls[f"{imdi}_tps"] + m_cls[f"{imdi}_fns"]
                )
                df[f"macro-sem-recall-{imdi}"] = torch.mean(
                    macro_recall[~macro_recall.isnan()]
                ).item()

            del m
            del m_cls
            gc.collect()
        df_new.append(pd.DataFrame(df, index=[i]))
    df_new = pd.concat([df_previous] + df_new, ignore_index=True)
    print(f"Saving results to {results_path}")
    df_new.to_csv(results_path, index=False)


def run_all(run, epoch):
    for test_on in [ChEBIOver100, Hazardous]:
        for metric in [binary, strict, weak, None, product, lukasiewicz]:
            print(
                f"\nStarting {test_on.__name__} - {metric.__name__ if metric is not None else 'supervised'}"
            )
            if not (test_on == Hazardous and metric is None):
                try:
                    analyse_run(run, epoch, test_on_data_cls=test_on, metric=metric)
                except Exception as e:
                    print(
                        f"Failed for {test_on.__name__} with metric {metric.__name__ if metric is not None else 'supervised'}"
                    )


if __name__ == "__main__":
    if len(sys.argv) > 3 and sys.argv[3] == "all":
        run_all(sys.argv[1], sys.argv[2])
    else:
        test_on = (
            Hazardous
            if len(sys.argv) > 4 and sys.argv[4].startswith("haz")
            else ChEBIOver100
        )
        metric = None
        if len(sys.argv) > 3:
            if sys.argv[3].startswith("prod"):
                metric = product
            elif sys.argv[3].startswith("luka"):
                metric = lukasiewicz
            elif sys.argv[3].startswith("bin"):
                metric = binary
            elif sys.argv[3].startswith("strict"):
                metric = strict
            elif sys.argv[3].startswith("weak"):
                metric = weak
        analyse_run(sys.argv[1], sys.argv[2], test_on_data_cls=test_on, metric=metric)
