"""
For each of one or more local W&B run files (run-*.wandb), find the step
with the best validation macro-F1 score, print that score along with the
corresponding validation micro-F1 (and any other metrics logged at that
same step). At the end, print the average of the best macro-F1 (and its
corresponding metrics) across all the given files.

Usage:
    python find_best_f1.py run1.wandb run2.wandb run3.wandb
    python find_best_f1.py *.wandb --macro-metric val/macro_f1 --micro-metric val/micro_f1
    python find_best_f1.py *.wandb --max-epoch 200

If --macro-metric / --micro-metric aren't given, the script auto-detects
them per file (case-insensitive match on "f1"+"macro" / "f1"+"micro",
preferring keys that also mention "val"/"eval"/"test").
"""

import argparse
import json
import sys
from pathlib import Path
from statistics import mean

try:
    # Newer wandb versions (>=0.16 or so)
    from wandb.sdk.internal.datastore import DataStore
except ImportError:
    # Older wandb versions
    from wandb.old.datastore import DataStore

from wandb.proto import wandb_internal_pb2 as pb

INTERNAL_KEY_PREFIXES = ("_",)  # e.g. _step, _timestamp, _runtime


def iter_history_rows(wandb_path: str):
    """
    Yields dicts of {key: value} for every 'history' record logged in the run,
    parsed straight out of the binary .wandb file (no network / API calls).
    """
    ds = DataStore()
    ds.open_for_scan(wandb_path)

    while True:
        data = ds.scan_data()
        if data is None:
            break
        record = pb.Record()
        record.ParseFromString(data)

        if record.WhichOneof("record_type") == "history":
            row = {}
            for item in record.history.item:
                key = item.key if item.key else ".".join(item.nested_key)
                try:
                    val = json.loads(item.value_json)
                except Exception:
                    val = item.value_json
                row[key] = val
            yield row


def detect_metric_key(all_keys, must_contain, explicit=None):
    """Find a logged key matching all substrings in must_contain (case-insensitive),
    preferring ones that also look like validation metrics."""
    if explicit:
        return explicit

    candidates = [k for k in all_keys if all(s in k.lower() for s in must_contain)]
    val_candidates = [
        k
        for k in candidates
        if any(tag in k.lower() for tag in ("val", "eval", "test"))
    ]
    chosen = val_candidates if val_candidates else candidates
    return chosen[0] if chosen else None


def get_epoch(row, epoch_key=None):
    if epoch_key:
        return row.get(epoch_key)
    if "epoch" in row:
        return row["epoch"]
    return row.get("_step")


def process_file(path, macro_metric, micro_metric, epoch_key, max_epoch):
    rows = list(iter_history_rows(str(path)))
    if not rows:
        print(f"  [!] No history records found in {path}, skipping.")
        return None

    all_keys = set()
    for row in rows:
        all_keys.update(row.keys())

    macro_key = detect_metric_key(all_keys, ("f1", "macro"), macro_metric)
    if macro_key is None:
        print(f"  [!] Could not find a macro-F1 metric in {path}. Available keys:")
        print("      " + ", ".join(sorted(all_keys)))
        return None

    micro_key = detect_metric_key(all_keys, ("f1", "micro"), micro_metric)

    best_row = None
    best_epoch = None
    best_val = None
    for row in rows:
        if macro_key not in row or row[macro_key] is None:
            continue
        epoch = get_epoch(row, epoch_key)
        if epoch is not None and max_epoch is not None and epoch > max_epoch:
            continue
        try:
            val = float(row[macro_key])
        except (TypeError, ValueError):
            continue
        if best_val is None or val > best_val:
            best_val = val
            best_row = row
            best_epoch = epoch

    if best_row is None:
        print(
            f"  [!] No numeric values for '{macro_key}' within epoch <= {max_epoch} in {path}."
        )
        return None

    return {
        "file": str(path),
        "macro_key": macro_key,
        "micro_key": micro_key,
        "epoch": best_epoch,
        "row": best_row,
    }


def print_result(result):
    row = result["row"]
    macro_key = result["macro_key"]
    micro_key = result["micro_key"]

    print(f"File: {result['file']}")
    print(f"  Best epoch/step: {result['epoch']}")
    print(f"  {macro_key}: {row[macro_key]:.4f}")

    if micro_key and micro_key in row and row[micro_key] is not None:
        print(f"  {micro_key} (corresponding): {row[micro_key]:.4f}")
    elif micro_key:
        print(f"  {micro_key} (corresponding): N/A")
    else:
        print("  (no micro-F1 metric found)")

    shown = {macro_key, micro_key, "epoch", "_step"}
    other_keys = sorted(
        k
        for k in row.keys()
        if k not in shown and not k.startswith(INTERNAL_KEY_PREFIXES)
    )
    if other_keys:
        print("  Other metrics at this step:")
        for k in other_keys:
            v = row[k]
            if isinstance(v, float):
                print(f"    {k}: {v:.4f}")
            else:
                print(f"    {k}: {v}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "wandb_files", nargs="+", help="Paths to one or more run-*.wandb files"
    )
    parser.add_argument(
        "--macro-metric", default=None, help="Exact key for macro-F1 (skip auto-detect)"
    )
    parser.add_argument(
        "--micro-metric", default=None, help="Exact key for micro-F1 (skip auto-detect)"
    )
    parser.add_argument(
        "--epoch-key",
        default=None,
        help="Key to use as the epoch/step number (default: auto epoch/_step)",
    )
    parser.add_argument(
        "--max-epoch",
        type=int,
        default=200,
        help="Only consider steps/epochs up to this value",
    )
    args = parser.parse_args()

    results = []
    for f in args.wandb_files:
        path = Path(f)
        if not path.exists():
            print(f"  [!] File not found: {path}, skipping.")
            continue
        print(f"Processing {path.name} ...")
        result = process_file(
            path, args.macro_metric, args.micro_metric, args.epoch_key, args.max_epoch
        )
        if result:
            print_result(result)
            results.append(result)

    if not results:
        sys.exit("No valid results across the given files.")

    if len(results) < len(args.wandb_files):
        print(
            f"({len(results)}/{len(args.wandb_files)} files produced a valid result)\n"
        )

    print("=" * 50)
    print(f"Average across {len(results)} file(s):")

    # Average the macro-F1 across files
    macro_vals = [r["row"][r["macro_key"]] for r in results]
    print(f"  Average best macro-F1: {mean(macro_vals):.4f}  (n={len(macro_vals)})")

    # Average the corresponding micro-F1 across files (where present)
    micro_vals = [
        r["row"][r["micro_key"]]
        for r in results
        if r["micro_key"]
        and r["micro_key"] in r["row"]
        and r["row"][r["micro_key"]] is not None
    ]
    if micro_vals:
        print(
            f"  Average corresponding micro-F1: {mean(micro_vals):.4f}  (n={len(micro_vals)}/{len(results)})"
        )

    # Average every other numeric key found in the best rows (union across files)
    shown = (
        {"epoch", "_step"}
        | {r["macro_key"] for r in results}
        | {r["micro_key"] for r in results if r["micro_key"]}
    )
    other_key_values = {}
    for r in results:
        for k, v in r["row"].items():
            if k in shown or k.startswith(INTERNAL_KEY_PREFIXES):
                continue
            if isinstance(v, (int, float)):
                other_key_values.setdefault(k, []).append(v)

    for k in sorted(other_key_values):
        vals = other_key_values[k]
        print(f"  Average {k}: {mean(vals):.4f}  (n={len(vals)}/{len(results)})")


if __name__ == "__main__":
    main()
