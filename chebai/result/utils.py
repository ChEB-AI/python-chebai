import os
import shutil
from typing import Optional, Tuple, Union

import torch
import tqdm
import wandb
import wandb.util as wandb_util

from chebai.models.base import ChebaiBaseNet
from chebai.models.electra import Electra
from chebai.preprocessing.datasets.base import XYBaseDataModule
from chebai.preprocessing.datasets.chebi import _ChEBIDataExtractor


def get_checkpoint_from_wandb(
    epoch: int,
    run: wandb.apis.public.Run,
    root: str = os.path.join("logs", "downloaded_ckpts"),
):
    """
    Gets a wandb checkpoint based on run and epoch, downloads it if necessary.

    Args:
        epoch: The epoch number of the checkpoint to retrieve.
        run: The wandb run object.
        root: The root directory to save the downloaded checkpoint.

    Returns:
        The location of the downloaded checkpoint.
    """
    api = wandb.Api()

    files = run.files()
    for file in files:
        if file.name.startswith(
            f"checkpoints/per_epoch={epoch}"
        ) or file.name.startswith(f"checkpoints/best_epoch={epoch}"):
            dest_path = os.path.join(
                root, run.id, file.name.split("/")[-1].split("_")[1] + ".ckpt"
            )
            # legacy: also look for ckpts in the old format
            old_dest_path = os.path.join(root, run.name, file.name.split("/")[-1])
            if not os.path.isfile(dest_path):
                if os.path.isfile(old_dest_path):
                    print(f"Copying checkpoint from {old_dest_path} to {dest_path}")
                    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                    shutil.copy2(old_dest_path, dest_path)
                else:
                    print(f"Downloading checkpoint to {dest_path}")
                    wandb_util.download_file_from_url(dest_path, file.url, api.api_key)
            return dest_path
    print(f"No model found for epoch {epoch}")
    return None


def _run_batch(batch, model, collate):
    collated = collate(batch)
    collated.x = collated.to_x(model.device)
    if collated.y is not None:
        collated.y = collated.to_y(model.device)
    processable_data = model._process_batch(collated, 0)
    del processable_data["loss_kwargs"]
    model_output = model(processable_data, **processable_data["model_kwargs"])
    preds, labels = model._get_prediction_and_labels(
        processable_data, processable_data["labels"], model_output
    )
    return preds, labels


def _concat_tuple(l):
    if isinstance(l[0], tuple):
        print(l[0])
        return tuple([torch.cat([t[i] for t in l]) for i in range(len(l[0]))])
    return torch.cat(l)


def evaluate_model(
    model: ChebaiBaseNet,
    data_module: XYBaseDataModule,
    filename: Optional[str] = None,
    buffer_dir: Optional[str] = None,
    batch_size: int = 32,
    skip_existing_preds: bool = False,
    kind: str = "test",
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Runs the model on the test set of the data module or on the dataset found in the specified file.
    If buffer_dir is set, results will be saved in buffer_dir.

    Note:
        No need to provide "filename" parameter for Chebi dataset, "kind" parameter should be provided.

    Args:
        model: The model to evaluate.
        data_module: The data module containing the dataset.
        filename: Optional file name for the dataset.
        buffer_dir: Optional directory to save the results.
        batch_size: The batch size for evaluation.
        skip_existing_preds: Whether to skip evaluation if predictions already exist.
        kind: Kind of split of the data to be used for testing the model. Default is `test`.

    Returns:
        Tensors with predictions and labels.
    """
    model.eval()
    collate = data_module.reader.COLLATOR()

    if isinstance(data_module, _ChEBIDataExtractor):
        # As the dynamic split change is implemented only for chebi-dataset as of now
        data_df = data_module.dynamic_split_dfs[kind]
        data_list = data_df.to_dict(orient="records")
    else:
        data_list = data_module.load_processed_data("test", filename)
    data_list = data_list[: data_module.data_limit]
    preds_list = []
    labels_list = []
    if buffer_dir is not None:
        os.makedirs(buffer_dir, exist_ok=True)
    save_ind = 0
    save_batch_size = 128
    n_saved = 1

    print(f"")
    for i in tqdm.tqdm(range(0, len(data_list), batch_size)):
        if not (
            skip_existing_preds
            and os.path.isfile(os.path.join(buffer_dir, f"preds{save_ind:03d}.pt"))
        ):
            preds, labels = _run_batch(data_list[i : i + batch_size], model, collate)
            preds_list.append(preds)
            labels_list.append(labels)

            if buffer_dir is not None:
                if n_saved * batch_size >= save_batch_size:
                    torch.save(
                        _concat_tuple(preds_list),
                        os.path.join(buffer_dir, f"preds{save_ind:03d}.pt"),
                    )
                    if labels_list[0] is not None:
                        torch.save(
                            _concat_tuple(labels_list),
                            os.path.join(buffer_dir, f"labels{save_ind:03d}.pt"),
                        )
                    preds_list = []
                    labels_list = []
        if n_saved * batch_size >= save_batch_size:
            save_ind += 1
            n_saved = 0
        n_saved += 1

    if buffer_dir is None:
        test_preds = _concat_tuple(preds_list)
        if labels_list is not None:
            test_labels = _concat_tuple(labels_list)
            return test_preds, test_labels
        return test_preds, None
    elif len(preds_list) < 0:
        if len(preds_list) > 0 and preds_list[0] is not None:
            torch.save(
                _concat_tuple(preds_list),
                os.path.join(buffer_dir, f"preds{save_ind:03d}.pt"),
            )
        if len(labels_list) > 0 and labels_list[0] is not None:
            torch.save(
                _concat_tuple(labels_list),
                os.path.join(buffer_dir, f"labels{save_ind:03d}.pt"),
            )


def load_results_from_buffer(
    buffer_dir: str, device: torch.device
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Load results stored in evaluate_model() from the buffer directory.

    Args:
        buffer_dir: The directory containing the buffered results.
        device: The device to load the results onto.

    Returns:
        Tensors with predictions and labels.
    """
    preds_list = []
    labels_list = []

    i = 0
    filename = f"preds{i:03d}.pt"
    while os.path.isfile(os.path.join(buffer_dir, filename)):
        preds_list.append(
            torch.load(
                os.path.join(buffer_dir, filename),
                map_location=torch.device(device),
                weights_only=False,
            )
        )
        i += 1
        filename = f"preds{i:03d}.pt"

    i = 0
    filename = f"labels{i:03d}.pt"
    while os.path.isfile(os.path.join(buffer_dir, filename)):
        labels_list.append(
            torch.load(
                os.path.join(buffer_dir, filename),
                map_location=torch.device(device),
                weights_only=False,
            )
        )
        i += 1
        filename = f"labels{i:03d}.pt"

    if len(preds_list) > 0:
        test_preds = torch.cat(preds_list)
    else:
        test_preds = None
    if len(labels_list) > 0:
        test_labels = torch.cat(labels_list)
    else:
        test_labels = None

    return test_preds, test_labels


if __name__ == "__main__":
    import sys

    buffer_dir = os.path.join("results_buffer", sys.argv[1], "ChEBIOver100_train")
    buffer_dir_concat = os.path.join(
        "results_buffer", "concatenated", sys.argv[1], "ChEBIOver100_train"
    )
    os.makedirs(buffer_dir_concat, exist_ok=True)
    preds, labels = load_results_from_buffer(buffer_dir, "cpu")
    torch.save(preds, os.path.join(buffer_dir_concat, f"preds000.pt"))
    torch.save(labels, os.path.join(buffer_dir_concat, f"labels000.pt"))
