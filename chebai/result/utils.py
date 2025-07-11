import importlib
import os
import shutil
from pathlib import Path
from typing import Optional, Tuple

import torch
import tqdm
import wandb
import wandb.util as wandb_util
import yaml

from chebai.models.base import ChebaiBaseNet
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

    print("")
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


def load_class(class_path: str) -> type:
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def load_data_instance(data_cls_path: str, data_cls_kwargs: dict):
    assert isinstance(data_cls_kwargs, dict), "data_cls_kwargs must be a dict"
    data_cls = load_class(data_cls_path)
    assert isinstance(data_cls, type), f"{data_cls} is not a class."
    assert issubclass(
        data_cls, XYBaseDataModule
    ), f"{data_cls} must inherit from XYBaseDataModule"
    return data_cls(**data_cls_kwargs)


def load_model_for_inference(
    model_ckpt_path: str, model_cls_path: str, model_load_kwargs: dict, **kwargs
) -> ChebaiBaseNet:
    """
    Loads a model checkpoint and its label-related properties.

    Returns:
        Tuple[LightningModule, Dict[str, torch.Tensor]]: The model and its label properties.
    """
    assert isinstance(model_load_kwargs, dict), "model_load_kwargs must be a dict"

    model_name = kwargs.get("model_name", model_ckpt_path)

    if not Path(model_ckpt_path).exists():
        raise FileNotFoundError(
            f"Model path '{model_ckpt_path}' for '{model_name}' does not exist."
        )

    lightning_cls = load_class(model_cls_path)

    assert isinstance(lightning_cls, type), f"{lightning_cls} is not a class."
    assert issubclass(
        lightning_cls, ChebaiBaseNet
    ), f"{lightning_cls} must inherit from ChebaiBaseNet"
    try:
        model = lightning_cls.load_from_checkpoint(model_ckpt_path, **model_load_kwargs)
    except Exception as e:
        raise RuntimeError(f"Error loading model {model_name} \n Error: {e}") from e

    assert isinstance(
        model, ChebaiBaseNet
    ), f"Model: {model}(Model Name: {model_name}) is not a ChebaiBaseNet instance."
    model.eval()
    model.freeze()
    return model


def parse_config_file(config_path: str) -> tuple[str, dict]:
    path = Path(config_path)

    # Check file existence
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Check file extension
    if path.suffix.lower() not in [".yml", ".yaml"]:
        raise ValueError(
            f"Unsupported config file type: {path.suffix}. Expected .yaml or .yml"
        )

    # Load YAML content
    with open(path, "r") as f:
        config: dict = yaml.safe_load(f)

    class_path: str = config["class_path"]
    init_args: dict = config.get("init_args", {})
    assert isinstance(init_args, dict), "init_args must be a dictionary"
    return class_path, init_args


if __name__ == "__main__":
    import sys

    buffer_dir = os.path.join("results_buffer", sys.argv[1], "ChEBIOver100_train")
    buffer_dir_concat = os.path.join(
        "results_buffer", "concatenated", sys.argv[1], "ChEBIOver100_train"
    )
    os.makedirs(buffer_dir_concat, exist_ok=True)
    preds, labels = load_results_from_buffer(buffer_dir, "cpu")
    torch.save(preds, os.path.join(buffer_dir_concat, "preds000.pt"))
    torch.save(labels, os.path.join(buffer_dir_concat, "labels000.pt"))
