import wandb.util as wandb_util
from chebai.models.electra import Electra
from chebai.models.base import ChebaiBaseNet
from chebai.preprocessing.datasets.base import XYBaseDataModule
import os
import wandb
import tqdm
import torch


def get_checkpoint_from_wandb(
    epoch,
    run,
    root=os.path.join("logs", "downloaded_ckpts"),
    model_class=None,
    map_device_to=None,
):
    """Gets wandb checkpoint based on run and epoch, downloads it if necessary"""
    api = wandb.Api()
    if model_class is None:
        model_class = Electra

    files = run.files()
    for file in files:
        if file.name.startswith(
            f"checkpoints/per_epoch={epoch}"
        ) or file.name.startswith(f"checkpoints/best_epoch={epoch}"):
            dest_path = os.path.join(root, run.name, file.name.split("/")[-1])
            if not os.path.isfile(dest_path):
                print(f"Downloading checkpoint to {dest_path}")
                wandb_util.download_file_from_url(dest_path, file.url, api.api_key)
            return model_class.load_from_checkpoint(
                dest_path, strict=False, map_location=map_device_to
            )
    print(f"No model found for epoch {epoch}")
    return None


def evaluate_model(
    model: ChebaiBaseNet,
    data_module: XYBaseDataModule,
    filename=None,
    buffer_dir=None,
    batch_size: int = 32,
    skip_existing_preds=False,
):
    """Runs model on test set of data_module (or, if filename is not None, on data set found in that file).
    If buffer_dir is set, results will be saved in buffer_dir. Returns tensors with predictions and labels.
    """
    model.eval()
    collate = data_module.reader.COLLATER()

    data_list = data_module.load_processed_data("test", filename)
    data_list = data_list[: data_module.data_limit]
    preds_list = []
    labels_list = []
    if buffer_dir is not None:
        os.makedirs(buffer_dir, exist_ok=True)
    save_ind = 0
    save_batch_size = 4
    n_saved = 1

    print(f"")
    for i in tqdm.tqdm(range(0, len(data_list), batch_size)):
        if not (
            skip_existing_preds
            and os.path.isfile(os.path.join(buffer_dir, f"preds{save_ind:03d}.pt"))
        ):
            collated = collate(data_list[i : min(i + batch_size, len(data_list) - 1)])
            collated.x = collated.to_x(model.device)
            if collated.y is not None:
                collated.y = collated.to_y(model.device)
            processable_data = model._process_batch(collated, 0)
            del processable_data["loss_kwargs"]
            model_output = model(processable_data, **processable_data["model_kwargs"])
            preds, labels = model._get_prediction_and_labels(
                processable_data, processable_data["labels"], model_output
            )
            preds_list.append(preds)
            labels_list.append(labels)
            if buffer_dir is not None:
                if n_saved >= save_batch_size:
                    torch.save(
                        torch.cat(preds_list),
                        os.path.join(buffer_dir, f"preds{save_ind:03d}.pt"),
                    )
                    if collated.y is not None:
                        torch.save(
                            torch.cat(labels_list),
                            os.path.join(buffer_dir, f"labels{save_ind:03d}.pt"),
                        )
                    preds_list = []
                    labels_list = []
        if n_saved >= save_batch_size:
            save_ind += 1
            n_saved = 0
        n_saved += 1

    if buffer_dir is None:
        test_preds = torch.cat(preds_list)
        if labels_list is not None:
            test_labels = torch.cat(labels_list)

            return test_preds, test_labels
        return test_preds, None


def load_results_from_buffer(buffer_dir, device):
    """Load results stored in evaluate_model()"""
    preds_list = []
    labels_list = []

    i = 0
    filename = f"preds{i:03d}.pt"
    while os.path.isfile(os.path.join(buffer_dir, filename)):
        preds_list.append(
            torch.load(
                os.path.join(buffer_dir, filename),
                map_location=torch.device(device),
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
