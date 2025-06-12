import importlib
from pathlib import Path

from chebai.models.base import ChebaiBaseNet
from chebai.preprocessing.datasets.base import XYBaseDataModule


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

    Args:
        input_dim (int): Name of the model to load.

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
        model = lightning_cls.load_from_checkpoint(
            model_ckpt_path, input_dim=5, **model_load_kwargs
        )
    except Exception as e:
        raise RuntimeError(f"Error loading model {model_name} \n Error: {e}") from e

    assert isinstance(
        model, ChebaiBaseNet
    ), f"Model: {model}(Model Name: {model_name}) is not a ChebaiBaseNet instance."
    model.eval()
    model.freeze()
    return model
