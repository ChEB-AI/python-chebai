import importlib

import yaml
from jsonargparse import ArgumentParser

from chebai.ensemble.base import EnsembleBase


def load_class(class_path: str):
    """Dynamically import a class from a full dotted path."""
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def load_config_and_instantiate(config_path: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    class_path = config["class_path"]
    init_args = config.get("init_args", {})

    cls = load_class(class_path)
    if not issubclass(cls, EnsembleBase):
        raise TypeError(f"{cls} must be subclass of EnsembleBase")
    return cls(**init_args)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to the YAML config file")

    args = parser.parse_args()
    ensemble = load_config_and_instantiate(args.config)

    if not isinstance(ensemble, EnsembleBase):
        raise TypeError("Object must be an instance of `EnsembleBase`")

    ensemble.run_ensemble()
