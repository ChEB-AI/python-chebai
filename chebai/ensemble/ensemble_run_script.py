import importlib
from typing import Any, Dict, Type

import yaml
from jsonargparse import ArgumentParser

from chebai.ensemble.base import EnsembleBase


def load_class(class_path: str) -> Type[EnsembleBase]:
    """
    Dynamically imports and returns a class from a full dotted path.

    Args:
        class_path (str): Full module path to the class (e.g., 'my_package.module.MyClass').

    Returns:
        Type[EnsembleBase]: The imported class object.

    Raises:
        ModuleNotFoundError, AttributeError: If module or class cannot be loaded.
    """
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def load_config_and_instantiate(config_path: str) -> EnsembleBase:
    """
    Loads a YAML config file, imports the specified class, and instantiates it with the provided arguments.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        EnsembleBase: An instantiated object of the specified class.

    Raises:
        TypeError: If the loaded class is not a subclass of EnsembleBase.
    """
    with open(config_path, "r") as f:
        config: Dict[str, Any] = yaml.safe_load(f)

    class_path: str = config["class_path"]
    init_args: Dict[str, Any] = config.get("init_args", {})

    cls = load_class(class_path)

    if not issubclass(cls, EnsembleBase):
        raise TypeError(f"{cls} must be subclass of EnsembleBase")

    return cls(**init_args)


if __name__ == "__main__":
    # Example usage:
    # python ensemble_run_script.py --config=configs/ensemble/fullEnsembleWithWMV.yaml

    # Set up argument parser to receive config file path from CLI
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to the YAML config file")

    # Parse arguments from the command line
    args = parser.parse_args()

    # Load and instantiate the ensemble object
    ensemble = load_config_and_instantiate(args.config)

    # Ensure the loaded object is a valid EnsembleBase instance
    if not isinstance(ensemble, EnsembleBase):
        raise TypeError("Object must be an instance of `EnsembleBase`")

    # Run the ensemble pipeline
    ensemble.run_ensemble()
