from jsonargparse import ArgumentParser

from chebai.ensemble._base import EnsembleBase
from chebai.ensemble._utils import load_class, parse_config_file


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

    class_path, init_args = parse_config_file(config_path)

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
