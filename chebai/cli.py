from typing import Dict, Set, Type

from lightning.pytorch.cli import LightningArgumentParser, LightningCLI

from chebai.preprocessing.datasets.base import XYBaseDataModule
from chebai.trainer.CustomTrainer import CustomTrainer


class ChebaiCLI(LightningCLI):
    """
    Custom CLI subclass for Chebai project based on PyTorch Lightning's LightningCLI.

    Args:
        save_config_kwargs (dict): Keyword arguments for saving configuration.
        parser_kwargs (dict): Keyword arguments for parser configuration.

    Attributes:
        save_config_kwargs (dict): Configuration options for saving.
        parser_kwargs (dict): Configuration options for the argument parser.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize ChebaiCLI with custom trainer and configure parser settings.

        Args:
            args (list): List of arguments for LightningCLI.
            kwargs (dict): Keyword arguments for LightningCLI.
                save_config_kwargs (dict): Keyword arguments for saving configuration.
                parser_kwargs (dict): Keyword arguments for parser configuration.
        """
        super().__init__(trainer_class=CustomTrainer, *args, **kwargs)

    def before_instantiate_classes(self) -> None:
        """
        Hook called before instantiating classes (Lightning 2.6+ compatible).
        Instantiate the datamodule early to compute num_labels and feature_vector_size.
        """
        # Get the current subcommand config (fit, test, validate, predict, etc.)
        subcommand = self.config.get(self.config["subcommand"])
        
        if subcommand and "data" in subcommand:
            # Instantiate datamodule to get num_labels and feature_vector_size
            data_config = subcommand["data"]
            
            if "class_path" in data_config:
                # Import and instantiate the datamodule class
                module_path, class_name = data_config["class_path"].rsplit(".", 1)
                import importlib
                module = importlib.import_module(module_path)
                data_class = getattr(module, class_name)
                
                # Instantiate with init_args
                init_args = data_config.get("init_args", {})
                data_instance = data_class(**init_args)
                
                # Call prepare_data and setup to initialize dynamic properties
                if hasattr(data_instance, "_num_of_labels") and data_instance._num_of_labels is None:
                    data_instance.prepare_data()
                    data_instance.setup()
                
                num_labels = data_instance.num_of_labels
                feature_vector_size = data_instance.feature_vector_size
                
                # Update config with the computed values if not already set
                if "model" in subcommand and "init_args" in subcommand["model"]:
                    model_init_args = subcommand["model"]["init_args"]
                    if model_init_args.get("out_dim") is None:
                        model_init_args["out_dim"] = num_labels
                    if model_init_args.get("input_dim") is None:
                        model_init_args["input_dim"] = feature_vector_size
                    
                    # Update metrics num_labels in all metrics configurations
                    for kind in ("train", "val", "test"):
                        metrics_key = f"{kind}_metrics"
                        if metrics_key in model_init_args and model_init_args[metrics_key]:
                            metrics_config = model_init_args[metrics_key]
                            if "init_args" in metrics_config and "metrics" in metrics_config["init_args"]:
                                for metric_name, metric_config in metrics_config["init_args"]["metrics"].items():
                                    if "init_args" in metric_config and "num_labels" in metric_config["init_args"]:
                                        if metric_config["init_args"]["num_labels"] is None:
                                            metric_config["init_args"]["num_labels"] = num_labels
                
                # Update trainer callbacks num_labels
                if "trainer" in subcommand and "callbacks" in subcommand["trainer"]:
                    callbacks = subcommand["trainer"]["callbacks"]
                    if isinstance(callbacks, list):
                        for callback in callbacks:
                            if "init_args" in callback and "num_labels" in callback["init_args"]:
                                if callback["init_args"]["num_labels"] is None:
                                    callback["init_args"]["num_labels"] = num_labels
                    elif "init_args" in callbacks and "num_labels" in callbacks["init_args"]:
                        if callbacks["init_args"]["num_labels"] is None:
                            callbacks["init_args"]["num_labels"] = num_labels

    def add_arguments_to_parser(self, parser: LightningArgumentParser):
        """
        Link input parameters that are used by different classes (e.g. number of labels)
        see https://lightning.ai/docs/pytorch/stable/cli/lightning_cli_expert.html#argument-linking

        Args:
            parser (LightningArgumentParser): Argument parser instance.
        """

        # Link num_labels to metrics configurations
        # These links use the values set in before_instantiate_classes()
        for kind in ("train", "val", "test"):
            for average in (
                "micro-f1",
                "macro-f1",
                "balanced-accuracy",
                "roc-auc",
                "f1",
                "mse",
                "rmse",
                "r2",
            ):
                parser.link_arguments(
                    "model.init_args.out_dim",
                    f"model.init_args.{kind}_metrics.init_args.metrics.{average}.init_args.num_labels",
                )

        parser.link_arguments(
            "model.init_args.out_dim", "trainer.callbacks.init_args.num_labels"
        )

        parser.link_arguments(
            "data", "model.init_args.criterion.init_args.data_extractor"
        )

    @staticmethod
    def subcommands() -> Dict[str, Set[str]]:
        """
        Defines the list of available subcommands and the arguments to skip.

        Returns:
            Dict[str, Set[str]]: Dictionary where keys are subcommands and values are sets of arguments to skip.
        """
        return {
            "fit": {"model", "train_dataloaders", "val_dataloaders", "datamodule"},
            "validate": {"model", "dataloaders", "datamodule"},
            "test": {"model", "dataloaders", "datamodule"},
            "predict": {"model", "dataloaders", "datamodule"},
            "predict_from_file": {"model"},
        }


def cli():
    """
    Main function to instantiate and run the ChebaiCLI.
    """
    ChebaiCLI(
        save_config_kwargs={"config_filename": "lightning_config.yaml"},
        parser_kwargs={"parser_mode": "omegaconf"},
    )
