from typing import Dict, Set

from lightning.pytorch.cli import LightningArgumentParser, LightningCLI

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

    def add_arguments_to_parser(self, parser: LightningArgumentParser):
        """
        Link input parameters that are used by different classes (e.g. number of labels)
        see https://lightning.ai/docs/pytorch/stable/cli/lightning_cli_expert.html#argument-linking

        Args:
            parser (LightningArgumentParser): Argument parser instance.
        """
        for kind in ("train", "val", "test"):
            for average in ("micro-f1", "macro-f1", "balanced-accuracy"):
                parser.link_arguments(
                    "model.init_args.out_dim",
                    f"model.init_args.{kind}_metrics.init_args.metrics.{average}.init_args.num_labels",
                )
        parser.link_arguments(
            "model.init_args.out_dim", "trainer.callbacks.init_args.num_labels"
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
    r = ChebaiCLI(
        save_config_kwargs={"config_filename": "lightning_config.yaml"},
        parser_kwargs={"parser_mode": "omegaconf"},
    )
