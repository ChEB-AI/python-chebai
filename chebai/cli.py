from typing import Dict, Set

from lightning.pytorch.cli import LightningArgumentParser, LightningCLI

from chebai.trainer.CustomTrainer import CustomTrainer


class ChebaiCLI(LightningCLI):
    def __init__(self, *args, **kwargs):
        super().__init__(trainer_class=CustomTrainer, *args, **kwargs)

    def add_arguments_to_parser(self, parser: LightningArgumentParser):
        for kind in ("train", "val", "test"):
            for average in ("micro-f1", "macro-f1", "balanced-accuracy"):
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
        """Defines the list of available subcommands and the arguments to skip."""
        return {
            "fit": {"model", "train_dataloaders", "val_dataloaders", "datamodule"},
            "validate": {"model", "dataloaders", "datamodule"},
            "test": {"model", "dataloaders", "datamodule"},
            "predict": {"model", "dataloaders", "datamodule"},
            "predict_from_file": {"model"},
        }


def cli():
    r = ChebaiCLI(
        save_config_kwargs={"config_filename": "lightning_config.yaml"},
        parser_kwargs={"parser_mode": "omegaconf"},
    )
