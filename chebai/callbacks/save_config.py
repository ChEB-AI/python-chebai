import os
from typing import TYPE_CHECKING

from lightning import LightningModule, Trainer
from lightning.pytorch.cli import SaveConfigCallback
from lightning.pytorch.loggers import WandbLogger

if TYPE_CHECKING:
    pass


class CustomSaveConfigCallback(SaveConfigCallback):
    """
    Custom SaveConfigCallback that uploads the Lightning config file to W&B.

    This callback extends the default SaveConfigCallback to automatically upload
    the lightning_config.yaml file to Weights & Biases online run logs when using
    WandbLogger. This ensures better traceability and reproducibility of experiments.

    The config file is uploaded using wandb.save(), which makes it available in the
    W&B web interface under the "Files" tab of the run.
    """

    def save_config(
        self, trainer: Trainer, pl_module: LightningModule, stage: str
    ) -> None:
        """
        Save the config to W&B if a WandbLogger is being used.

        This method is called after the config file has been saved to the log directory.
        It checks if the trainer is using a WandbLogger and, if so, uploads the config
        file to W&B using wandb.save().

        Note:
            We don't call super().save_config() because the parent class implementation
            is empty. The actual config file saving to disk happens in the setup() method
            before this method is called.

        Args:
            trainer: The PyTorch Lightning Trainer instance.
            pl_module: The LightningModule being trained.
            stage: The current stage of training (e.g., 'fit', 'validate', 'test').
        """
        # Only proceed if we're saving to log_dir and have a valid trainer
        if not self.save_to_log_dir or trainer.log_dir is None:
            return

        # Check if we're using WandbLogger
        wandb_logger = None
        for logger in trainer.loggers if hasattr(trainer, "loggers") else []:
            if isinstance(logger, WandbLogger):
                wandb_logger = logger
                break

        # If WandbLogger is found, upload the config file
        if wandb_logger is not None:
            config_path = os.path.join(trainer.log_dir, self.config_filename)
            if os.path.exists(config_path):
                try:
                    import wandb

                    # Upload the config file to W&B
                    # This will make it available in the W&B web interface
                    wandb.save(config_path, base_path=trainer.log_dir, policy="now")
                except ImportError:
                    # wandb is not installed, skip uploading
                    pass
                except Exception as e:
                    # Log the error but don't fail the training run
                    import logging

                    logging.warning(
                        f"Failed to upload {self.config_filename} to W&B: {e}"
                    )
