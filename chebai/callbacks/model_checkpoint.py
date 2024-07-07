import os

from lightning.fabric.utilities.cloud_io import _is_dir
from lightning.fabric.utilities.types import _PATH
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities.rank_zero import rank_zero_info
from lightning_utilities.core.rank_zero import rank_zero_warn


class CustomModelCheckpoint(ModelCheckpoint):
    """
    Custom checkpoint class that resolves checkpoint paths to ensure checkpoints are saved in the same directory
    as other logs when using CustomLogger.
    Inherits from PyTorch Lightning's ModelCheckpoint class.
    """

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        """
        Setup the directory path for saving checkpoints. If the directory path is not set, it resolves the checkpoint
        directory using the custom logger's directory.

        Note:
            Same as in parent class, duplicated to be able to call self.__resolve_ckpt_dir

        Args:
            trainer (Trainer): The Trainer instance.
            pl_module (LightningModule): The LightningModule instance.
            stage (str): The stage of training (e.g., 'fit').
        """
        if self.dirpath is not None:
            self.dirpath = None
        dirpath = self.__resolve_ckpt_dir(trainer)
        dirpath = trainer.strategy.broadcast(dirpath)
        self.dirpath = dirpath
        if trainer.is_global_zero and stage == "fit":
            self.__warn_if_dir_not_empty(self.dirpath)

    def __warn_if_dir_not_empty(self, dirpath: _PATH) -> None:
        """
        Warn if the checkpoint directory is not empty.

        Note:
            Same as in parent class, duplicated because method in parent class is not accessible

        Args:
            dirpath (_PATH): The path to the checkpoint directory.
        """
        if (
            self.save_top_k != 0
            and _is_dir(self._fs, dirpath, strict=True)
            and len(self._fs.ls(dirpath)) > 0
        ):
            rank_zero_warn(f"Checkpoint directory {dirpath} exists and is not empty.")

    def __resolve_ckpt_dir(self, trainer: Trainer) -> _PATH:
        """
        Resolve the checkpoint directory path, ensuring compatibility with WandbLogger by saving checkpoints
        in the same directory as Wandb logs.

        Note:
            Overwritten for compatibility with wandb -> saves checkpoints in same dir as wandb logs

        Args:
            trainer (Trainer): The Trainer instance.

        Returns:
            _PATH: The resolved checkpoint directory path.
        """
        rank_zero_info(f"Resolving checkpoint dir (custom)")
        if self.dirpath is not None:
            # short circuit if dirpath was passed to ModelCheckpoint
            return self.dirpath
        if len(trainer.loggers) > 0:
            if trainer.loggers[0].save_dir is not None:
                save_dir = trainer.loggers[0].save_dir
            else:
                save_dir = trainer.default_root_dir
            name = trainer.loggers[0].name
            version = trainer.loggers[0].version
            version = version if isinstance(version, str) else f"version_{version}"
            logger = trainer.loggers[0]
            if isinstance(logger, WandbLogger) and isinstance(
                logger.experiment.dir, str
            ):
                ckpt_path = os.path.join(logger.experiment.dir, "checkpoints")
            else:
                ckpt_path = os.path.join(save_dir, str(name), version, "checkpoints")
        else:
            # if no loggers, use default_root_dir
            ckpt_path = os.path.join(trainer.default_root_dir, "checkpoints")

        rank_zero_info(f"Now using checkpoint path {ckpt_path}")
        return ckpt_path
