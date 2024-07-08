import os
from datetime import datetime
from typing import List, Literal, Optional, Union

import wandb
from lightning.fabric.utilities.types import _PATH
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger


class CustomLogger(WandbLogger):
    """
    A custom logger that extends WandbLogger to add support for custom naming of runs and cross-validation.

    Args:
        save_dir (_PATH): Directory where logs are saved.
        name (str): Name of the logging run.
        version (Optional[Union[int, str]]): Version of the logging run.
        prefix (str): Prefix for logging.
        fold (Optional[int]): Cross-validation fold number.
        project (Optional[str]): Wandb project name.
        entity (Optional[str]): Wandb entity name.
        offline (bool): Whether to log offline.
        log_model (Union[Literal["all"], bool]): Whether to log the model.
        verbose_hyperparameters (bool): Whether to log hyperparameters verbosely.
        tags (Optional[List[str]]): List of tags for the run.
        **kwargs: Additional keyword arguments for WandbLogger.
    """

    def __init__(
        self,
        save_dir: _PATH,
        name: str = "logs",
        version: Optional[Union[int, str]] = None,
        prefix: str = "",
        fold: Optional[int] = None,
        project: Optional[str] = None,
        entity: Optional[str] = None,
        offline: bool = False,
        log_model: Union[Literal["all"], bool] = False,
        verbose_hyperparameters: bool = False,
        tags: Optional[List[str]] = None,
        **kwargs,
    ):
        if version is None:
            version = f"{datetime.now():%y%m%d-%H%M}"
        self._version = version
        self._name = name
        self._fold = fold
        self.verbose_hyperparameters = verbose_hyperparameters
        super().__init__(
            name=self.name,
            save_dir=save_dir,
            version=None,
            prefix=prefix,
            log_model=log_model,
            entity=entity,
            project=project,
            offline=offline,
            **kwargs,
        )
        if tags:
            self.experiment.tags += tuple(tags)

    @property
    def name(self) -> Optional[str]:
        """
        Returns the name of the logging run, including the version and fold number if applicable.
        """
        name = f"{self._name}_{self.version}"
        if self._fold is not None:
            name += f"_fold{self._fold}"
        return name

    @property
    def version(self) -> Optional[str]:
        """
        Returns the version of the logging run.
        """
        return self._version

    @property
    def root_dir(self) -> Optional[str]:
        """
        Returns the root directory for saving logs.
        """
        return os.path.join(self.save_dir, self.name)

    @property
    def log_dir(self) -> str:
        """
        Returns the directory for saving logs, including the version and fold number if applicable.
        """
        version = (
            self.version if isinstance(self.version, str) else f"version_{self.version}"
        )
        if self._fold is None:
            return os.path.join(self.root_dir, version)
        return os.path.join(self.root_dir, version, f"fold_{self._fold}")

    def set_fold(self, fold: int) -> None:
        """
        Sets the fold number and restarts the Wandb experiment with the new fold number.

        Args:
            fold (int): Cross-validation fold number.
        """
        if fold != self._fold:
            self._fold = fold
            # Start new experiment
            wandb.finish()
            self._wandb_init["name"] = self.name
            self._experiment = None
            _ = self.experiment

    @property
    def fold(self) -> Optional[int]:
        """
        Returns the current fold number.
        """
        return self._fold

    def after_save_checkpoint(self, checkpoint_callback: ModelCheckpoint) -> None:
        """
        Override method to prevent saving checkpoints as Wandb artifacts.
        """
        pass
