from datetime import datetime
from typing import Literal, Optional, Union
import os

from lightning.fabric.utilities.types import _PATH
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
import wandb


class CustomLogger(WandbLogger):
    """Adds support for custom naming of runs and cross-validation"""

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
        **kwargs,
    ):
        if version is None:
            version = f"{datetime.now():%y%m%d-%H%M}"
        self._version = version
        self._name = name
        self._fold = fold
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

    @property
    def name(self) -> Optional[str]:
        name = f"{self._name}_{self.version}"
        if self._fold is not None:
            name += f"_fold{self._fold}"
        return name

    @property
    def version(self) -> Optional[str]:
        return self._version

    @property
    def root_dir(self) -> Optional[str]:
        return os.path.join(self.save_dir, self.name)

    @property
    def log_dir(self) -> str:
        version = (
            self.version if isinstance(self.version, str) else f"version_{self.version}"
        )
        if self._fold is None:
            return os.path.join(self.root_dir, version)
        return os.path.join(self.root_dir, version, f"fold_{self._fold}")

    def set_fold(self, fold: int):
        if fold != self._fold:
            self._fold = fold
            # start new experiment
            wandb.finish()
            self._wandb_init["name"] = self.name
            self._experiment = None
            _ = self.experiment

    @property
    def fold(self):
        return self._fold

    def after_save_checkpoint(self, checkpoint_callback: ModelCheckpoint) -> None:
        # don't save checkpoint as wandb artifact
        pass
