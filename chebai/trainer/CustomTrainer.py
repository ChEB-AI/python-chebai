import logging
from typing import Any, Optional, Tuple

import torch
from lightning import Trainer
from lightning.fabric.utilities.data import _set_sampler_epoch
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.loops.fit_loop import _FitLoop
from lightning.pytorch.trainer import call

from chebai.loggers.custom import CustomLogger

log = logging.getLogger(__name__)


class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        """
        Initializes the CustomTrainer class, logging additional hyperparameters to the custom logger if specified.

        Args:
            *args: Positional arguments for the Trainer class.
            **kwargs: Keyword arguments for the Trainer class.
        """
        self.init_args = args
        self.init_kwargs = kwargs
        super().__init__(*args, **kwargs, deterministic=True)
        # instantiation custom logger connector
        self._logger_connector.on_trainer_init(self.logger, 1)
        # log additional hyperparameters to wandb
        if isinstance(self.logger, CustomLogger):
            custom_logger = self.logger
            assert isinstance(custom_logger, CustomLogger)
            if custom_logger.verbose_hyperparameters:
                log_kwargs = {}
                for key, value in self.init_kwargs.items():
                    log_key, log_value = self._resolve_logging_argument(key, value)
                    log_kwargs[log_key] = log_value
                self.logger.log_hyperparams(log_kwargs)

        # use custom fit loop (https://lightning.ai/docs/pytorch/LTS/extensions/loops.html#overriding-the-default-loops)
        self.fit_loop = LoadDataLaterFitLoop(self, self.min_epochs, self.max_epochs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _resolve_logging_argument(self, key: str, value: Any) -> Tuple[str, Any]:
        """
        Resolves logging arguments, handling nested structures such as lists and complex objects.

        Args:
            key: The key of the argument.
            value: The value of the argument.

        Returns:
            A tuple containing the resolved key and value.
        """
        if isinstance(value, list):
            key_value_pairs = [
                self._resolve_logging_argument(f"{key}_{i}", v)
                for i, v in enumerate(value)
            ]
            return key, {k: v for k, v in key_value_pairs}
        if not (
            isinstance(value, str)
            or isinstance(value, float)
            or isinstance(value, int)
            or value is None
        ):
            params = {"class": value.__class__}
            params.update(value.__dict__)
            return key, params
        else:
            return key, value

    def predict(
        self,
        model=None,
        dataloaders=None,
        datamodule=None,
        return_predictions=None,
        ckpt_path=None,
    ):
        raise NotImplementedError(
            "CustomTrainer.predict is not implemented."
            "Use `Prediction.predict_from_file` or `Prediction.predict_smiles` from `chebai/result/prediction.py` instead."
        )

    @property
    def log_dir(self) -> Optional[str]:
        """
        Returns the logging directory.

        Returns:
            The path to the logging directory if available, else the default root directory.
        """
        if len(self.loggers) > 0:
            logger = self.loggers[0]
            if isinstance(logger, WandbLogger):
                dirpath = logger.experiment.dir
            else:
                dirpath = self.loggers[0].log_dir
        else:
            dirpath = self.default_root_dir

        dirpath = self.strategy.broadcast(dirpath)
        return dirpath


class LoadDataLaterFitLoop(_FitLoop):
    def on_advance_start(self) -> None:
        """Calls the hook ``on_train_epoch_start`` **before** the dataloaders are setup. This is necessary
         so that the dataloaders can get information from the model. For example: The on_train_epoch_start
        hook sets the curr_epoch attribute of the PubChemBatched dataset. With the Lightning configuration,
        the dataloaders would always load batch 0 first, run an epoch, then get the epoch number (usually 0,
        unless resuming from a checkpoint), then load batch 0 again (or some other batch). With this
        implementation, the dataloaders are setup after the epoch number is set, so that the correct
        batch is loaded."""
        trainer = self.trainer

        # update the epoch value for all samplers
        assert self._combined_loader is not None
        for i, dl in enumerate(self._combined_loader.flattened):
            _set_sampler_epoch(dl, self.epoch_progress.current.processed)

        if not self.restarted_mid_epoch and not self.restarted_on_epoch_end:
            if not self.restarted_on_epoch_start:
                self.epoch_progress.increment_ready()

            call._call_callback_hooks(trainer, "on_train_epoch_start")
            call._call_lightning_module_hook(trainer, "on_train_epoch_start")

            self.epoch_progress.increment_started()

        # this is usually at the front of advance_start, but here we need it at the end
        # might need to setup data again depending on `trainer.reload_dataloaders_every_n_epochs`
        self.setup_data()
