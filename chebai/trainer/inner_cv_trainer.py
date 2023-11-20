import logging
import os

from lightning import Trainer
from sklearn import model_selection

from chebai.preprocessing.datasets.base import XYBaseDataModule

log = logging.getLogger(__name__)


class InnerCVTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        self.init_args = args
        self.init_kwargs = kwargs
        super().__init__(*args, **kwargs)

    def cv_fit(self, datamodule: XYBaseDataModule, n_splits: int = -1, *args, **kwargs):
        if n_splits < 2:
            self.fit(datamodule=datamodule, *args, **kwargs)
        else:
            datamodule.prepare_data()
            datamodule.setup()

            kfold = model_selection.KFold(n_splits=n_splits)

            for fold, (train_ids, val_ids) in enumerate(kfold.split(datamodule.train_val_data)):
                train_dataloader = datamodule.train_dataloader(ids=train_ids)
                val_dataloader = datamodule.val_dataloader(ids=val_ids)
                init_kwargs = self.init_kwargs
                print(f'init_kwargs: {init_kwargs}')
                init_kwargs['default_root_dir'] = os.path.join(self.default_root_dir, f'fold_{fold}')
                new_trainer = Trainer(*self.init_args, **self.init_kwargs)
                print(f'new default_root_dir: {new_trainer.default_root_dir}')
                print(f'new logger.save_dir: {new_trainer.logger.save_dir}')
                new_trainer.fit(train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, *args, **kwargs)
