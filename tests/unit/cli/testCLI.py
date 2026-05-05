import unittest
from pathlib import Path

import torch
from lightning import LightningDataModule, LightningModule

from chebai.cli import ChebaiCLI
from chebai.result.prediction import Predictor


class TestChebaiCLI(unittest.TestCase):
    def setUp(self):
        self.cli_args = [
            "fit",
            "--trainer=configs/training/default_trainer.yml",
            "--model=configs/model/ffn.yml",
            "--model.init_args.hidden_layers=[1]",
            "--model.train_metrics=configs/metrics/micro-macro-f1.yml",
            "--model.val_metrics=configs/metrics/micro-macro-f1.yml",
            "--data=tests/unit/cli/mock_dm_config.yml",
            "--model.pass_loss_kwargs=false",
            "--trainer.min_epochs=1",
            "--trainer.max_epochs=1",
            "--model.criterion=tests/unit/cli/bce_loss.yml",
        ]

    def test_mlp_on_chebai_cli(self):
        # Instantiate ChebaiCLI and ensure no exceptions are raised
        cli = ChebaiCLI(
            args=self.cli_args,
            save_config_kwargs={"config_filename": "lightning_config.yaml"},
            parser_kwargs={"parser_mode": "omegaconf"},
        )
        assert cli.trainer.log_dir is not None
        checkpoint_path = next(
            Path(cli.trainer.log_dir, "checkpoints").glob("*.ckpt"), None
        )
        assert checkpoint_path is not None and checkpoint_path.is_file()
        loaded_checkpoint = torch.load(checkpoint_path, weights_only=True)
        model_hparams = loaded_checkpoint[LightningModule.CHECKPOINT_HYPER_PARAMS_KEY]
        dm_hparams = loaded_checkpoint[LightningDataModule.CHECKPOINT_HYPER_PARAMS_KEY]
        assert "classification_labels" in loaded_checkpoint, (
            "Checkpoint is missing 'classification_labels' key."
        )
        assert "_class_path" in model_hparams, (
            "Model hyperparameters missing '_class_path' key."
        )
        assert "_class_path" in dm_hparams, (
            "DataModule hyperparameters missing '_class_path' key."
        )
        assert "classes_txt_file_path" not in model_hparams, (
            "Model hyperparameters should not contain 'classes_txt_file_path' key."
        )

        Predictor(checkpoint_path=checkpoint_path)


if __name__ == "__main__":
    unittest.main()
