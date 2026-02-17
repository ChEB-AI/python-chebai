import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from lightning import LightningModule, Trainer
from lightning.pytorch.loggers import WandbLogger

from chebai.callbacks.save_config import CustomSaveConfigCallback


class DummyModule(LightningModule):
    """Dummy module for testing."""

    def __init__(self):
        super().__init__()
        self.layer = None

    def forward(self, x):
        return x


class TestCustomSaveConfigCallback(unittest.TestCase):
    """Test CustomSaveConfigCallback functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def test_callback_uploads_config_with_wandb_logger(self):
        """Test that the callback uploads config when WandbLogger is present."""
        # Create a mock parser and config
        mock_parser = MagicMock()
        mock_config = MagicMock()

        # Create the callback
        callback = CustomSaveConfigCallback(
            parser=mock_parser,
            config=mock_config,
            config_filename="lightning_config.yaml",
            overwrite=True,
        )

        # Create a config file in the temp directory
        config_path = os.path.join(self.temp_dir, "lightning_config.yaml")
        with open(config_path, "w") as f:
            f.write("test: config\n")

        # Create a mock WandbLogger
        mock_wandb_logger = MagicMock(spec=WandbLogger)

        # Create a mock trainer with the WandbLogger
        mock_trainer = MagicMock(spec=Trainer)
        mock_trainer.log_dir = self.temp_dir
        mock_trainer.loggers = [mock_wandb_logger]
        mock_trainer.is_global_zero = True

        # Create a dummy module
        pl_module = DummyModule()

        # Mock wandb module
        with patch("wandb.save") as mock_wandb_save:
            # Call save_config
            callback.save_config(mock_trainer, pl_module, "fit")

            # Verify wandb.save was called with the correct arguments
            mock_wandb_save.assert_called_once_with(
                config_path, base_path=self.temp_dir, policy="now"
            )

    def test_callback_skips_upload_without_wandb_logger(self):
        """Test that the callback skips upload when no WandbLogger is present."""
        # Create a mock parser and config
        mock_parser = MagicMock()
        mock_config = MagicMock()

        # Create the callback
        callback = CustomSaveConfigCallback(
            parser=mock_parser,
            config=mock_config,
            config_filename="lightning_config.yaml",
            overwrite=True,
        )

        # Create a config file in the temp directory
        config_path = os.path.join(self.temp_dir, "lightning_config.yaml")
        with open(config_path, "w") as f:
            f.write("test: config\n")

        # Create a mock trainer WITHOUT WandbLogger
        mock_trainer = MagicMock(spec=Trainer)
        mock_trainer.log_dir = self.temp_dir
        mock_trainer.loggers = []  # No loggers
        mock_trainer.is_global_zero = True

        # Create a dummy module
        pl_module = DummyModule()

        # Mock wandb module
        with patch("wandb.save") as mock_wandb_save:
            # Call save_config
            callback.save_config(mock_trainer, pl_module, "fit")

            # Verify wandb.save was NOT called
            mock_wandb_save.assert_not_called()

    def test_callback_handles_missing_config_file(self):
        """Test that the callback handles missing config file gracefully."""
        # Create a mock parser and config
        mock_parser = MagicMock()
        mock_config = MagicMock()

        # Create the callback
        callback = CustomSaveConfigCallback(
            parser=mock_parser,
            config=mock_config,
            config_filename="nonexistent_config.yaml",
            overwrite=True,
        )

        # Create a mock WandbLogger
        mock_wandb_logger = MagicMock(spec=WandbLogger)

        # Create a mock trainer with the WandbLogger
        mock_trainer = MagicMock(spec=Trainer)
        mock_trainer.log_dir = self.temp_dir
        mock_trainer.loggers = [mock_wandb_logger]
        mock_trainer.is_global_zero = True

        # Create a dummy module
        pl_module = DummyModule()

        # Mock wandb module
        with patch("wandb.save") as mock_wandb_save:
            # Call save_config - should not raise an error
            callback.save_config(mock_trainer, pl_module, "fit")

            # Verify wandb.save was NOT called (because file doesn't exist)
            mock_wandb_save.assert_not_called()

    def test_callback_handles_wandb_not_installed(self):
        """Test that the callback handles missing wandb package gracefully."""
        # Create a mock parser and config
        mock_parser = MagicMock()
        mock_config = MagicMock()

        # Create the callback
        callback = CustomSaveConfigCallback(
            parser=mock_parser,
            config=mock_config,
            config_filename="lightning_config.yaml",
            overwrite=True,
        )

        # Create a config file in the temp directory
        config_path = os.path.join(self.temp_dir, "lightning_config.yaml")
        with open(config_path, "w") as f:
            f.write("test: config\n")

        # Create a mock WandbLogger
        mock_wandb_logger = MagicMock(spec=WandbLogger)

        # Create a mock trainer with the WandbLogger
        mock_trainer = MagicMock(spec=Trainer)
        mock_trainer.log_dir = self.temp_dir
        mock_trainer.loggers = [mock_wandb_logger]
        mock_trainer.is_global_zero = True

        # Create a dummy module
        pl_module = DummyModule()

        # Mock wandb import to raise ImportError
        # This simulates wandb not being installed
        with patch("builtins.__import__") as mock_import:

            def import_side_effect(name, *args, **kwargs):
                if name == "wandb":
                    raise ImportError("No module named 'wandb'")
                return __import__(name, *args, **kwargs)

            mock_import.side_effect = import_side_effect

            # Call save_config - should not raise an error
            # The callback should catch the ImportError and continue gracefully
            callback.save_config(mock_trainer, pl_module, "fit")


if __name__ == "__main__":
    unittest.main()
