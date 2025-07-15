import unittest

from chebai.cli import ChebaiCLI


class TestChebaiCLI(unittest.TestCase):
    def setUp(self):
        self.cli_args = [
            "fit",
            "--trainer=configs/training/default_trainer.yml",
            "--model=configs/model/ffn.yml",
            "--model.init_args.hidden_layers=[10]",
            "--model.train_metrics=configs/metrics/micro-macro-f1.yml",
            "--model.test_metrics=configs/metrics/micro-macro-f1.yml",
            "--model.val_metrics=configs/metrics/micro-macro-f1.yml",
            "--data=tests/unit/cli/mock_dm_config.yml",
            "--model.pass_loss_kwargs=false",
            "--trainer.min_epochs=1",
            "--trainer.max_epochs=1",
            "--model.criterion=configs/loss/bce.yml",
            "--model.criterion.init_args.beta=0.99",
        ]

    def test_mlp_on_chebai_cli(self):
        # Instantiate ChebaiCLI and ensure no exceptions are raised
        try:
            ChebaiCLI(
                args=self.cli_args,
                save_config_kwargs={"config_filename": "lightning_config.yaml"},
                parser_kwargs={"parser_mode": "omegaconf"},
            )
        except Exception as e:
            self.fail(f"ChebaiCLI raised an unexpected exception: {e}")


if __name__ == "__main__":
    unittest.main()
