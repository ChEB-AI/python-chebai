from lightning.pytorch.cli import LightningCLI

def cli():
    r = LightningCLI(save_config_callback=None)