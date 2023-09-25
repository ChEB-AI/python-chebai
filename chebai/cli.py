from lightning.pytorch.cli import LightningCLI


class ChebaiCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        for kind in ("train", "val", "test"):
            for average in ("micro", "macro"):
                parser.link_arguments(
                    "model.init_args.out_dim",
                    f"model.init_args.{kind}_metrics.init_args.metrics.{average}-f1.init_args.num_labels",
                )


def cli():
    r = ChebaiCLI(save_config_callback=None, parser_kwargs={"parser_mode": "omegaconf"})
