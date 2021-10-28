import click

from chebai import experiments


@click.group()
def cli():
    pass


@click.command()
@click.argument("experiment")
@click.argument("batch_size", type=click.INT)
@click.argument("args", nargs=-1)
def train(experiment, batch_size, args):
    """Run experiment identified by EXPERIMENT in batches of size BATCH_SIZE."""
    try:
        ex = experiments.EXPERIMENTS[experiment](*args)
    except KeyError:
        raise Exception(
            "Experiment ID not found. The following are available:"
            + ", ".join(experiments.EXPERIMENTS.keys())
        )
    ex.train(batch_size)


@click.command()
@click.argument("experiment")
@click.argument("ckpt_path")
@click.argument("data_path")
def predict(experiment, ckpt_path, data_path):
    """Run experiment identified by EXPERIMENT in batches of size BATCH_SIZE."""
    try:
        ex = experiments.EXPERIMENTS[experiment]()
    except KeyError:
        raise Exception(
            "Experiment ID not found. The following are available:"
            + ", ".join(experiments.EXPERIMENTS.keys())
        )
    ex.predict(ckpt_path, data_path)


cli.add_command(train)
cli.add_command(predict)
