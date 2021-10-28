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
        ex = experiments.EXPERIMENTS[experiment](batch_size, *args)
    except KeyError:
        raise Exception(
            "Experiment ID not found. The following are available:"
            + ", ".join(experiments.EXPERIMENTS.keys())
        )
    ex.execute()


cli.add_command(train)
