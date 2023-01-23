import click

from chebai import experiments
from chebai.result.base import PROCESSORS, ResultFactory
from chebai.result.molplot import AttentionOnMoleculesProcessor
from chebai.result.prediction_json import JSONResultProcessor


@click.group()
def cli():
    pass


@click.command()
@click.argument("experiment")
@click.argument("batch_size", type=click.INT)
@click.option("-g", "--group", default="default")
@click.option("--version", default=None)
@click.option("--load-prefix", default=None)
@click.option("--epochs", default=100)
@click.argument("args", nargs=-1)
def train(experiment, batch_size, group, version, load_prefix, epochs, args):
    """Run experiment identified by EXPERIMENT in batches of size BATCH_SIZE."""
    try:
        ex = experiments.EXPERIMENTS[experiment](batch_size, group, version=version)
    except KeyError:
        raise Exception(
            "Experiment ID not found. The following are available:"
            + ", ".join(experiments.EXPERIMENTS.keys())
        )
    ex.train(batch_size, epochs, *args, load_prefix=load_prefix)


@click.command()
@click.argument("experiment")
@click.argument("batch_size", type=click.INT)
@click.argument("ckpt_path")
@click.argument("args", nargs=-1)
def test(experiment, batch_size, ckpt_path, args):
    """Run experiment identified by EXPERIMENT in batches of size BATCH_SIZE."""
    try:
        ex = experiments.EXPERIMENTS[experiment](
            batch_size,
        )
    except KeyError:
        raise Exception(
            "Experiment ID not found. The following are available:"
            + ", ".join(experiments.EXPERIMENTS.keys())
        )
    ex.test(ckpt_path, *args)


@click.command()
@click.argument("experiment")
@click.argument("ckpt_path")
@click.argument("data_path")
@click.option("--processors", "-p", default=["json"], multiple=True)
@click.option("--compiled-data", "-c", is_flag=True, default=False)
@click.option("--headers", default=None)
def predict(experiment, ckpt_path, data_path, processors, compiled_data, headers):
    """Run experiment identified by EXPERIMENT in batches of size BATCH_SIZE."""
    try:
        ex = experiments.EXPERIMENTS[experiment](1)
    except KeyError:
        raise Exception(
            "Experiment ID not found. The following are available:"
            + ", ".join(experiments.EXPERIMENTS.keys())
        )
    if headers is not None:
        headers = headers.split(",")
    processor_list = []
    for p in processors:
        try:
            processor_list.append(PROCESSORS[p]())
        except KeyError:
            raise Exception(
                f"Processor {p} not found. Available processors are {', '.join(PROCESSORS.keys())}"
            )
    ex.predict(
        data_path, ckpt_path, processor_list, raw=not compiled_data, headers=headers
    )


cli.add_command(train)
cli.add_command(predict)
cli.add_command(test)
