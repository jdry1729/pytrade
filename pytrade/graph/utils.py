import click

METADATA_REGEX = "(?P<key>[a-zA-Z0-9_.]*)=(?P<value>[a-zA-Z0-9_.]*)"


# TODO: remove
def graph_options(fn):
    fn = click.option("--executor-type", "-e", help="Executor to use.",
                      type=click.Choice(["SYNC", "RAY", "PROC"]), default="SYNC")(fn)
    fn = click.option("--node", "-n", multiple=True, help="Node to run.")(fn)
    fn = click.option("--resume", "-r", is_flag=True, default=False,
                      help="Whether to use stored data.")(fn)
    fn = click.option("--zap", "-z", multiple=True, help="Node to zap.")(fn)
    fn = click.option("--max-workers", "-w", help="Max workers.", type=int)(fn)
    return fn
