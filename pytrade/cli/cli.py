import logging

import click

from pytrade.cli.dataset import dataset
from pytrade.cli.portfolio import portfolio
from pytrade.cli.signal import signal
from pytrade.cli.net import net
from pytrade.cli.arctic import arctic
from pytrade.cli.facts import facts
from pytrade.cli.schedule import schedule

logging.basicConfig(
    format="[%(asctime)s] %(levelname)-4s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO)


@click.group()
def main():
    pass


main.add_command(signal)
main.add_command(portfolio)
main.add_command(dataset)
main.add_command(net)
main.add_command(arctic)
main.add_command(facts)
main.add_command(schedule)

if __name__ == "__main__":
    main()
