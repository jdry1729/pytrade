import logging

import click

from pytrade.cli.arctic import arctic
from pytrade.cli.schedule import schedule

logging.basicConfig(
    format="[%(asctime)s] %(levelname)-4s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO)


@click.group()
def main():
    pass


main.add_command(arctic)
main.add_command(schedule)

if __name__ == "__main__":
    main()
