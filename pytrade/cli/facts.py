import logging
from datetime import datetime
from typing import Optional

import click


# TODO: facts CLI shouldn't be part of pytrade!


@click.group()
def facts():
    pass


@facts.command()
@click.argument("id_", nargs=1, type=int)
def rm(id_: int):
    from pytrade.data.sec import delete_manual_fact
    from pytrade.data.postgres import sqlalchemy_engine

    print(f"This operation will delete the fact with ID: {id_}")
    while True:
        res = input("Do you want to continue? [Y/n]")
        if res in ["N", "n"]:
            print("Aborted.")
            return
        if res in ["", "Y", "y"]:
            with sqlalchemy_engine().begin() as conn:
                delete_manual_fact(conn, id_)
            return


@facts.command()
@click.option("--cik", type=str)
@click.option("--accession-number", "-a", type=str)
@click.option("--name", "-n", type=str)
@click.option("--value", "-v", type=str)
@click.option("--start-time", type=click.DateTime(formats=["%Y-%m-%d"]))
@click.option("--end-time", type=click.DateTime(formats=["%Y-%m-%d"]))
@click.option("--dimensions", "-d", type=str)
@click.option("--unit", "-u", type=str)
def create(cik: int, accession_number: str, name: str, value: str,
           start_time: Optional[datetime] = None, end_time: Optional[datetime] = None,
           dimensions: Optional[str] = None, unit: Optional[str] = None):
    from pytrade.data.sec import create_manual_fact
    from pytrade.data.postgres import sqlalchemy_engine
    import json

    if dimensions is not None:
        dimensions = json.loads(dimensions)

    with sqlalchemy_engine().begin() as conn:
        create_manual_fact(conn, cik, accession_number, name, value, start_time,
                           end_time, dimensions, unit)


@facts.command()
def ls():
    """
    List facts.
    """
    logging.disable()

    from pytrade.data.sec import get_manual_facts
    from pytrade.data.postgres import sqlalchemy_engine
    from pytrade.utils.pandas import tabulate
    import pandas as pd

    with sqlalchemy_engine().begin() as conn:
        facts = get_manual_facts(conn)

    facts = pd.DataFrame(
        facts, columns=["id", "cik", "accession_number", "name", "value",
                        "start_time", "end_time", "dimensions",
                        "unit"])
    print(tabulate(facts))


if __name__ == "__main__":
    facts()
