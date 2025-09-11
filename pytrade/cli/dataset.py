import logging
from datetime import datetime
from typing import Optional, Collection

import click


def _get_dataset_names(ctx, param, incomplete):
    import psycopg2
    from psycopg2 import sql

    # TODO: better to use credentials from load_profile, but this slows function
    #  down considerably
    with psycopg2.connect(user="postgres",
                          password="password",
                          host="gamma",
                          port=30007,
                          database="trading") as conn, conn.cursor() as cursor:
        query = sql.SQL("SELECT name FROM datasets WHERE name LIKE %s")
        cursor.execute(query, (f"%{incomplete}%",))
        return [x[0] for x in cursor.fetchall()]


@click.group()
def dataset():
    pass


@dataset.command()
@click.argument("name", nargs=1, type=str, shell_complete=_get_dataset_names)
@click.option("--symbol", "-s", type=str, multiple=True, default=())
@click.option("--force", "-f", is_flag=True, default=False)
def fix(name: str, symbol: Optional[Collection[str]] = None, force: bool = False):
    from pytrade.data.postgres import sqlalchemy_engine
    from pytrade.data.datasets import fix_dataset

    symbols = symbol
    if not symbols:
        symbols = None

    with sqlalchemy_engine().begin() as conn:
        fix_dataset(conn, name, symbols, force)


@dataset.command()
@click.argument("name", nargs=1, type=str, shell_complete=_get_dataset_names)
def rm(name: str):
    from pytrade.data.datasets import delete_dataset, get_dataset, get_library_from_path
    from pytrade.data.arctic import delete_lib
    from pytrade.data.postgres import sqlalchemy_engine

    print(f"This operation will completely delete the {name} dataset.")
    while True:
        res = input("Do you want to continue? [Y/n]")
        if res in ["N", "n"]:
            print("Aborted.")
            return
        if res in ["", "Y", "y"]:
            with sqlalchemy_engine().begin() as conn:
                dataset = get_dataset(conn, name)
                library = get_library_from_path(dataset["path"])
                delete_dataset(conn, name)
                delete_lib(library)
            return


@dataset.command()
@click.argument("name", nargs=1, type=str, shell_complete=_get_dataset_names)
@click.option("--symbol", "-s", type=str, multiple=True, default=())
@click.option("--keep-spans", is_flag=True, default=False)
def wipe(name: str, symbol: Optional[Collection[str]], keep_spans: bool = False):
    from pytrade.data.datasets import delete_dataset_spans, get_dataset, \
        get_library_from_path
    from pytrade.data.arctic import delete_symbol
    from pytrade.data.postgres import sqlalchemy_engine
    from pytrade.data.datasets import get_symbols
    import fnmatch

    with sqlalchemy_engine().begin() as conn:
        # TODO: add start and end time args
        patterns = symbol
        all_symbols = get_symbols(conn, name)
        if not patterns:
            symbols = all_symbols
            message = f"This operation will delete all symbols from the {name} dataset"
        else:
            symbols = []
            for p in patterns:
                symbols.extend(fnmatch.filter(all_symbols, p))
            message = (f"This operation will delete the following symbols from the"
                       f" {name} dataset: {', '.join(symbols)}.")

        print(message)
        while True:
            res = input("Do you want to continue? [Y/n]")
            if res in ["N", "n"]:
                print("Aborted.")
                return
            if res in ["", "Y", "y"]:
                dataset = get_dataset(conn, name)
                library = get_library_from_path(dataset["path"])
                for symbol in symbols:
                    delete_symbol(library, symbol)
                if symbols and not keep_spans:
                    delete_dataset_spans(conn, dataset["id"], symbols)
                print(f"Deleted {len(symbols)} symbols")
                return


@dataset.command()
@click.argument("name", nargs=1, type=str)
@click.option("--path", "-p", type=str)
@click.option("--type", "-t", type=click.Choice(["TIME_SERIES", "REFERENCE"]),
              default="TIME_SERIES")
def create(name: str, path: str, type: str):
    from sequel.tables import DatasetType
    from pytrade.data.datasets import create_dataset
    from pytrade.data.postgres import sqlalchemy_engine

    type_ = DatasetType[type]
    with sqlalchemy_engine().begin() as conn:
        create_dataset(conn, name, path, type_)


@dataset.command()
def ls():
    """
    List datasets.
    """
    logging.disable()

    from pytrade.data.datasets import get_datasets
    from pytrade.data.postgres import sqlalchemy_engine
    from pytrade.utils.pandas import tabulate
    import pandas as pd

    with sqlalchemy_engine().begin() as conn:
        datasets = get_datasets(conn)

    datasets = pd.DataFrame(
        datasets, columns=["id", "name", "creation_time", "type", "path"])
    datasets["type"] = datasets["type"].map(lambda x: x.name)
    print(tabulate(datasets))


@dataset.command()
@click.argument("name", nargs=1, type=str, shell_complete=_get_dataset_names)
@click.option("--symbol", "-s", type=str, multiple=True, default=())
@click.option("--coverage-start-time", type=click.DateTime(formats=["%Y-%m-%d"]),
              help="Must be in UTC.")
@click.option("--coverage-end-time", type=click.DateTime(formats=["%Y-%m-%d"]),
              help="Must be in UTC.", default=None)
def status(name: str, symbol: str, coverage_start_time: datetime,
           coverage_end_time: datetime):
    from pytrade.data.postgres import sqlalchemy_engine
    from pytrade.data.datasets import get_status
    from pytrade.utils.pandas import tabulate
    from pytrade.data.datasets import get_dataset_id

    symbols = symbol
    if not symbols:
        symbols = None

    with sqlalchemy_engine().begin() as conn:
        dataset_id = get_dataset_id(conn, name)
        status = get_status(conn, dataset_id, symbols,
                            coverage_start_time=coverage_start_time,
                            coverage_end_time=coverage_end_time)
        status["coverage"] = status["coverage"].round(2)

    print(tabulate(status.reset_index()))
