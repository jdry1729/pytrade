import logging
from datetime import datetime
from typing import Optional, Collection

import click

logger = logging.getLogger(__name__)


@click.group()
def arctic():
    pass


@arctic.command()
@click.argument("src", nargs=1, type=str)
@click.argument("dest", nargs=1, type=str)
@click.option("--symbol", "-s", type=str, multiple=True, default=())
@click.option("--batch-size", type=int, default=None)
def cp(src: str, dest: str, symbol: Optional[Collection[str]],
       batch_size: Optional[int]):
    from pytrade.data.arctic import copy_data

    symbols = symbol
    if not symbols:
        symbols = None

    copy_data(src, dest, symbols, batch_size=batch_size)


@arctic.command()
@click.option("--library", "-l", type=str)
@click.option("--symbol", "-s", type=str, multiple=True)
@click.option("--symbols-file", "-f", type=str)
@click.option("--symbol-regex", "-r", type=str, multiple=True, default=None)
@click.option("--log-suffix", type=str, default=None)
@click.option("--log-prefix", type=str, default="log/")
@click.option("--snap", type=str, default=None)
@click.option("--time", "-t", type=click.DateTime(), default=None)
@click.option("--start-index", type=int, default=None)
@click.option("--end-index", type=int, default=None)
@click.option("--stack", is_flag=True, default=False)
def log(library: str, symbol: Collection[str] = (), symbols_file: Optional[str] = None,
        symbol_regex: Collection[str] = (), log_suffix: Optional[str] = None,
        log_prefix: Optional[str] = "log/", snap: Optional[str] = None,
        time: Optional[datetime] = None,
        start_index: Optional[int] = None, end_index: Optional[int] = None,
        stack: bool = False):
    from pytrade.data.arctic import log, ls
    from pytrade.utils.files import read_file_to_list

    if log_prefix is None:
        log_prefix = ""
    if log_suffix is None:
        log_suffix = ""

    symbols = set(symbol)
    if symbols_file is not None:
        symbols.update(read_file_to_list(symbols_file))
    for symbol_regex_ in symbol_regex:
        symbols.update(ls(library, symbol_regex_, snap=snap))

    for symbol in symbols:
        try:
            log(symbol, library, f"{log_prefix}{symbol}{log_suffix}", library,
                as_of=snap, time=time, start_index=start_index,
                end_index=end_index, stack=stack)
            print(f"Logged data for: {symbol}")
        except Exception:
            print(f"Error logging data for: {symbol}")


@arctic.command()
@click.argument("library", nargs=1, type=str)
@click.option("--regex", "-r", type=str)
@click.option("--snap", "-s", type=str)
@click.option("--profile", "-p", type=str)
def ls(library: str, regex: Optional[str], snap: Optional[str],
       profile: Optional[str]):
    logging.disable()
    from pytrade.data.arctic import ls as ls_
    symbols = ls_(library, regex=regex, snap=snap, profile=profile)
    for symbol in symbols:
        print(symbol)


@arctic.group()
def rm():
    pass


@rm.command()
@click.argument("name", nargs=1, type=str)
@click.argument("library", nargs=1, type=str)
def snap(name: str, library: str):
    from pytrade.data.arctic import delete_snap

    delete_snap(name, library)
