import logging

import click

logger = logging.getLogger(__name__)


@click.group()
def signal():
    pass


@signal.command()
def ls():
    """
    List signals.
    """
    logging.disable()

    from pytrade.model.signal import read_signals
    from pytrade.data.postgres import sqlalchemy_engine
    from pytrade.utils.pandas import tabulate
    import pandas as pd

    with sqlalchemy_engine().begin() as conn:
        signals = read_signals(conn)

    signals = pd.DataFrame(signals, columns=["id", "name", "universe", "freq",
                                             "start_time", "end_time"])
    signals = signals[
        ["id", "name", "universe", "freq", "start_time", "end_time"]]
    signals["id"] = signals["id"].str.slice(0, 12)
    print(tabulate(pd.DataFrame(signals)))


@signal.command()
@click.argument("signal", nargs=-1, type=str)
def analyse(signal: str):
    """
    Analyse signals.
    """
    from pytrade.data.arctic import read_data, write_data
    from pytrade.data.postgres import sqlalchemy_engine
    from pytrade.model.signal import read_signal
    from pytrade.signal.analysis import analyse_signal
    from pytrade.utils.profile import load_profile

    profile = load_profile()
    signal_ids = signal

    for signal_id in signal_ids:
        with sqlalchemy_engine().begin() as conn:
            signal = read_signal(conn, signal_id)

        logger.info(f"Computing analytics: {signal.id}")

        universe_lib = f"graph/{signal.universe}"
        logger.info("Reading returns")
        returns = read_data(universe_lib, f"returns_{signal.freq}",
                            start_time=signal.start_time,
                            end_time=signal.end_time)

        logger.info("Reading signal")
        signal_values = read_data(profile.signal_values_lib,
                                  f"{signal.id}/values")

        analytics = analyse_signal(signal_values, returns, q=5)
        write_data(profile.signal_analytics_lib, signal.id, analytics,
                   unpack=True, create_library=True)
