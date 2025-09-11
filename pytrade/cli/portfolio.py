import logging

import click

logger = logging.getLogger(__name__)


@click.group()
def portfolio():
    pass


@portfolio.command()
def ls():
    """
    List portfolios.
    """
    logging.disable()

    from pytrade.model.portfolio import read_portfolios
    from pytrade.data.postgres import sqlalchemy_engine
    from pytrade.utils.pandas import tabulate
    import pandas as pd

    with sqlalchemy_engine().begin() as conn:
        portfolios = read_portfolios(conn)

    portfolios = pd.DataFrame(portfolios)
    portfolios = portfolios[
        ["id", "name", "universe", "freq", "start_time", "end_time"]]
    portfolios["id"] = portfolios["id"].str.slice(0, 12)
    print(tabulate(pd.DataFrame(portfolios)))


@portfolio.command()
@click.argument("ids", nargs=-1)
def rm(ids):
    logging.disable()

    from pytrade.data.arctic import get_arctic_lib
    from pytrade.utils.profile import load_profile
    from pytrade.model.portfolio import read_portfolio
    from pytrade.data.postgres import sqlalchemy_engine
    from pytrade.model.portfolio import delete_portfolio

    profile = load_profile()
    weights_lib = get_arctic_lib(profile.portfolio_weights_lib)
    analytics_lib = get_arctic_lib(profile.portfolio_weights_lib)

    for id_ in ids:

        with sqlalchemy_engine().begin() as conn:
            portfolio = read_portfolio(conn, id_)
            print(f"Deleting portfolio: {portfolio['id'][:12]}")
            delete_portfolio(conn, id_)

            weights_lib.delete(f"{portfolio.id}/weights")

            # TODO: avoid list_symbols
            analytics_symbols = analytics_lib.list_symbols()
            to_delete = [x for x in analytics_symbols if
                         x.startswith(portfolio.id)]
            for symbol in to_delete:
                analytics_lib.delete(symbol)


@portfolio.command()
@click.argument("portfolio", nargs=-1, type=str)
@click.option("--commission", type=float, default=None)
def analyse(portfolio: str, commission: float):
    """
    Analyse a portfolio.
    """
    # import locally to speed up cli
    from pytrade.data.arctic import read_data, write_data
    from pytrade.data.postgres import sqlalchemy_engine
    from pytrade.model.portfolio import read_portfolio
    from pytrade.portfolio.analysis import analyse_portfolio
    from pytrade.utils.profile import load_profile

    profile = load_profile()
    portfolio_ids = portfolio

    for portfolio_id in portfolio_ids:
        with sqlalchemy_engine().begin() as conn:
            portfolio = read_portfolio(conn, portfolio_id)

        logger.info(f"Computing analytics: {portfolio_id}")

        universe_lib = f"graph/{portfolio.universe}"
        ann_factor = read_data(universe_lib,
                               f"ann_factor_{portfolio.freq}")
        logger.info("Reading returns")
        returns = read_data(universe_lib, f"returns_{portfolio.freq}",
                            start_time=portfolio.start_time,
                            end_time=portfolio.end_time)

        logger.info("Reading weights")
        weights = read_data(profile.portfolio_weights_lib,
                            f"{portfolio.id}/weights")

        analytics = analyse_portfolio(
            weights, returns, ann_factor=ann_factor,
            commission=commission)
        write_data(profile.portfolio_analytics_lib, portfolio.id, analytics,
                   unpack=True, create_library=True)
