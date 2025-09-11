from datetime import datetime

from sequel.tables import portfolios_table
from sqlalchemy import insert, select, delete


def write_portfolio(conn, portfolio_id: str, name: str, universe: str,
                    freq: str, start_time: datetime, end_time: datetime):
    return conn.execute(insert(portfolios_table).values(
        id=portfolio_id, name=name, universe=universe, freq=freq,
        start_time=start_time, end_time=end_time))


def read_portfolio(conn, portfolio_id: str):
    # TODO: throw error if multiple portfolios exist starting with portfolio_id
    return conn.execute(select(portfolios_table).where(
        portfolios_table.c.id.like(f"{portfolio_id}%"))).mappings().one()


def read_portfolios(conn):
    return conn.execute(select(portfolios_table)).mappings().all()


def delete_portfolio(conn, portfolio_id: str):
    return conn.execute(delete(portfolios_table).where(
        portfolios_table.c.id.like(f"{portfolio_id}%")))
