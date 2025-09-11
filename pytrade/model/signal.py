from datetime import datetime

from sequel.tables import signals_table
from sqlalchemy import insert, select, delete


def write_signal(conn, signal_id: str, name: str, universe: str,
                 freq: str, start_time: datetime, end_time: datetime):
    return conn.execute(insert(signals_table).values(
        id=signal_id, name=name, universe=universe, freq=freq,
        start_time=start_time, end_time=end_time))


def read_signal(conn, signal_id: str):
    # TODO: throw error if multiple signals exist starting with portfolio_id
    return conn.execute(select(signals_table).where(
        signals_table.c.id.like(f"{signal_id}%"))).mappings().one()


def read_signals(conn):
    return conn.execute(select(signals_table)).mappings().all()


def delete_signal(conn, signal_id: str):
    return conn.execute(delete(signals_table).where(
        signals_table.c.id.like(f"{signal_id}%")))
