import logging
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from enum import Enum
from functools import partial
from typing import Optional, List

import pandas as pd
import psycopg2
import psycopg2.extensions
import sqlalchemy as sa
from psycopg2 import sql
from psycopg2.extras import execute_values

from pytrade.utils.profile import load_profile

logger = logging.getLogger(__file__)


class WriteMode(Enum):
    REPLACE = 0
    UPDATE = 1
    APPEND = 2
    PARALLEL_WRITE = 3
    PARALLEL_APPEND = 4


def _get_table_identifier(table: str, schema: Optional[str]):
    objs = []
    if schema:
        objs.append(schema)
    objs.append(table)
    return sql.Identifier(*objs)


def _get_table_literal(table: str, schema: Optional[str]):
    objs = []
    if schema:
        objs.append(schema)
    objs.append(table)
    return sql.Literal(".".join(objs))


def psycopg2_conn() -> psycopg2.extensions.connection:
    # TODO: ok to always create new connection?
    profile = load_profile()
    return psycopg2.connect(user=profile.postgres_user,
                            password=profile.postgres_password,
                            host=profile.postgres_host,
                            port=profile.postgres_port,
                            database=profile.postgres_database)


def sqlalchemy_engine(echo=False):
    profile = load_profile()
    url = sa.engine.URL("postgresql+psycopg2", username=profile.postgres_user,
                        password=profile.postgres_password,
                        host=profile.postgres_host, port=profile.postgres_port,
                        database=profile.postgres_database)
    return sa.create_engine(url, echo=echo, future=True)


def read_table(cursor: psycopg2.extensions.cursor, table: str,
               columns: Optional[List[str]] = None,
               filters: Optional[List[sql.Composable]] = None,
               schema: Optional[str] = None,
               order_by: Optional[str] = None):
    # must use sql.SQL for *
    fields = sql.SQL("*") if columns is None else sql.SQL(
        ", ").join(map(sql.Identifier, columns))
    query = sql.SQL("SELECT {fields} FROM {table}").format(
        table=_get_table_identifier(table, schema), fields=fields)
    if filters:
        query = sql.SQL(" ").join(
            [query, sql.SQL("WHERE"), sql.SQL(" AND ").join(filters)])
    logger.debug(query.as_string(cursor))
    cursor.execute(query)
    columns = [x.name for x in cursor.description]
    records = cursor.fetchall()
    index_col = get_primary_key(cursor, table, schema)
    data = pd.DataFrame(records, columns=columns)
    return data.set_index(index_col)


def write_table(cursor: psycopg2.extensions.cursor, table: str,
                data: pd.DataFrame, schema: Optional[str] = None,
                page_size=10000, mode: WriteMode = WriteMode.APPEND):
    index_col = data.index.names
    columns = data.columns.tolist()
    all_columns = index_col + columns
    records = data.reset_index().values.tolist()
    insert_query = sql.SQL(
        "INSERT INTO {table} ({all_columns}) VALUES {values}").format(
        table=_get_table_identifier(table, schema),
        all_columns=sql.SQL(", ").join(
            map(sql.Identifier, all_columns)),
        values=sql.Placeholder()
    )
    if mode == WriteMode.UPDATE:
        conflict_clause = sql.SQL(
            "ON CONFLICT ({conflict_target}) DO UPDATE SET ({columns}) = ROW("
            "{excluded})").format(
            conflict_target=sql.SQL(", ").join(
                map(sql.Identifier, index_col)),
            columns=sql.SQL(", ").join(map(sql.Identifier, columns)),
            excluded=sql.SQL(", ").join(
                map(sql.Identifier, ["excluded"] * len(columns), columns))
        )
        insert_query = sql.SQL(" ").join([insert_query, conflict_clause])
    if mode == WriteMode.REPLACE:
        delete_query = sql.SQL("DELETE FROM {table}").format(
            table=sql.Identifier(schema, table))
        cursor.execute(delete_query)
    execute_values(cursor, insert_query, records, page_size=page_size)


def get_primary_key(cursor: psycopg2.extensions.cursor, table: str,
                    schema: Optional[str] = None):
    query = sql.SQL(
        "SELECT a.attname FROM pg_index i JOIN pg_attribute a ON "
        "a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey) WHERE "
        "i.indrelid = {table}::regclass AND i.indisprimary").format(
        table=_get_table_literal(table, schema))
    cursor.execute(query)
    records = cursor.fetchall()
    return [x[0] for x in records]


def _get_filter_connective(value):
    if isinstance(value, tuple):
        return "IN"
    elif "%" in value:
        return "LIKE"
    return "="


def read_data(table: str, columns: Optional[List[str]] = None,
              start_time: Optional[datetime] = None,
              end_time: Optional[datetime] = None,
              schema: Optional[str] = None, squeeze: bool = False,
              **kwargs):
    # TODO: parallel read?
    filters = []
    if start_time is not None:
        filters.append(
            sql.SQL("time >= {time}").format(time=sql.Literal(start_time)))
    if end_time is not None:
        filters.append(
            sql.SQL("time <= {time}").format(time=sql.Literal(end_time)))
    for k, v in kwargs.items():
        connective = _get_filter_connective(v)
        filters.append(sql.SQL("{column} {connective} {value}").format(
            column=sql.Identifier(k), connective=sql.SQL(connective),
            value=sql.Literal(v)))
    with psycopg2_conn() as conn, conn.cursor() as cursor:
        data = read_table(cursor, table, columns=columns, filters=filters,
                          schema=schema)
    # squeeze is a noop if dataframe can't be squeezed
    if squeeze:
        data = data.squeeze()
    return data


def _write_chunk(chunk: pd.DataFrame, table: str, mode: WriteMode,
                 schema: Optional[str] = None):
    # must be defined outside write_data_data
    with psycopg2_conn() as conn, conn.cursor() as cursor:
        try:
            write_table(cursor, table, chunk, schema=schema, mode=mode)
        except Exception:
            logger.exception(f"Error writing chunk to: {table}")


def write_data(table: str,
               data: pd.DataFrame,
               mode: WriteMode,
               num_workers: int = 1,
               batch_size: Optional[int] = None,
               schema: Optional[str] = None):
    if batch_size is None:
        batch_size = len(data)

    slices = []
    for i in range(0, len(data), batch_size):
        slices.append(data.iloc[i:i + batch_size])

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        executor.map(
            partial(_write_chunk, table=table, schema=schema, mode=mode),
            slices)
