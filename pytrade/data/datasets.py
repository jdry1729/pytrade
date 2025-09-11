import warnings

from tqdm import tqdm

from pytrade.data.arctic import list_symbols, get_description
from pytrade.utils.random import generate_uid
from pytrade.utils.typing import DatetimeOrFloat, TimedeltaOrFloat

warnings.filterwarnings("ignore")

import logging
import re
from datetime import timedelta, datetime
from typing import Collection, Optional, Callable, Any, Dict, List, Iterable

import pandas as pd
import sqlalchemy as sa
from pytrade.data import write_data, sqlalchemy_engine, read_data
from pytrade.data.postgres import WriteMode
from pytrade.data.processing import merge_overlapping_intervals, interval_difference
from pytrade.graph import ExecutorType, run_graph, Graph, set_active_graph, NodeRef, \
    add_node, set_ns
from pytrade.graph.active import add_edge
from pytrade.utils.functions import partial
from pytrade.utils.time import get_equally_spaced_times, ISO_8601_FORMAT
from sequel.tables import DatasetType
from sequel.tables.datasets import datasets_table, dataset_spans_table
from sqlalchemy.engine import Connection
from unidecode import unidecode
import numpy as np

logger = logging.getLogger(__name__)

ARCTIC_DATASET_PATH_REGEX = "arctic://(?P<library>[A-Za-z0-9\-\_\/]+)"


# The datasets API is designed to load data from various data sources.
#
# Time Series Datasets
#
# If loading time series data, the data_fn must return a dataframe/ series indexed
# by time. The times don't have to represent the time each event was published,
# although it's convenient to stick to this rule where possible.
#
# There may be a time delay between an event occurring and being published. In
# these situations it's useful to specify a settling period. The data in each span
# between the creation time of the span and the creation time minus the settling
# period is ignored when load_data figures out what time periods to download data
# for.
#
# The settling period option is also useful if the data is restated for a number
# of days after first being published. Specifying a settling period will ensure that
# the latest data is redownloaded.
#
# Dataset spans should be considered an append-only log, and should only be deleted
# if the dataset is deleted. If you have downloaded bad data, rather than meddling
# with the dataset_spans table, you should simply redownload it.


def get_dataset_id(conn, name: str) -> int:
    return conn.execute(
        sa.select(datasets_table.c.id).where(datasets_table.c.name == name)
    ).scalar_one()


def _get_dataset(conn: Connection, id: str) -> Dict:
    return conn.execute(
        sa.select(datasets_table).where(datasets_table.c.id == id)
    ).one()


def get_dataset(conn: Connection, name: str) -> Dict:
    return conn.execute(
        sa.select(datasets_table).where(datasets_table.c.name == name)
    ).one()


def get_dataset_names(conn: Connection, pattern: str):
    return conn.execute(
        sa.select(datasets_table.c.name).where(datasets_table.c.name.like(pattern))
    ).scalars().all()


def create_dataset(conn: Connection, name: str, path: str,
                   type_: DatasetType) -> int:
    res = conn.execute(
        sa.insert(datasets_table).values(
            name=name, creation_time=datetime.utcnow(), path=path, type=type_))
    logger.info(f"Created dataset: {name}")
    return res.inserted_primary_key[0]


def delete_dataset(conn, name: str) -> None:
    # below deletes spans as well due to cascade deletion
    conn.execute(sa.delete(datasets_table).where(
        datasets_table.c.name == name))
    logger.info(f"Deleted dataset: {name}")


def delete_dataset_spans(conn, dataset_id: int,
                         symbols: Optional[Iterable[str]] = None):
    message = "Deleted spans"
    conds = [dataset_spans_table.c.dataset_id == dataset_id]
    if symbols is not None:
        message += f" for: {', '.join(symbols)}"
        conds.append(dataset_spans_table.c.symbol.in_(symbols))
    conn.execute(sa.delete(dataset_spans_table).where(*conds))
    logger.info(message)


def get_datasets(conn: Connection) -> List[Dict]:
    return conn.execute(sa.select(datasets_table)).mappings().all()


def _insert_dataset_span(conn, dataset_id: int, symbol: str, creation_time: datetime,
                         start_time: datetime, end_time: datetime,
                         row_count: int, data_start_time: Optional[datetime] = None,
                         data_end_time: Optional[datetime] = None) -> None:
    conn.execute(sa.insert(dataset_spans_table).values(
        dataset_id=dataset_id, creation_time=creation_time,
        symbol=symbol, start_time=start_time, end_time=end_time,
        row_count=row_count, data_start_time=data_start_time,
        data_end_time=data_end_time,
    ))


def _get_dataset_spans(conn, dataset_id: int,
                       *,
                       symbols: Optional[Collection[str]] = None,
                       start_time: Optional[datetime] = None,
                       end_time: Optional[datetime] = None) -> pd.DataFrame:
    """
    Gets all spans for a dataset for an optional set of symbols. If start/ end times
    specified, spans will only be returned which overlap [start_time, end_time).
    """
    # start time of spans is inclusive, end time is exclusive
    params = {"dataset_id": dataset_id}
    stmt = ("SELECT id, creation_time, symbol, start_time, end_time"
            " FROM dataset_spans WHERE dataset_id = %(dataset_id)s")
    if symbols is not None:
        stmt += " AND symbol in %(symbols)s"
        params["symbols"] = tuple(symbols)
    if start_time is not None:
        stmt += " AND end_time > %(interval_start_time)s"
        params["interval_start_time"] = start_time
    if end_time is not None:
        stmt += " AND start_time < %(interval_end_time)s"
        params["interval_end_time"] = end_time
    return pd.read_sql(stmt, con=conn.connection, params=params,
                       index_col="id")


def get_dataset_spans(conn, dataset_name: str,
                      symbols: Optional[Collection[str]] = None,
                      start_time: Optional[datetime] = None,
                      end_time: Optional[datetime] = None) -> pd.DataFrame:
    dataset_id = get_dataset_id(conn, dataset_name)
    return _get_dataset_spans(conn, dataset_id, symbols=symbols,
                              start_time=start_time, end_time=end_time)


def get_symbols(conn, dataset_name: str) -> List[str]:
    dataset_id = get_dataset_id(conn, dataset_name)
    with sqlalchemy_engine().connect() as conn:
        return conn.execute(
            sa.select(sa.distinct(dataset_spans_table.c.symbol)).where(
                dataset_spans_table.c.dataset_id == dataset_id)).scalars().all()


def get_library_from_path(path: str) -> str:
    if match := re.search(ARCTIC_DATASET_PATH_REGEX, path):
        return match.group("library")
    raise ValueError("Error getting library; invalid path")


def _coverage_fn(spans, start_time, end_time):
    total_period = (end_time - start_time).total_seconds()

    coverage = 0
    spans = merge_overlapping_intervals(spans)
    spans = spans[(spans["start"] <= end_time) & (spans["end"] >= start_time)]
    spans = spans.clip(lower=start_time, upper=end_time)
    for i in range(len(spans)):
        span = spans.iloc[i]
        coverage += (span["end"] - span["start"]).total_seconds()

    return coverage / total_period


def _get_coverage_from_spans(spans: pd.DataFrame, start_time: datetime,
                             end_time: datetime,
                             symbols: Optional[Iterable[str]] = None):
    if spans.empty:
        return pd.Series([], index=[])

    spans = spans.drop(columns=["creation_time"])
    spans = spans.rename(columns={"start_time": "start", "end_time": "end"})
    coverage = spans.groupby("symbol").apply(_coverage_fn, start_time=start_time,
                                             end_time=end_time)
    if symbols is not None:
        coverage = coverage.reindex(index=symbols).fillna(0)
    return coverage


def _get_coverage(conn, dataset_id: int, start_time: datetime, end_time: datetime,
                  symbols: Optional[Iterable[str]] = None) -> pd.Series:
    spans = _get_dataset_spans(conn, dataset_id, symbols=symbols)
    return _get_coverage_from_spans(spans, start_time, end_time, symbols)


def get_status(conn, dataset_id: int,
               symbols: Optional[Iterable[str]] = None,
               coverage_start_time: datetime = datetime(2010, 1, 1),
               coverage_end_time: Optional[datetime] = None) -> pd.DataFrame:
    if coverage_start_time is None:
        coverage_start_time = datetime(2010, 1, 1)

    if coverage_end_time is None:
        coverage_end_time = datetime.utcnow()

    spans = _get_dataset_spans(conn, dataset_id, symbols=symbols)
    updated_at = spans.groupby("symbol")["creation_time"].max().reindex(
        symbols).fillna(pd.NaT)
    start_time = spans.groupby("symbol")["start_time"].min().reindex(
        symbols).fillna(pd.NaT)
    end_time = spans.groupby("symbol")["end_time"].max().reindex(
        symbols).fillna(pd.NaT)
    num_spans = spans.groupby("symbol")["creation_time"].count().reindex(
        symbols).fillna(0)
    coverage = _get_coverage_from_spans(spans, start_time=coverage_start_time,
                                        end_time=coverage_end_time, symbols=symbols)

    status = pd.concat([num_spans.rename("spans"),
                        start_time.rename("start_time"),
                        end_time.rename("end_time"),
                        coverage.rename("coverage"),
                        updated_at.rename("updated_at")], axis=1)
    return status.sort_values(["updated_at", "coverage"], ascending=[True, True])


def fix_dataset(conn, dataset_name: str, symbols: Optional[Collection[str]] = None,
                force: bool = False):
    # TODO: insert multiple spans if large gaps in data
    logger.info(f"Fixing dataset: {dataset_name}; {force=}")
    dataset = get_dataset(conn, dataset_name)
    library = get_library_from_path(dataset["path"])

    if symbols is None:
        symbols = list_symbols(library)

    if force:
        delete_dataset_spans(conn, dataset["id"], symbols)

    spans = _get_dataset_spans(conn, dataset["id"], symbols=symbols)
    num_spans = spans.groupby("symbol")["creation_time"].count().reindex(
        symbols).fillna(0)

    symbols_ = []
    for symbol in symbols:
        if num_spans[symbol] == 0:
            symbols_.append(symbol)

    if symbols_:
        for symbol in tqdm(symbols_, ncols=50):
            # just read description, head and tail for speed
            row_count = get_description(library, symbol).row_count
            head = read_data(library, symbol, end_index=5)
            tail = read_data(library, symbol, start_index=-5)
            data = pd.concat([head, tail])
            if not data.empty:
                times = data.index.get_level_values(0)
                start_time = times[0]
                end_time = times[-1]
                _insert_dataset_span(conn, dataset["id"], symbol,
                                     creation_time=datetime.utcnow(),
                                     start_time=start_time, end_time=end_time,
                                     row_count=row_count,
                                     data_start_time=start_time,
                                     data_end_time=end_time)


# TODO: better name?
def _ts_data_fn(symbol: str, start_time: datetime, end_time: datetime,
                data_fn: Callable, allow_data_outside_span_period: bool = False):
    logger.info(f"Getting data for {symbol} from"
                f" {start_time.strftime(ISO_8601_FORMAT)} to"
                f" {end_time.strftime(ISO_8601_FORMAT)}")
    data = data_fn(symbol, start_time, end_time)
    if not allow_data_outside_span_period:
        # ensure data_fn returns data from start time inclusive to end time exclsuive
        return data.loc[start_time:end_time - timedelta(milliseconds=1)]
    return data


def _ts_write_fn(node: NodeRef, data: Any, dataset_id: str, library: str,
                 metadata: Dict) -> None:
    if node.name == "clear":
        return
    metadata_ = metadata[node]
    symbol = metadata_["symbol"]
    start_time = metadata_["start_time"]
    end_time = metadata_["end_time"]
    start_time_str = start_time.strftime(ISO_8601_FORMAT)
    end_time_str = end_time.strftime(ISO_8601_FORMAT)

    row_count = len(data)
    data_start_time = None
    data_end_time = None
    if row_count > 0:
        times = data.index.get_level_values(0)
        data_start_time = times[0]
        data_end_time = times[-1]
        data_start_time_str = data_start_time.strftime(ISO_8601_FORMAT)
        data_end_time_str = data_end_time.strftime(ISO_8601_FORMAT)

        logger.info(f"Writing data for {symbol} for period {start_time_str} to"
                    f" {end_time_str}; {row_count=};"
                    f" data_start_time={data_start_time_str},"
                    f" data_end_time={data_end_time_str}")
        write_data(library, symbol, data, write_mode=WriteMode.UPDATE,
                   create_library=True)
    else:
        logger.info(f"Not writing data for {symbol} for period {start_time_str}"
                    f" to {end_time_str}; no data in time range")

    with sqlalchemy_engine().begin() as conn:
        # TODO: settling period would be more conservative if creation time set to
        #  time just before data_fn called rather than just after
        _insert_dataset_span(conn, dataset_id, symbol, creation_time=datetime.utcnow(),
                             start_time=start_time, end_time=end_time,
                             row_count=len(data), data_start_time=data_start_time,
                             data_end_time=data_end_time)


def _get_load_intervals(
        spans: pd.DataFrame, load_time: DatetimeOrFloat, start_time: DatetimeOrFloat,
        end_time: DatetimeOrFloat, *, reload: bool = False,
        settling_period: Optional[TimedeltaOrFloat] = None,
        buffer_period: Optional[TimedeltaOrFloat] = None) -> pd.DataFrame:
    spans = spans.rename(
        columns={"creation_time": "created_at", "start_time": "start",
                 "end_time": "end"})

    if reload:
        intervals = pd.DataFrame([[start_time, end_time]], columns=["start", "end"])
    else:
        # must do shallow copy so end column of spans isn't modified
        loaded_intervals = spans.copy(deep=False)
        if settling_period is not None:
            loaded_intervals["end"] = np.minimum(
                loaded_intervals["created_at"] - settling_period,
                loaded_intervals["end"])
            loaded_intervals = loaded_intervals[
                loaded_intervals["end"] >= loaded_intervals["start"]]
        loaded_intervals = loaded_intervals.drop(columns=["created_at"])
        intervals = pd.DataFrame([[start_time, end_time]], columns=["start", "end"])
        intervals = interval_difference(intervals, loaded_intervals)

    if buffer_period is not None:
        # use less than below so buffer of 0 has same behaviour as one of none
        buffer = spans.loc[(load_time - spans["created_at"]) <
                           buffer_period][["start", "end"]]
        intervals = interval_difference(intervals, buffer)

    return intervals


def _get_ts_dataset_tasks(conn: Connection, dataset_id: int,
                          symbols: Collection[str], load_time: datetime,
                          start_time: datetime, end_time: datetime,
                          reload: bool = False,
                          span_size: Optional[timedelta] = None,
                          settling_period: Optional[timedelta] = None,
                          buffer_period: Optional[timedelta] = None) -> pd.DataFrame:
    tasks = []
    if span_size is None:
        span_size = end_time - start_time

    spans = _get_dataset_spans(conn, dataset_id, symbols=symbols,
                               start_time=start_time, end_time=end_time)

    for symbol in symbols:
        spans_ = spans[spans["symbol"] == symbol].drop(columns="symbol")
        intervals = _get_load_intervals(
            spans_, load_time, start_time, end_time, reload=reload,
            settling_period=settling_period, buffer_period=buffer_period)
        for i in range(len(intervals)):
            interval_start_time = intervals.iloc[i]["start"]
            interval_end_time = intervals.iloc[i]["end"]
            span_times = get_equally_spaced_times(
                interval_start_time, interval_end_time, period=span_size)
            if (interval_end_time not in span_times or
                    interval_start_time == interval_end_time):
                span_times.append(interval_end_time)
            for j in range(len(span_times) - 1):
                tasks.append({"symbol": symbol,
                              "start_time": span_times[j],
                              "end_time": span_times[j + 1]})
    return pd.DataFrame(tasks, columns=["symbol", "start_time", "end_time"])


def _get_total_load_period_by_symbol(tasks: pd.DataFrame) -> pd.Series:
    tasks["period"] = tasks["end_time"] - tasks["start_time"]
    return tasks.groupby("symbol")["period"].sum()


def _load_ts_data(conn, data_fn: Callable, dataset_name: str, *,
                  symbols: Optional[Collection[str]] = None,
                  start_time: datetime,
                  end_time: Optional[datetime] = None,
                  redownload: bool = False, span_size: Optional[timedelta] = None,
                  settling_period: Optional[timedelta] = None,
                  raise_if_error: bool = False,
                  prioritize_by_coverage: bool = False,
                  buffer_period: Optional[timedelta] = None,
                  allow_data_outside_span_period: bool = False,
                  executor_type: ExecutorType = ExecutorType.SYNC,
                  max_workers: Optional[int] = None,
                  load_time: Optional[datetime] = None):
    """
    Loads data.

    Notes
    -----
    Since when a graph is run with executor type SYNC the nodes will be computed in
    an order determined by a depth-first search, it's guaranteed that each node's data
    will be cleared before running another node; this ensures memory requirements
    are kept as minimal as possible. Running with executor type PROC doesn't currently
    offer this guarantee.
    """
    # TODO: allow executor_type PROC
    if load_time is None:
        load_time = datetime.utcnow()
    if end_time is None:
        end_time = load_time

    logger.info(f"Loading data: {dataset_name};"
                f" load_time={load_time.strftime(ISO_8601_FORMAT)},"
                f" start_time={start_time.strftime(ISO_8601_FORMAT)},"
                f" end_time={end_time.strftime(ISO_8601_FORMAT)},"
                f" {symbols=}, {redownload=}, {span_size=}, {settling_period=},"
                f" {allow_data_outside_span_period=} {prioritize_by_coverage=}")

    dataset = get_dataset(conn, dataset_name)
    if match := re.search(ARCTIC_DATASET_PATH_REGEX, dataset["path"]):
        library = match.group("library")
    else:
        raise ValueError("Error loading data; invalid dataset path")

    tasks = _get_ts_dataset_tasks(conn, dataset_id=dataset["id"], symbols=symbols,
                                  load_time=load_time, start_time=start_time,
                                  end_time=end_time, reload=redownload,
                                  span_size=span_size, settling_period=settling_period,
                                  buffer_period=buffer_period)
    logger.info(f"Number of tasks: {len(tasks)}")
    if len(tasks) == 0:
        return

    if prioritize_by_coverage:
        total_load_period = _get_total_load_period_by_symbol(tasks)
        symbols = total_load_period.sort_values(ascending=False).index.tolist()

    nodes = set()
    metadata = {}
    with set_active_graph(Graph()):
        for symbol in symbols:
            clear_ref: Optional[NodeRef] = None
            tasks_ = tasks[tasks["symbol"] == symbol]
            # unidecode below removes accents
            with set_ns((re.sub("\W", "", unidecode(symbol)), generate_uid(6))):
                for i in range(len(tasks_)):
                    task = tasks_.iloc[i]
                    curr_ref = add_node(
                        partial(
                            _ts_data_fn, data_fn=data_fn,
                            allow_data_outside_span_period=
                            allow_data_outside_span_period),
                        f"{i}/data",
                        args=(task["symbol"],), start_time=task["start_time"],
                        end_time=task["end_time"])
                    metadata[curr_ref] = {"symbol": symbol,
                                          "start_time": task["start_time"],
                                          "end_time": task["end_time"]}
                    if clear_ref is not None:
                        add_edge(clear_ref, curr_ref)
                    clear_ref = add_node(None, f"{i}/clear")
                    add_edge(curr_ref, clear_ref)
            if clear_ref is not None:
                nodes.add(clear_ref)

        # set raise_if_error=False below so graph continues if node fails
        return run_graph(nodes, write_fn=partial(
            _ts_write_fn, dataset_id=dataset["id"], library=library, metadata=metadata),
                         raise_if_error=raise_if_error, executor_type=executor_type,
                         max_workers=max_workers)


# TODO: rename redownload to reload
def load_data(conn, data_fn: Callable, dataset_name: str, *,
              symbols: Optional[Collection[str]] = None,
              start_time: Optional[datetime] = None,
              end_time: Optional[datetime] = None,
              redownload: bool = False,
              span_size: Optional[timedelta] = None,
              settling_period: Optional[timedelta] = None,
              raise_if_error: bool = False,
              prioritize_by_coverage: bool = False,
              buffer_period: Optional[timedelta] = None,
              allow_data_outside_span_period: bool = False,
              executor_type: ExecutorType = ExecutorType.SYNC,
              max_workers: Optional[int] = None,
              load_time: Optional[datetime] = None) -> None:
    """
    Loads data into a dataset.

    Parameters
    ----------
    conn
        SQLAlchemy connection.
    data_fn
        Must accept symbol, start_time and end_time arguments. Must return data from
        start time inclusive to end time exclusive.
    dataset_name
        Dataset name.
    symbols
        Symbols to get data for.
    start_time
        Start time to get data for. Must be specified if dataset is a time series
        dataset.
    end_time
        End time to get data for.
    redownload
        Whether to redownload data for the specified period.
    span_size
        Span size to use for downloading data.
    settling_period
        Settling period to use.
    raise_if_error
        Whether to raise exception if node errors.
    prioritize_by_coverage
        If True, symbols with less coverage are prioritized when downloading data.
    buffer_period
        Minimum period of time between loads of the same data.
    allow_data_outside_span_period
        Indicates whether data function is allowed to return data which falls
        outside of span period.
    executor_type
        Executor type.
    max_workers
        Max workers to use to collect the data.
    load_time
        Load time.
    """
    dataset = get_dataset(conn, dataset_name)
    if dataset["type"] == DatasetType.TIME_SERIES:
        if start_time is None:
            raise ValueError("Error loading data; start time must be provided")
        return _load_ts_data(
            conn, data_fn=data_fn, dataset_name=dataset_name,
            symbols=symbols, start_time=start_time, end_time=end_time,
            redownload=redownload, span_size=span_size,
            settling_period=settling_period,
            raise_if_error=raise_if_error,
            prioritize_by_coverage=prioritize_by_coverage,
            buffer_period=buffer_period,
            allow_data_outside_span_period=allow_data_outside_span_period,
            executor_type=executor_type, max_workers=max_workers,
            load_time=load_time,
        )

    raise ValueError("Error loading data; data can only be loaded currently for time"
                     " series datasets")
