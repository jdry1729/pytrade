import functools
import logging
import re
import sys
from contextlib import contextmanager
from dataclasses import dataclass, fields
from datetime import datetime
from typing import Optional, Union, Dict, Iterable, Any, Collection, List, Sequence
from warnings import simplefilter

import pandas as pd
from arcticdb import Arctic, QueryBuilder
from arcticdb.version_store.library import (StagedDataFinalizeMethod, SymbolDescription,
                                            Library, SymbolVersion, VersionInfo)
from arcticdb_ext.exceptions import ArcticException
from arcticdb_ext.version_store import NoSuchVersionException, ExpressionNode
from pytrade.data.postgres import WriteMode
from pytrade.utils import stack, str_to_num
from pytrade.utils.collections import replace, is_collection_of, \
    ensure_list
from pytrade.utils.profile import load_profile
from tqdm import tqdm

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

logger = logging.getLogger(__name__)

_CONTEXT_STACK: List["Context"] = []

ARCTIC_REF_REGEX = "(?P<symbol>[A-Za-z0-9\-\_\/\=]+):(?P<library>[" \
                   "A-Za-z0-9\-\_\/\=]*)(@(?P<as_of>[A-Za-z0-9\-\_\/\=]*))?"


@dataclass
class Context:
    library: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    snap: Optional[str] = None
    use_cache: bool = False


@contextmanager
def set_ctx(library: Optional[str] = None, *, start_time: Optional[datetime] = None,
            end_time: Optional[datetime] = None, snap: Optional[str] = None,
            use_cache: bool = False):
    _CONTEXT_STACK.append(Context(
        library=library, start_time=start_time, end_time=end_time, snap=snap,
        use_cache=use_cache))
    yield
    _CONTEXT_STACK.pop()


def _merge_contexts(contexts: Sequence[Context]) -> Context:
    kwargs = {}
    fields_ = [x.name for x in fields(Context)]
    for field in fields_:
        for context in contexts[::-1]:
            if (value := getattr(context, field)) is not None:
                kwargs[field] = value
                break
    return Context(**kwargs)


def get_ctx() -> Context:
    return _merge_contexts(_CONTEXT_STACK)


@dataclass
class ArcticRef:
    symbol: str
    library: str
    as_of: Optional[Union[str, int, datetime]] = None

    @staticmethod
    def from_str(s: str):
        if match := re.search(ARCTIC_REF_REGEX, s):
            as_of = None
            symbol = match.group("symbol")
            library = match.group("library")
            if "as_of" in match.groupdict():
                as_of = match.group("as_of")
                as_of = str_to_num(as_of)
            return ArcticRef(
                symbol=symbol,
                library=library,
                as_of=as_of
            )
        raise ValueError("Error creating arctic ref")


@functools.lru_cache(maxsize=None)
def _get_arctic_client(uri: str):
    return Arctic(uri)


def get_arctic_client(profile: Optional[str] = None):
    profile = load_profile(profile)
    return _get_arctic_client(profile.arctic_uri)


def get_arctic_lib(name: str, create: bool = False, profile: Optional[str] = None):
    arctic = get_arctic_client(profile)
    if create and not arctic.has_library(name):
        logger.info(f"Creating arctic library: {name}")
        arctic.create_library(name)
    return arctic.get_library(name)


def ensure_lib_exists(name: str, profile: Optional[str] = None) -> bool:
    arctic = get_arctic_client(profile)
    if not arctic.has_library(name):
        logger.info(f"Creating arctic library: {name}")
        arctic.create_library(name)
        return True
    return False


def delete_lib(library: str, profile: Optional[str] = None):
    arctic = get_arctic_client(profile)
    arctic.delete_library(library)


def delete_symbol(library: str, symbol: str,
                  versions: Optional[Union[int, Iterable[int]]] = None,
                  profile: Optional[str] = None):
    logger.info(f"Deleting symbol; {library=}, {symbol=}")
    lib = get_arctic_lib(library, profile=profile)
    lib.delete(symbol, versions)


def get_description(library: str, symbol: str) -> SymbolDescription:
    lib = get_arctic_lib(library)
    return lib.get_description(symbol)


def delete_data(library: str, symbol: Optional[Union[str, Collection[str]]] = None,
                start_time: Optional[datetime] = None,
                end_time: Optional[datetime] = None, profile: Optional[str] = None):
    lib = get_arctic_lib(library, profile=profile)

    if symbol is None:
        symbols = list_symbols(library, profile=profile)
    else:
        symbols = ensure_list(symbol)

    date_range = None
    if start_time is not None or end_time is not None:
        date_range = (start_time, end_time)

    for symbol in symbols:
        if has_symbol(library, symbol, profile=profile):
            if date_range is not None:
                logger.info(f"Deleting data for symbol: {symbol};"
                            f" start_time={start_time}, end_time={end_time}")
                lib.delete_data_in_range(symbol, date_range)
            else:
                logger.info(f"Deleting data for symbol: {symbol}")
                lib.delete(symbol)


def copy_data(src: str, dest: str, symbols: Optional[Collection[str]] = None,
              batch_size: Optional[int] = None) -> None:
    # TODO: allow only subset of data to be copied
    ensure_lib_exists(dest)
    if symbols is None:
        symbols = list_symbols(src)

    for symbol in symbols:
        # TODO: add option not to delete existing data
        if has_symbol(dest, symbol):
            logger.info(f"Deleting existing symbol: {symbol}")
            delete_symbol(dest, symbol)

        row_count = get_description(src, symbol).row_count
        batch_size_ = row_count
        if batch_size is not None:
            batch_size_ = batch_size

        for i in range(0, row_count, batch_size_):
            start_index = i
            end_index = i + batch_size_
            logger.info(f"Copying symbol: {symbol}; {start_index=}, {end_index=}")
            data = read_data(src, symbol, start_index=start_index, end_index=end_index)
            write_data(dest, symbol, data, write_mode=WriteMode.UPDATE)


def get_finalize_method(write_mode: WriteMode):
    if write_mode == WriteMode.PARALLEL_WRITE:
        return StagedDataFinalizeMethod.WRITE
    return StagedDataFinalizeMethod.APPEND


def _normalize_index_dtypes(idx: pd.Index) -> pd.Index:
    if isinstance(idx, pd.MultiIndex):
        level_values = []
        nlevels = idx.nlevels
        names = idx.names
        for i in range(nlevels):
            level_values_ = idx.get_level_values(i)
            if level_values_.dtype == "string":
                level_values.append(level_values_.astype(str))
            elif level_values_.dtype == "Int64":
                level_values.append(level_values_.astype(str))
            else:
                level_values.append(level_values_)
        return pd.MultiIndex.from_arrays(level_values, names=names)
    if idx.dtype == "string":
        return idx.astype(str)
    elif idx.dtype == "Int64":
        return idx.astype(str)
    return idx


def _restore_index_dtypes(idx: pd.Index, dtypes) -> pd.Index:
    if isinstance(idx, pd.MultiIndex):
        level_values = []
        nlevels = idx.nlevels
        names = idx.names
        for i in range(nlevels):
            level_values_ = idx.get_level_values(i)
            if dtypes.iloc[i] == "string":
                level_values.append(pd.Index(replace(
                    level_values_, {"<NA>": pd.NA}), dtype="string"))
            elif dtypes.iloc[i] == "Int64":
                level_values.append(pd.Index(replace(
                    level_values_, {"<NA>": pd.NA}), dtype="Int64"))
            else:
                level_values.append(level_values_)
        return pd.MultiIndex.from_arrays(level_values, names=names)
    if dtypes == "string":
        return pd.Index(replace(idx, {"<NA>": pd.NA}), dtype="string")
    elif dtypes == "Int64":
        return pd.Index(replace(idx, {"<NA>": pd.NA}), dtype="Int64")
    return idx


def _normalize_column_dtypes(obj: Union[pd.DataFrame, pd.Series]):
    if isinstance(obj, pd.DataFrame):
        dtypes = obj.dtypes
        string_dtypes = dtypes[(dtypes == "string")]
        obj[string_dtypes.index] = obj[string_dtypes.index].astype(str)
        int64_dtypes = dtypes[(dtypes == "Int64")]
        obj[int64_dtypes.index] = obj[int64_dtypes.index].astype(str)
    elif isinstance(obj, pd.Series):
        if obj.dtype == "string":
            obj = obj.astype(str)
        elif obj.dtype == "Int64":
            obj = obj.astype(str)
    return obj


def _restore_column_dtypes(obj: Union[pd.DataFrame, pd.Series], dtypes: pd.Series):
    if isinstance(obj, pd.DataFrame):
        # must slice dtypes below since obj might not contain all columns
        dtypes = dtypes.loc[obj.columns]
        string_dtypes = dtypes[(dtypes == "string")]
        obj[string_dtypes.index] = obj[string_dtypes.index].replace(
            "<NA>", pd.NA).astype("string")
        int64_dtypes = dtypes[(dtypes == "Int64")]
        obj[int64_dtypes.index] = obj[int64_dtypes.index].replace(
            "<NA>", pd.NA).astype("Int64")
    elif isinstance(obj, pd.Series):
        if dtypes == "string":
            obj = obj.replace("<NA>", pd.NA).astype("string")
        elif dtypes == "Int64":
            obj = obj.replace("<NA>", pd.NA).astype("Int64")
    return obj


def _normalize_data(obj: Union[pd.DataFrame, pd.Series]):
    # shallow copy so original obj isn't modified
    obj = obj.copy(deep=False)
    obj.index = _normalize_index_dtypes(obj.index)
    obj = _normalize_column_dtypes(obj)
    if isinstance(obj, pd.Series) and isinstance(obj.name, datetime):
        # arctic can't store series with datetime name
        obj.name = None
    return obj


def _restore_data(obj, column_dtypes, index_dtypes, series_name: Optional = None):
    # ok to modify obj inplace here
    obj = _restore_column_dtypes(obj, column_dtypes)
    obj.index = _restore_index_dtypes(obj.index, index_dtypes)
    if isinstance(obj, pd.Series) and series_name is not None:
        obj.name = series_name
    return obj


def _read_and_restore_data(library: Library, symbol: str, **kwargs):
    item = library.read(symbol, **kwargs)
    metadata = item.metadata
    data = item.data
    if metadata is not None:
        if "_column_dtypes" in metadata and "_index_dtypes" in metadata:
            data = _restore_data(data, metadata["_column_dtypes"],
                                 metadata["_index_dtypes"],
                                 metadata.get("_series_name"))
    return data


def write_data(library: str, symbol: str, data,
               write_mode: WriteMode = WriteMode.REPLACE,
               metadata: Optional[Dict] = None,
               create_library: bool = True, prune_previous_versions: bool = False,
               profile: Optional[str] = None) -> int:
    library = get_arctic_lib(library, create=create_library, profile=profile)

    if metadata is not None:
        # must copy metadata!
        metadata = metadata.copy()
    else:
        metadata = {}

    if isinstance(data, (pd.DataFrame, pd.Series)):
        metadata["_column_dtypes"] = data.dtypes
        if isinstance(data.index, pd.MultiIndex):
            metadata["_index_dtypes"] = data.index.dtypes
        else:
            metadata["_index_dtypes"] = data.index.dtype
        if isinstance(data, pd.Series):
            metadata["_series_name"] = data.name
        data = _normalize_data(data)

    if write_mode == WriteMode.UPDATE:
        # TODO: check dtypes unchanged since prev write
        item = library.update(symbol, data, upsert=True, metadata=metadata,
                              prune_previous_versions=prune_previous_versions)
    elif write_mode == WriteMode.APPEND:
        # TODO: check dtypes unchanged since prev write
        item = library.append(symbol, data, metadata=metadata,
                              prune_previous_versions=prune_previous_versions)
    elif write_mode == WriteMode.REPLACE:
        try:
            item = library.write(symbol, data, metadata=metadata,
                                 prune_previous_versions=prune_previous_versions)
        except ArcticException:
            item = library.write_pickle(symbol, data, metadata=metadata)
    elif write_mode == WriteMode.PARALLEL_WRITE:
        item = library.write(symbol, data, metadata=metadata, staged=True,
                             prune_previous_versions=prune_previous_versions)
    else:
        raise ValueError("Error writing data; write_mode must be: UPDATE, APPEND,"
                         " REPLACE or PARALLEL_WRITE")

    return item.version


def read_metadata(library: str, symbol: str, as_of: Optional[datetime] = None,
                  profile: Optional[str] = None) -> Dict:
    library = get_arctic_lib(library, profile=profile)
    return library.read_metadata(symbol, as_of=as_of).metadata


def read_head(library: str, symbol: str,
              n: int = 5,
              as_of: Optional[datetime] = None,
              columns: Optional[Iterable[str]] = None,
              profile: Optional[str] = None) -> Any:
    library = get_arctic_lib(library, profile=profile)
    return library.head(symbol, n, as_of=as_of, columns=columns).data


def read_tail(library: str, symbol: str,
              n: int = 5,
              as_of: Optional[datetime] = None,
              columns: Optional[Iterable[str]] = None,
              profile: Optional[str] = None) -> Any:
    library = get_arctic_lib(library, profile=profile)
    return library.tail(symbol, n, as_of=as_of, columns=columns).data


def build_query_from_dict(d: Dict[str, Any]) -> QueryBuilder:
    q = QueryBuilder()

    def _apply(f: Dict[str, Any]) -> ExpressionNode:
        op = f["op"]
        column = f["column"]
        value = f["value"]

        if op == "=":
            return q[column] == value
        elif op == "<=":
            return q[column] <= value
        elif op == ">=":
            return q[column] >= value
        elif op == "<":
            return q[column] < value
        elif op == ">":
            return q[column] > value
        elif op == "!=":
            return q[column] != value
        elif op == "in":
            return q[column].isin(value)

        raise ValueError(f"Unsupported operation: {op}")

    def _reduce(f: Dict[str, Any]) -> Optional:
        out = []
        type_ = list(f.keys())[0]
        for f_ in f[type_]:
            if "op" in f_:
                out.append(_apply(f_))
            else:
                out_ = _reduce(f_)
                if out_ is not None:
                    out.append(out_)
        if out:
            if type_ == "AND":
                return functools.reduce(lambda x, y: x & y, out)
            elif type_ == "OR":
                return functools.reduce(lambda x, y: x | y, out)
        return None

    if (res := _reduce(d)) is not None:
        return q[res]
    return q


def _r(symbol: Union[str, Iterable[str]],
       library: Optional[str] = None,
       as_of: Optional[datetime] = None,
       start_time: Optional[datetime] = None,
       end_time: Optional[datetime] = None,
       start_index: Optional[int] = None,
       end_index: Optional[int] = None,
       columns: Optional[Iterable[str]] = None,
       query: Optional[QueryBuilder] = None,
       keys: Optional[Iterable[Union[str, float]]] = None,
       names: Optional[Iterable[str]] = None,
       profile: Optional[str] = None,
       show_progress: bool = False,
       allow_missing: bool = False,
       **kwargs):
    ctx = get_ctx()
    if library is None:
        library = ctx.library
    if start_time is None:
        start_time = ctx.start_time
    if end_time is None:
        end_time = ctx.end_time
    if as_of is None:
        as_of = ctx.snap

    if library is None:
        raise ValueError("Error reading data; no library specified")

    data = read_data(library, symbol,
                     as_of=as_of,
                     start_time=start_time,
                     end_time=end_time,
                     start_index=start_index,
                     end_index=end_index,
                     columns=columns,
                     query=query,
                     keys=keys,
                     names=names,
                     profile=profile,
                     show_progress=show_progress,
                     allow_missing=allow_missing,
                     **kwargs)

    if isinstance(data, ArcticRef):
        data = read_data(data.library, data.symbol, start_time=start_time,
                         end_time=end_time, start_index=start_index,
                         end_index=end_index, columns=columns, as_of=data.as_of,
                         query=query, profile=profile, **kwargs)
    elif is_collection_of(data, ArcticRef):
        data = tuple([
            read_data(e.library, e.symbol, start_time=start_time,
                      end_time=end_time, start_index=start_index,
                      end_index=end_index, columns=columns, as_of=e.as_of,
                      query=query, profile=profile) for e in data])

    return data


@functools.lru_cache()
def _r_cache(symbol: Union[str, Iterable[str]],
             library: Optional[str] = None,
             as_of: Optional[datetime] = None,
             start_time: Optional[datetime] = None,
             end_time: Optional[datetime] = None,
             start_index: Optional[int] = None,
             end_index: Optional[int] = None,
             columns: Optional[Iterable[str]] = None,
             query: Optional[QueryBuilder] = None,
             keys: Optional[Iterable[Union[str, float]]] = None,
             names: Optional[Iterable[str]] = None,
             profile: Optional[str] = None,
             show_progress: bool = False,
             allow_missing: bool = False,
             **kwargs):
    return _r(symbol=symbol, library=library, as_of=as_of, start_time=start_time,
              end_time=end_time, start_index=start_index, end_index=end_index,
              columns=columns, query=query, keys=keys, names=names, profile=profile,
              show_progress=show_progress, allow_missing=allow_missing,
              **kwargs)


def r(symbol: Union[str, Iterable[str]],
      library: Optional[str] = None,
      as_of: Optional[datetime] = None,
      start_time: Optional[datetime] = None,
      end_time: Optional[datetime] = None,
      start_index: Optional[int] = None,
      end_index: Optional[int] = None,
      columns: Optional[Iterable[str]] = None,
      query: Optional[QueryBuilder] = None,
      keys: Optional[Iterable[Union[str, float]]] = None,
      names: Optional[Iterable[str]] = None,
      profile: Optional[str] = None,
      show_progress: bool = False,
      allow_missing: bool = False,
      use_cache: Optional[bool] = None,
      **kwargs):
    ctx = get_ctx()
    if use_cache is None:
        # use context use_cache attribute if not specified
        use_cache = ctx.use_cache

    kwargs_ = {
        "symbol": symbol,
        "library": library,
        "as_of": as_of,
        "start_time": start_time,
        "end_time": end_time,
        "start_index": start_index,
        "end_index": end_index,
        "columns": columns,
        "query": query,
        "keys": keys,
        "names": names,
        "profile": profile,
        "show_progress": show_progress,
        "allow_missing": allow_missing,
        **kwargs
    }
    for x in ["symbol", "columns", "keys", "names"]:
        if isinstance(kwargs_[x], list):
            kwargs_[x] = tuple(kwargs_[x])

    if use_cache:
        return _r_cache(**kwargs_)
    return _r(**kwargs_)


def read_data(library: str,
              symbol: Union[str, Iterable[str]],
              as_of: Optional[datetime] = None,
              start_time: Optional[datetime] = None,
              end_time: Optional[datetime] = None,
              start_index: Optional[int] = None,
              end_index: Optional[int] = None,
              columns: Optional[Iterable[str]] = None,
              query: Optional[QueryBuilder] = None,
              keys: Optional[Iterable[Union[str, float]]] = None,
              names: Optional[Iterable[str]] = None,
              profile: Optional[str] = None,
              show_progress: bool = False,
              allow_missing: bool = False,
              **kwargs) -> Any:
    """
    Reads data.

    Parameters
    ----------
    library
        Library to read from.
    symbol
        Symbol to read.
    as_of
        As of.
    start_time
        Start time (inclusive).
    end_time
        End time (inclusive).
    start_index
    end_index
    columns
    query
    keys
    names
    profile
    show_progress
    allow_missing
    kwargs

    Returns
    -------
    Data.
    """
    if isinstance(symbol, (list, tuple, set)):
        if not symbol:
            raise ValueError("Error reading data; symbol is empty collection")

    date_range = None
    if start_time is not None or end_time is not None:
        date_range = (start_time, end_time)

    row_range = None
    if start_index is not None or end_index is not None:
        if start_index is None:
            start_index = 0
        elif end_index is None:
            end_index = sys.maxsize
        row_range = (start_index, end_index)

    if keys is not None:
        keys = list(keys)
    if names is not None:
        names = list(names)

    q = query
    if q is None:
        q = QueryBuilder()
    for k, v in kwargs.items():
        if isinstance(v, Iterable):
            q = q[q[k].isin(v)]
        else:
            q = q[q[k] == v]

    kwargs = dict(as_of=as_of, date_range=date_range, row_range=row_range,
                  columns=columns, query_builder=q)
    library = get_arctic_lib(library, profile=profile)
    if isinstance(symbol, str):
        if library._nvs.is_symbol_pickled(symbol, as_of=as_of):
            kwargs["date_range"] = None
            kwargs["row_range"] = None
            kwargs["columns"] = None
            kwargs["query_builder"] = None
        data = _read_and_restore_data(library, symbol, **kwargs)
    else:
        data = {}
        symbols = symbol
        if keys is not None and (len(symbols) != len(keys)):
            raise ValueError("Error reading data; symbols and keys must be same length")

        for i, symbol in enumerate(tqdm(symbols, disable=not show_progress)):
            # TODO: in arcticdb 4.5.0 you cannot re-use query builder instances and
            #  specify date range: https://github.com/man-group/ArcticDB/issues/1788;
            #  will pin 4.4.3 until this is resolved
            key = keys[i] if keys is not None else symbol
            try:
                data_ = _read_and_restore_data(library, symbol, **kwargs)
            except NoSuchVersionException:
                if allow_missing:
                    logger.info(f"No data found for symbol: {symbol}")
                    continue
                raise
            if not data_.empty:
                data[key] = data_
        data = stack(data, names=names)

    return data


def has_symbol(library: str, symbol: str,
               as_of: Optional[Union[int, str, datetime]] = None,
               profile: Optional[str] = None):
    library = get_arctic_lib(library, profile=profile)
    return library.has_symbol(symbol, as_of=as_of)


def ls(library: Optional[str] = None, regex: Optional[str] = None,
       snap: Optional[str] = None, profile: Optional[str] = None) -> List[str]:
    ctx = get_ctx()
    if library is None:
        library = ctx.library
    if snap is None:
        snap = ctx.snap

    if library is None:
        raise ValueError("Error listing symbols; no library specified")

    library = get_arctic_lib(library, profile=profile)
    return library.list_symbols(snap, regex=regex)


def lv(library: Optional[str] = None, symbol: Optional[str] = None,
       snap: Optional[str] = None, profile: Optional[str] = None) -> Dict[
    SymbolVersion, VersionInfo]:
    ctx = get_ctx()
    if library is None:
        library = ctx.library
    if snap is None:
        snap = ctx.snap

    if library is None:
        raise ValueError("Error listing symbols; no library specified")

    library = get_arctic_lib(library, profile=profile)
    return library.list_versions(symbol, snapshot=snap)


def md(symbol: str, library: Optional[str] = None, show_hidden: bool = False,
       as_of: Optional[datetime] = None, profile: Optional[str] = None) -> Dict:
    ctx = get_ctx()
    if library is None:
        library = ctx.library
    if as_of is None:
        as_of = ctx.snap

    if library is None:
        raise ValueError("Error reading metadata; no library specified")

    library = get_arctic_lib(library, profile=profile)
    metadata = library.read_metadata(symbol, as_of=as_of).metadata
    if not show_hidden:
        metadata = {k: v for k, v in metadata.items() if not k.startswith("_")}
    return metadata


def list_snaps(library: Optional[str] = None, regex: Optional[str] = None,
               ascending: bool = True, profile: Optional[str] = None) -> List[str]:
    ctx = get_ctx()
    if library is None:
        library = ctx.library

    if library is None:
        raise ValueError("Error listing symbols; no library specified")

    library = get_arctic_lib(library, profile=profile)
    snaps = list(library.list_snapshots().keys())

    if regex is not None:
        snaps = [x for x in snaps if re.match(regex, x)]

    snaps = sorted(snaps, reverse=not ascending)
    return snaps


def snap(name: str, library: Optional[str] = None,
         symbols: Optional[Collection[str]] = None, profile: Optional[str] = None):
    ctx = get_ctx()
    if library is None:
        library = ctx.library

    library = get_arctic_lib(library, profile=profile)

    skip_symbols = []
    if symbols is not None:
        all_symbols = library.list_symbols()
        skip_symbols = [x for x in all_symbols if x not in symbols]
    return library.snapshot(name, skip_symbols=skip_symbols)


def delete_snap(name: str, library: Optional[str] = None,
                profile: Optional[str] = None):
    ctx = get_ctx()
    if library is None:
        library = ctx.library

    library = get_arctic_lib(library, profile=profile)

    return library.delete_snapshot(name)


def add_to_snap(symbol: str, snap: str, library: Optional[str] = None, *,
                version: Optional[int] = None, create: bool = False,
                profile: Optional[str] = None):
    ctx = get_ctx()
    if library is None:
        library = ctx.library

    library = get_arctic_lib(library, profile=profile)
    snaps = library.list_snapshots()
    if version is None:
        versions = library.list_versions(symbol, latest_only=True)
        version = list(versions.keys())[0].version
    if snap in snaps:
        return library._nvs.add_to_snapshot(snap, [symbol], [version])
    elif create:
        return library.snapshot(snap, versions={symbol: version})
    raise ValueError("Error adding to snap; snap doesn't exist and create=False")


def get_latest_version(symbol: str, library: Optional[str] = None,
                       profile: Optional[str] = None):
    ctx = get_ctx()
    if library is None:
        library = ctx.library

    library = get_arctic_lib(library, profile=profile)
    versions = library.list_versions(symbol, latest_only=True)
    return list(versions.keys())[0].version


def list_symbols(library: str, snap: Optional[str] = None,
                 profile: Optional[str] = None):
    library = get_arctic_lib(library, profile=profile)
    return library.list_symbols(snap)


def log(symbol: str, library: str,
        log_symbol: Optional[str] = None,
        log_library: Optional[str] = None,
        as_of: Optional[Union[str, int]] = None,
        time: Optional[datetime] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None, start_index: Optional[int] = None,
        end_index: Optional[int] = None,
        stack: bool = False) -> None:
    """
    Logs a symbol.

    Parameters
    ----------
    symbol
        Symbol to log.
    library
        Library of symbol to log.
    log_symbol
        Symbol to log to.
    log_library
        Library of symbol to log to.
    as_of
    time
        Log time.
    start_time
    end_time
    start_index
    end_index
    stack
        Boolean indicating whether to log a stacked version of symbol. This may be
        necessary if the schema of the symbol will change and the library to which
        the data is being logged doesn't have dynamic schema enabled.
    """
    if log_symbol is None:
        log_symbol = f"log/{symbol}"
    if log_library is None:
        log_library = library

    if time is None:
        time = datetime.now()

    if isinstance(as_of, int):
        version = as_of
    else:
        version = max([x.version for x in lv(library, symbol, snap=as_of)])

    data = r(symbol, library, as_of=as_of, start_time=start_time,
             end_time=end_time, start_index=start_index, end_index=end_index)

    if not isinstance(data, (pd.DataFrame, pd.Series)):
        raise ValueError(
            "Error logging data; only pandas dataframes/ series can be logged")

    if stack:
        data = data.stack(future_stack=True)

    N = len(data)
    levels = [N * [time], N * [version]]
    names = ["_time", "_version"]
    for i in range(data.index.nlevels):
        levels.append(data.index.get_level_values(i))
        names.append(data.index.names[i])
    data.index = pd.MultiIndex.from_arrays(levels, names=names)

    return write_data(log_library, log_symbol, data, write_mode=WriteMode.APPEND,
                      create_library=True)
