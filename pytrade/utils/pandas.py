import functools
from io import StringIO
from typing import (Union, Optional, Sequence, List, Any, Iterable, Tuple, Dict,
                    Literal, Callable, Collection)

import numpy as np
import pandas as pd
import tabulate as tabulate_
from IPython.display import display as ipython_display

from pytrade.utils.collections import is_iterable_of, ensure_list, contains_duplicates
from pytrade.utils.typing import T1, DataFrameOrSeries


def display(obj: Union[pd.DataFrame, pd.Series], *,
            max_rows: Optional[int] = None,
            max_columns: Optional[int] = None,
            max_colwidth: Optional[int] = None):
    with pd.option_context("display.max_rows", max_rows,
                           "display.max_columns", max_columns,
                           "display.max_colwidth", max_colwidth):
        ipython_display(obj)


def assert_equal(actual, expected):
    if isinstance(actual, pd.Series):
        return pd.testing.assert_series_equal(actual, expected)
    return pd.testing.assert_frame_equal(actual, expected)


def to_frame(obj: Union[pd.DataFrame, pd.Series]):
    if isinstance(obj, pd.Series):
        return obj.to_frame()
    return obj


def get_one_row(df: pd.DataFrame, **kwargs) -> pd.Series:
    # noinspection PyUnresolvedReferences
    df = df.loc[(df[list(kwargs)] == pd.Series(kwargs)).all(axis=1)]

    if df.empty:
        raise ValueError("Error getting row; no rows found")

    if len(df) > 1:
        raise ValueError("Error getting row; multiple rows found")

    return df.iloc[0]


def get_cols_with_value(df: pd.DataFrame, value) -> List[str]:
    cols = []
    for col in df.columns:
        if df[col].eq(value).any():
            cols.append(col)
    return cols


def xs(obj: Union[pd.DataFrame, pd.Series], keys: Collection,
       level: Union[str, int, Sequence[int], Sequence[str]]):
    """
    Slices a dataframe/ series.

    Parameters
    ----------
    obj
        Object to slice.
    keys
        Can be a list of values if a single level is passed. Otherwise should be
        a list of tuples.
    level
        Level to slice.

    Returns
    -------
    Sliced data.
    """
    if isinstance(level, (str, int)):
        level = [level]

    levels = []
    for level_ in level:
        if isinstance(level_, str):
            levels.append(obj.index.names.index(level_))
        else:
            levels.append(level_)

    to_drop = [i for i in range(obj.index.nlevels) if i not in levels]
    # TODO: below won't work if level order isn't same as that in index
    return obj.loc[obj.index.droplevel(to_drop).isin(keys)]


def is_frame_equal(df1, df2, **kwargs):
    try:
        pd.testing.assert_frame_equal(df1, df2, check_names=False, **kwargs)
        return True
    except AssertionError:
        return False


def map_level(index: pd.MultiIndex, level: Union[str, int], arg,
              name: Optional[str] = None) -> pd.MultiIndex:
    # cannot call index.levels.map and then use set_levels to set new levels since
    # this approach won't work if one-to-many mapping between old and new values
    if isinstance(level, str):
        level = index.names.index(level)
    if isinstance(arg, (pd.Series, dict)):
        arg_ = arg
        arg = lambda x: arg_[x]
    index = pd.MultiIndex.from_tuples([tuple(
        arg(v) if i == level else v for i, v in enumerate(x)) for x in index
    ], names=index.names)
    if name is not None:
        index = index.rename(name, level=level)
    return index


def replace_level(index: pd.MultiIndex, values: pd.Index,
                  level: Union[str, int]) -> pd.MultiIndex:
    names = []
    level_values = []
    for i in range(index.nlevels):
        values_ = values if level == i or level == index.names[
            i] else index.get_level_values(i)
        level_values.append(values_)
        names.append(values_.name)
    return pd.MultiIndex.from_arrays(level_values, names=names)


def ensure_index_names(index: pd.Index, levels: Iterable[int] = None):
    if levels is None:
        levels = range(index.nlevels)
    levels = list(levels)

    index = index.copy()
    for i, name in enumerate(index.names):
        if i in levels and name is None:
            index = index.set_names(f"level_{i}", level=i)

    return index


def convert_multiindex_dtypes(index: pd.MultiIndex, dtype, level: Optional = None):
    dtypes = ensure_list(dtype)
    if level is None:
        level = list(range(len(dtype)))

    levels = ensure_list(level)
    names = index.names

    error_msg = "Error converting multiindex levels"
    if not all(x is None or isinstance(x, str) for x in names):
        raise ValueError(f"{error_msg}; index names must be None or str")

    if contains_duplicates(level):
        raise ValueError(f"{error_msg}; levels contains duplicates")

    for level in levels:
        if isinstance(level, int) and level >= index.nlevels:
            raise ValueError(f"{error_msg}; integer level doesn't exist")
        elif isinstance(level, str) and level not in index.names:
            raise ValueError(f"{error_msg}; string level doesn't exist")

    def _get_new_dtype(i: int) -> Optional:
        if i in levels:
            return dtypes[levels.index(i)]
        if names[i] in levels:
            return dtypes[levels.index(names[i])]
        return None

    level_values = []
    for i in range(index.nlevels):
        level_values_ = index.get_level_values(i)
        if (dtype := _get_new_dtype(i)) is not None:
            level_values.append(level_values_.astype(dtype))
        else:
            level_values.append(level_values_)
    return pd.MultiIndex.from_arrays(level_values, names=names)


def invert_series(s: pd.Series):
    if s.index.nlevels > 1:
        raise ValueError("Error inverting series; series has multi-index")
    return pd.Series(s.index.values, index=pd.Index(s.values, name=s.name),
                     name=s.index.name)


# TODO: delete below? don't like it!
def reindex_level(obj: Union[pd.DataFrame, pd.Series], index, level: int = 0,
                  method=None):
    idx = obj.index

    lev_values = []
    for i in range(idx.nlevels):
        if i != level:
            lev_values.append(idx.unique(level=i))
        else:
            lev_values.append(index)

    return obj.reindex(pd.MultiIndex.from_product(lev_values, names=idx.names),
                       method=method)


def align_idx(index: pd.Index, times: pd.DatetimeIndex,
              level: Optional[Union[str, int]] = None, method: str = "bfill"):
    """
    Aligns index to times.

    Parameters
    ----------
    index
        Index to align.
    times
        Times to align to.
    level
        Must be specified if index is a multiindex.
    method
        Method to use when aligning.

    Returns
    -------
    Aligned index.
    """
    values = []
    nlevels = index.nlevels
    if nlevels > 1 and level is None:
        raise ValueError("Error aligning index; level must be specified if index is"
                         " multiindex")

    if isinstance(level, str):
        level = index.names.index(level)

    for i in range(nlevels):
        if nlevels == 1 or i == level:
            indexer = times.get_indexer(
                index.get_level_values(i),
                method=method,
            )
            values.append(times[indexer].where(indexer > -1))
        else:
            values.append(index.get_level_values(i))

    if nlevels == 1:
        return values[0].rename(index.name)
    return pd.MultiIndex.from_arrays(values, names=index.names)


def get_levshape(idx: pd.Index):
    levshape = []
    for i in range(idx.nlevels):
        levshape.append(len(idx.unique(level=i)))
    return tuple(levshape)


def flatten_index(df, axis=0, inplace=False):
    if not inplace:
        df = df.copy()
    if axis == 0:
        if df.index.nlevels > 1:
            df.index = ["_".join([str(y) for y in x]) for x in
                        df.index.to_flat_index()]
    elif axis == 1:
        if df.columns.nlevels > 1:
            df.columns = ["_".join([str(y) for y in x]) for x in
                          df.columns.to_flat_index()]
    else:
        raise ValueError("Axis must be 0 or 1")
    return df


def raise_if_index_not_equal(index_1: pd.Index, index_2: pd.Index) -> None:
    if not index_1.equals(index_2):
        raise ValueError("Indexes aren't equal")


def pandas_to_numpy(obj: Union[DataFrameOrSeries, Sequence[DataFrameOrSeries]]):
    """
    Converts a pandas dataframe/ series to a numpy array.

    Parameters
    ----------
    obj
        The pandas dataframe/ series. Or a sequence of such objects.

    Returns
    -------
    Numpy array.
    """
    if isinstance(obj, (tuple, list)):
        res = tuple(pandas_to_numpy(x) for x in obj)
        if isinstance(obj, list):
            res = list(res)
        return res
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        if obj.index.nlevels > 1:
            shape = get_levshape(obj.index)
            if isinstance(obj, pd.DataFrame):
                shape += (len(obj.columns),)
            return obj.values.reshape(shape)
        return obj.values
    return obj


def stack(objs: Union[Dict[Any, Union[pd.DataFrame, pd.Series]], Sequence[
    Union[pd.DataFrame, pd.Series]]],
          keys: Optional[Sequence] = None,
          names=None, sort_level: Optional[Union[str, int]] = 0,
          sort_remaining: bool = True, axis: int = 0):
    """
    Stacks objs. The new levels are appended to the end of index of the
    resulting obj.

    Parameters
    ----------
    objs
        Objs to stack.
    keys
        Keys to use for each obj.
    names
        Names of new levels.
    sort_level
        Level to sort on.
    sort_remaining
        Whether to sort remaining levels of multiindex after.
    axis
        Axis to stack on.

    Returns
    -------
    Stacked objs.
    """
    if isinstance(objs, dict):
        keys = objs.keys()
        objs = objs.values()

    if keys is None:
        keys = range(len(objs))

    key_len = 1
    if is_iterable_of(keys, (List, Tuple)):
        key_lens = list(set(len(x) for x in keys))
        if len(key_lens) != 1:
            raise ValueError("Keys don't all have same length")
        key_len = key_lens[0]
    new_levels = list(range(key_len))

    data = pd.concat(objs, keys=keys, axis=axis)
    idx = data.index if axis == 0 else data.columns
    idx.set_names(names, level=new_levels, inplace=True)

    axis_kwarg = {}
    if isinstance(data, pd.DataFrame):
        axis_kwarg["axis"] = axis
    new_order = list(range(key_len, idx.nlevels)) + new_levels
    data = data.reorder_levels(new_order, **axis_kwarg)
    if sort_level is not None:
        data = data.sort_index(level=sort_level, sort_remaining=sort_remaining,
                               **axis_kwarg)

    return data


def full(value, columns, index):
    return pd.DataFrame(value, columns=columns, index=index)


def resample(obj: Union[pd.DataFrame, pd.Series], rule: str,
             closed: Optional[Literal["right", "left"]] = "right",
             label: Optional[Literal["right", "left"]] = "right",
             agg_fn: Union[Callable, str] = "last"):
    return obj.resample(rule, closed=closed, label=label).aggregate(agg_fn)


def unstack(obj, level=-1):
    res = {}
    keys = obj.index.unique(level=level)
    for k in keys:
        res[k] = obj.xs(k, level=level)
    return res


def count_nonzero(data, threshold=1e-5, axis=1):
    """
    Computes non-zero positions.

    Parameters
    ----------
    data
        May have multiindex or single index.

    Returns
    -------
    Non-zero count.
    """
    return data.abs().gt(threshold).sum(axis=axis)


def loc(data: pd.DataFrame, idx: Any, name: Optional[str] = None, method=None):
    # TODO: check idx isn't -1!
    idx = data.index.get_indexer([idx], method=method)[0]
    data = data.iloc[idx]
    if name is not None:
        data.name = name
    return data


def tile_series(obj: Union[pd.Series, Dict], index: pd.Index):
    if isinstance(obj, Dict):
        obj = pd.Series(obj)
    return pd.DataFrame(np.tile(obj.values, (len(index), 1)), columns=obj.index,
                        index=index)


def tabulate(df: pd.DataFrame, index=False) -> str:
    df = df.copy()
    df.columns = [x.replace("_", " ").upper() for x in df.columns]
    datetime_cols = df.select_dtypes(include=['datetime64']).columns
    for x in datetime_cols:
        df[x] = df[x].dt.strftime("%Y-%m-%d %H:%M:%S")
    return tabulate_.tabulate(df, headers="keys", tablefmt="plain",
                              showindex=index)


def round_series(s: pd.Series, decimals: Union[int, pd.Series]) -> pd.Series:
    res = s.to_frame().T.round(decimals).squeeze()
    res.name = s.name
    return res


def round_(obj: T1, decimals: Union[float, pd.Series]) -> T1:
    if isinstance(obj, pd.DataFrame):
        return obj.round(decimals)
    return round_series(obj, decimals)


def str_to_pandas(s: str, index_col: Optional[Union[str, Iterable[str]]] = None,
                  squeeze: bool = False, **kwargs) -> Union[pd.DataFrame, pd.Series]:
    """
    Takes a tabular string representation of a dataframe and converts it to
    a pandas dataframe.

    Parameters
    ----------
    s
        The string.
    index_col
        Index to set.
    squeeze
        Whether to squeeze the resulting dataframe to a series.

    Returns
    -------
    Dataframe.

    Notes
    -----
    To parse a time column, pass parse_dates=col_name as an additional keyword argument.
    Alternatively, if a single index col is specified, and it contains times, you can
    just set parse_dates=True.
    """
    df = pd.read_table(StringIO(s), sep="\s+", index_col=index_col, **kwargs)
    if squeeze:
        return df.squeeze()
    return df


def color(obj: pd.DataFrame, cmap="viridis", axis=None):
    return obj.style.background_gradient(cmap=cmap, axis=axis)


# deprecated: use df.query instead
def query(df: pd.DataFrame, f: Dict[str, Any]) -> pd.DataFrame:
    # must reset index to filter on index values
    nlevels = df.index.nlevels
    df = df.reset_index()

    def _reduce(f_: Dict[str, Any], fn) -> Optional:
        out = []
        type_ = list(f_.keys())[0]
        for f_ in f_[type_]:
            if "op" in f_:
                out.append(fn(f_))
            else:
                out_ = _reduce(f_, fn)
                if out_ is not None:
                    out.append(out_)
        if out:
            if type_ == "AND":
                return functools.reduce(lambda x, y: x & y, out)
            elif type_ == "OR":
                return functools.reduce(lambda x, y: x | y, out)
        return None

    # noinspection PyTypeChecker
    def apply(f_: Dict[str, Any]) -> pd.Series:
        op = f_["op"]
        column = f_["column"]
        value = f_.get("value")

        if op == "=":
            return df[column] == value
        elif op == "<=":
            return df[column] <= value
        elif op == ">=":
            return df[column] >= value
        elif op == "<":
            return df[column] < value
        elif op == ">":
            return df[column] > value
        elif op == "!=":
            return df[column] != value
        elif op == "in":
            return df[column].isin(value)
        elif op == "re":
            return df[column].str.match(value)
        elif op == "isna":
            return df[column].isna()

        raise ValueError(f"Unsupported operation: {op}")

    if (res := _reduce(f, apply)) is not None:
        df = df[res]
    return df.set_index(list(df.columns[:nlevels]))


def empty_df(index: Optional = None, columns: Optional = None) -> pd.DataFrame:
    return pd.DataFrame([], index=index, columns=columns)


def empty_series(index: Optional = None, name: Optional[str] = None) -> pd.Series:
    return pd.Series([], index=index, name=name)


def empty_time_idx(name: Optional[str] = "time") -> pd.DatetimeIndex:
    return pd.DatetimeIndex([], name=name)


def empty_idx(name: Optional[str] = None) -> pd.Index:
    return pd.Index([], name=name)


def index_mask(obj, cond):
    obj = obj.copy()
    obj.loc[cond] = np.nan
    return obj


def transform_series(series: pd.Series, transformations: dict) -> pd.Series:
    return pd.Series({key: func(series) for key, func in transformations.items()})


def rescale_series(series: pd.Series, target_mean: float,
                   target_std: float) -> pd.Series:
    return ((series - series.mean()) / series.std()) * target_std + target_mean


def elementwise_min(*args) -> pd.DataFrame:
    """
    Computes element-wise minimum across same-indexed dataframes, ignoring nans.
    """
    return pd.concat(args).groupby(level=list(range(args[0].index.nlevels))).min()
