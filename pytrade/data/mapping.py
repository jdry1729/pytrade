from typing import Collection

import pandas as pd
from pytrade.utils.pandas import empty_df


# TODO: rename to map_data?
def map_ids(data: pd.DataFrame, mappings: pd.DataFrame, on: str) -> pd.DataFrame:
    """
    Maps identifiers in data to an external ID. At each point in time, each internal
    ID must map to at most one external ID.

    Parameters
    ----------
    data
        Must have time column.
    mappings
        Must have start_time and end_time columns, as well as one for the internal IDs
        and one for the external IDs.
    on
        Column to map. Must exist in both data and mappings.
    """
    # TODO: check data doesn't have duplicates on time and on columns?
    if len(mappings.columns) != 4 or not all(
            x in mappings.columns for x in ["start_time", "end_time", on]):
        raise ValueError("Error mapping IDs; mappings must have start and end time"
                         " columns, as well as one for the internal IDs and one for"
                         " the external IDs")
    external_id_col = mappings.columns.difference(["start_time", "end_time", on])[0]

    index_cols = ["time", on]
    mapped = pd.merge(data[index_cols], mappings, on=on)
    mapped = mapped[
        ((mapped["time"] >= mapped["start_time"]) | mapped["start_time"].isnull()) &
        ((mapped["time"] < mapped["end_time"]) | mapped["end_time"].isnull())
        ]
    mapped = mapped.drop(columns=["start_time", "end_time"])
    # below guards against duplicate mapping information
    mapped = mapped.loc[~mapped.duplicated(keep="first")]
    count = mapped.groupby(index_cols).size()
    count = count.loc[count > 1]
    if count.empty:
        return pd.merge(data, mapped, how="left", on=index_cols)
    else:
        errors = ""
        for idx in count.index:
            mapped_ = mapped[(mapped["time"] == idx[0]) & (mapped[on] == idx[1])]
            external_ids = mapped_[external_id_col].unique().tolist()
            errors += (f"- {idx[1]} maps to {', '.join(external_ids)} at time"
                       f" {idx[0]}\n")
        raise ValueError("Internal ID maps to multiple external IDs"
                         f" at single point in time:\n{errors}")


def map_ids_over_time(ids: Collection[str], times: pd.DatetimeIndex,
                      mappings: pd.DataFrame, on: str) -> pd.DataFrame:
    external_id_col = mappings.columns.difference(["start_time", "end_time", on])[0]

    res = []
    for id_ in ids:
        mappings_ = mappings.loc[mappings[on] == id_]
        mapped = map_ids(pd.DataFrame({"time": times, on: id_}), mappings_, on)
        res.append(mapped.set_index("time")[external_id_col].rename(id_))
    return pd.concat(res, axis=1).reindex(columns=ids)


def log_to_period_mappings(mappings: pd.Series) -> pd.DataFrame:
    """
    Converts mappings from log to period format.

    Parameters
    ----------
    mappings
        Log-style mappings. Must have multindex where the first level gives time
        and the second gives internal ID. The name of the series should be name of
        external ID. The first level must be named "time".

    Returns
    -------
    Period-style mappings.
    """
    mappings = mappings.copy()
    internal_id = mappings.index.names[1]
    external_id = mappings.name
    if internal_id is None:
        internal_id = "internal_id"
        mappings.index = mappings.index.set_names(internal_id, level=1)
    if external_id is None:
        external_id = "external_id"
        mappings.name = external_id

    if mappings.empty:
        return empty_df(columns=[internal_id, external_id, "start_time", "end_time"])

    end_times = mappings.groupby(level=1).apply(
        lambda x: x.index.get_level_values(0).to_series().shift(-1))
    end_times = end_times.swaplevel().sort_index()

    mappings = pd.merge(mappings, end_times.rename("end_time"), left_index=True,
                        right_index=True).reset_index()
    mappings = mappings.rename(columns={"time": "start_time"})
    return mappings[[internal_id, external_id, "start_time", "end_time"]]
