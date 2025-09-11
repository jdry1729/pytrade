import pandas as pd
from pytrade.utils.pandas import stack


def infer_q4_items(data: pd.DataFrame) -> pd.DataFrame:
    q123_groupby = data.loc[
        data.index.get_level_values("fiscal_period") != "FY"].groupby(
        level="fiscal_year"
    )
    q123_sum = q123_groupby.sum().where(q123_groupby.count() == 3)

    q4_items = data.xs("FY", level="fiscal_period") - q123_sum
    q4_items = stack([q4_items], keys=["Q4"], names=["fiscal_period"])
    return q4_items.loc[~q4_items.isnull().all(axis=1)]
