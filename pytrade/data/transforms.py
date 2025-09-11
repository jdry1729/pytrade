from typing import Any

import pandas as pd


# functions in this module are designed to be used in loader pipelines
# maybe we can remove them
def add_constant_column(df: pd.DataFrame, column: str, value: Any):
    df[column] = value
    return df


def reset_index(df: pd.DataFrame):
    return df.reset_index()
