import enum

import pandas as pd
import numpy as np


class EntryCondition(enum.Enum):
    GREATER_THAN = 0
    LESS_THAN = 1


def apply_mask(data: pd.DataFrame, mask: pd.DataFrame):
    return data[mask]


def entry_exit_mask(data: pd.DataFrame, entry_threshold: float,
                    exit_threshold: float,
                    entry_condition: EntryCondition =
                    EntryCondition.GREATER_THAN):
    """
    Computes entry-exit boolean mask. The mask will have a value of True
    where the data is greater than the entry threshold.

    Parameters
    ----------
    data
        Data.
    entry_threshold
        Entry threshold.
    exit_threshold
        Exit threshold.
    entry_condition
        Entry condition.

    Returns
    -------
    Mask.
    """
    # TODO: assert data doesn't contain nans?
    mask = pd.DataFrame(np.nan, index=data.index, columns=data.columns)
    if entry_condition == EntryCondition.GREATER_THAN:
        mask[data >= entry_threshold] = 1.0
        mask[data <= exit_threshold] = 0.0
    elif entry_condition == EntryCondition.LESS_THAN:
        mask[data <= entry_threshold] = 1.0
        mask[data >= exit_threshold] = 0.0
    return mask.ffill().fillna(0.0).astype(bool)
