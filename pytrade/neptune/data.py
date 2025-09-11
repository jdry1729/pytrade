from typing import Optional, Collection, Union, Any, Tuple

import pandas as pd
from pytrade.data.processing import orth, zscore as zscore_


def reindex(data: Union[pd.DataFrame, pd.Series], index: Optional[Any] = None,
            columns: Optional[Any] = None, method: Optional[str] = None,
            limit: Optional[int] = None):
    if method in ["ffill", "bfill"] and limit == 0:
        method, limit = None, None
    return data.reindex(index=index, method=method, limit=limit).reindex(
        columns=columns)


def process_signal(
        signal: pd.DataFrame,
        *,
        ffill_limit: Optional[int] = 0,
        negate: bool = False,
        mask: Optional[pd.DataFrame] = None,
        multiplier: Optional[pd.DataFrame] = None,
        zscore: bool = False,
        clip: Optional[Union[float, Tuple[float, float]]] = None,
        fill_value: Optional[float] = None,
        orth_to: Collection[pd.DataFrame] = (),
) -> pd.DataFrame:
    """
    Applies common transformations to a signal.

    Parameters
    ----------
    signal
        Signal.
    ffill_limit
        Ffill limit to use.
    negate
        Whether to negate the signal.
    mask
        Mask to apply. Signal will be nan where False.
    multiplier
        Multiplier to apply.
    zscore
        Whether to cross-sectionally z-score the signal.
    clip
        Clip threshold to apply.
    fill_value
        Value to use to fill nans (only where mask is False).
    orth_to
        List of factors to orthogonalize to.

    Returns
    -------
    Processed signal.
    """
    if ffill_limit is None or ffill_limit > 0:
        signal = signal.ffill(limit=ffill_limit)

    if negate:
        signal *= -1.0

    if mask is not None:
        signal = signal.where(mask)

    if multiplier is not None:
        signal *= multiplier

    if zscore:
        signal = zscore_(signal, axis=1)

    if clip is not None:
        if isinstance(clip, (float, int)):
            if clip < 0:
                raise ValueError("Error processing signal; clip must be > 0")
            clip = (-clip, clip)
        signal = signal.clip(*clip)

    if fill_value is not None:
        signal = signal.fillna(fill_value)
        if mask is not None:
            signal = signal.where(mask)

    if orth_to:
        signal = orth(signal, orth_to)

    return signal
