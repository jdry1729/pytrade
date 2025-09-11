import pandas as pd


def compute_volume_put_call_ratio(put_trades: pd.DataFrame, call_trades: pd.DataFrame,
                                  freq: str):
    """
    Computes volume put-call ratio.

    Parameters
    ----------
    put_trades
        Put trades. Must have datetime index and have "volume" column.
    call_trades
        Call trades. Must have datetime index and "volume" column.
    freq
        Frequency at which to aggregate volume.

    Returns
    -------
    Volume put-call ratio.
    """
    put_volume = put_trades.resample(freq, closed="right", label="right")[
        "volume"].sum()
    call_volume = call_trades.resample(freq, closed="right", label="right")[
        "volume"].sum()
    return put_volume / call_volume
