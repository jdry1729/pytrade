import pandas as pd


def signal_to_alpha(signal: pd.DataFrame, asset_vol: pd.DataFrame) -> pd.DataFrame:
    """
    Converts signal to alpha.

    Parameters
    ----------
    signal
        Can have multiindex.
    asset_vol
        Asset vol.

    Returns
    -------
    Alpha.
    """
    return signal.mul(asset_vol, level=0)
