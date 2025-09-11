from typing import Optional

import numpy as np
import pandas as pd


def corwin_schultz_bid_ask_spread(low: pd.DataFrame, high: pd.DataFrame,
                                  close: pd.DataFrame, window: int,
                                  min_periods: Optional[int] = None) -> pd.DataFrame:
    """
    Estimates bid/ask spread using Corwin-Schultz 2012 approach.
    """
    gap_up = low - close.shift()
    gap_down = high - close.shift()

    high = high.where(gap_up <= 0, high - gap_up)
    low = low.where(gap_up <= 0, low - gap_up)
    high = high.where(gap_down >= 0, high - gap_down)
    low = low.where(gap_down >= 0, low - gap_down)

    beta = (np.log(high / low) ** 2).rolling(2).sum()
    gamma = np.log(high.rolling(2).max() / low.rolling(2).min()) ** 2

    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / (3 - 2 * np.sqrt(2)) - np.sqrt(
        gamma / (3 - 2 * np.sqrt(2)))

    S = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    return S.where((S > 0) | S.isnull(), 0).rolling(
        window, min_periods=min_periods).mean()
