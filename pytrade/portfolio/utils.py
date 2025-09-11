import numpy as np
import pandas as pd


def reindex_weights_to_returns(weights: pd.DataFrame,
                               returns: pd.DataFrame) -> pd.DataFrame:
    res = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)
    weights_ = None
    times = returns.index
    for time in times:
        if time in weights.index:
            weights_ = weights.loc[time]
            res.loc[time] = weights_
        elif weights_ is not None:
            weights_ = weights_ * (1 + returns.loc[time])
            res.loc[time] = weights_
    return res
