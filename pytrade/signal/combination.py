from typing import Dict

import pandas as pd


def const_factor_combo(factors: pd.DataFrame,
                       weights: Dict[str, float]) -> pd.DataFrame:
    """
    Combines factors using constant weights.

    Parameters
    ----------
    factors
        Factors. Should havae a multi-index of time and factor. Columns should give
        the factor scores for each asset at each time.
    weights
        Weights to give to each factor.
    """
    return sum(weights[x] * factors.xs(x, level=1) for x in weights)
