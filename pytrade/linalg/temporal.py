import pandas as pd


def mat_vec_mul(m: pd.DataFrame, v: pd.DataFrame, min_count: int = 1) -> pd.DataFrame:
    """
    Performs matrix-vector multiplication at each time step.

    Parameters
    ----------
    m
        Multi-indexed series containing matrix values for each timestep. First level
        of index must represent time, and last level must match columns of v.
    v
        Dataframe storing vector values at each time step.
    min_count
        Minimum count.

    Returns
    -------
    Dataframe containing results of multiplication at each time step.
    """
    names = m.index.names
    v = v.stack(future_stack=True)
    v.index = v.index.rename(names[-1], level=1)
    r = m.mul(v).groupby(names[:-1]).sum(min_count=min_count)
    return r.unstack(level=list(range(1, len(names) - 1)))
