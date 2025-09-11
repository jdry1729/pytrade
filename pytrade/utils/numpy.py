import numpy as np


def shift(arr: np.ndarray, periods: int, fill_value: float = np.nan) -> np.ndarray:
    out = np.empty_like(arr)
    if periods > 0:
        out[:periods] = fill_value
        out[periods:] = arr[:-periods]
    elif periods < 0:
        out[periods:] = fill_value
        out[:periods] = arr[-periods:]
    else:
        out[:] = arr
    return out
