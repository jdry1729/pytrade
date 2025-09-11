import numpy as np
import pandas as pd


def _numpy_buffer_signal(signal: np.array, buffer: float = 0.2):
    """
    Modifies the signal so that it is constant if it doesn't change much. This
    is done to minimize trading.

    Parameters
    ----------
    signal: np.array

    Returns
    -------
    Buffered signal.
    """
    T = signal.shape[0]
    buffered_signal = np.empty(shape=signal.shape)
    for i in range(0, T):
        if i == 0:
            buffered_signal[i] = signal[i]
        else:
            buffered_signal[i] = np.minimum(
                np.maximum(buffered_signal[i - 1], signal[i] - buffer),
                signal[i] + buffer)
    return buffered_signal


def _pandas_buffer_signal(signal: pd.DataFrame, buffer: float = 0.2):
    return pd.DataFrame(_numpy_buffer_signal(signal.values, buffer),
                        index=signal.index, columns=signal.columns)


def buffer_signal(signal, buffer: float = 0.2):
    if isinstance(signal, pd.DataFrame):
        return _pandas_buffer_signal(signal, buffer)
    return _numpy_buffer_signal(signal, buffer)


