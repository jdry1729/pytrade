import logging
import random
import time
from functools import partial
from typing import Iterable, Callable, Optional, Union, Tuple

logger = logging.getLogger(__name__)


def _retry_inner(f, exceptions=Exception, tries=-1, delay=0, max_delay=None,
                 backoff=1, jitter=0, callback_fns: Iterable[Callable] = ()):
    """
    Executes a function and retries it if it failed.
    """
    _tries, _delay = tries, delay
    while _tries:
        try:
            return f()
        except exceptions as e:

            for callback_fn in callback_fns:
                callback_fn()

            _tries -= 1
            if not _tries:
                raise

            logger.warning(f"{e}, retrying in {_delay} seconds...")

            time.sleep(_delay)
            _delay *= backoff

            if isinstance(jitter, tuple):
                _delay += random.uniform(*jitter)
            else:
                _delay += jitter

            if max_delay is not None:
                _delay = min(_delay, max_delay)


# TODO: make below function nicer! at the moment I've basically copied it
#  from the retry package and added callback_fns, but the code should be
#  tidied
def retry_call(f, fargs=None, fkwargs=None, exceptions=Exception, tries=-1,
               delay=0, max_delay=None, backoff=1,
               jitter=0, callback_fns: Iterable[Callable] = ()):
    """
    Calls a function and re-executes it if it failed.
    """
    args = fargs if fargs else list()
    kwargs = fkwargs if fkwargs else dict()
    return _retry_inner(partial(f, *args, **kwargs), exceptions, tries,
                        delay, max_delay, backoff, jitter, callback_fns)


# TODO: remove e arg - very confusing!
def retry(fn: Callable, initial_interval: int = 1, max_interval: Optional[int] = 60,
          max_tries: Optional[int] = 3, multiplier: float = 2,
          pre_retry: Optional[Callable[[int], None]] = None, e: Optional[
            Union[Tuple[Exception], Callable[[Exception], bool]]] = None,
          condition: Optional[
              Union[Tuple[Exception], Callable[[Exception], bool]]] = None,
          args: Iterable = (),
          format_fn: Optional[Callable[[Exception], str]] = None,
          **kwargs):
    if e is None:
        e = condition
    tries = 0
    interval = initial_interval
    while True:
        try:
            return fn(*args, **kwargs)
        except Exception as e_:
            if e is not None:
                if callable(e):
                    if not e(e_):
                        raise e_
                elif not isinstance(e_, e):
                    raise e_
            tries += 1
            e_str = f"{e_}"
            if format_fn is not None:
                e_str = format_fn(e_)
            logger.warning(f"Error making function call: {e_str}")
        if max_tries is not None:
            if tries >= max_tries:
                raise ValueError("Error retrying function call; max retries exceeded")
        if pre_retry is not None:
            pre_retry(tries - 1)
        logger.info(f"Retrying function call in: {interval}s")
        time.sleep(interval)
        interval *= multiplier
        if max_interval is not None:
            interval = min(interval, max_interval)
