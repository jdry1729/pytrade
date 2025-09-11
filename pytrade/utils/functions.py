# TODO: surely functools has a compose method?
from dataclasses import dataclass, field
from typing import List, Callable, Iterable, Dict, Any, Tuple, Optional


def raise_(e):
    raise e


def identity(x):
    return x


def coalesce(*args):
    for arg in args:
        if arg is not None:
            return arg
    return None


def safe_call(fn: Callable, default: Optional = None, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception:
        return default


def partial(func, *args, **kwargs):
    """
    Needed when you want to use functools.partial, but need it to return a
    function rather than an object.
    """

    def wrapper(*extra_args, **extra_kwargs):
        new_args = list(args)
        new_args.extend(extra_args)
        new_kwargs = dict(kwargs)
        new_kwargs.update(extra_kwargs)
        return func(*new_args, **new_kwargs)

    return wrapper


def map_args_and_kwargs(map_fn, args, kwargs):
    args = [map_fn(x) for x in args]
    kwargs = {k: map_fn(v) for k, v in kwargs.items()}
    return args, kwargs


def compose(fns: List[Callable]):
    def inner(*args, **kwargs):
        out = None
        for i, fn in enumerate(fns):
            if i == 0:
                out = fn(*args, **kwargs)
            else:
                out = fn(out)
        return out

    return inner


def check(fn: Callable, check_fns: Iterable[Callable], *args, **kwargs):
    """
    Checks the output of a function.

    Parameters
    ----------
    fn
        Function to check the output of.
    check_fns
        Check functions. Each receives the output of fn as input, and should
        raise an exception upon failure.
    args
        Positional arguments to pass to fn.
    kwargs
        Keyword arguments to pass to fn.

    Returns
    -------
    The output of the function, or raises an Exception if a check fails.
    """
    res = fn(*args, **kwargs)
    for check_fn in check_fns:
        check_fn(data=res)
    return res
