import logging
from typing import Callable, Tuple, Any, Optional, Union, Collection

from pytrade.graph import NodeRef, set_ns, add_node

from pytrade.neptune.data import process_signal, reindex
from pytrade.neptune.graph import TIMES, ASSETS, LIVE_TIME
from pytrade.neptune.utils import loc

logger = logging.getLogger(__name__)


def _add_signal_analytics(values: NodeRef) -> None:
    with set_ns("analytics"):
        with set_ns("live"):
            pass


def add_signal(fn: Union[Callable[..., NodeRef], NodeRef],
               *,
               reindex_limit: int = 0,
               ffill_limit: Optional[int] = 0,
               negate: bool = False,
               mask: Optional[NodeRef] = None,
               multiplier: Optional[NodeRef] = None,
               zscore: bool = False,
               clip: Optional[Union[float, Tuple[float, float]]] = None,
               fill_value: Optional[float] = None,
               orth_to: Collection[NodeRef] = (),
               args: Tuple[Any] = (),
               **kwargs) -> NodeRef:
    with set_ns("raw"):
        with set_ns("unaligned"):
            values = fn(*args, **kwargs) if callable(fn) else fn
        values = add_node(reindex, "values", args=(values,),
                          index=TIMES, columns=ASSETS, method="ffill",
                          limit=reindex_limit)

    values = add_node(process_signal, "values", args=(values,),
                      ffill_limit=ffill_limit, negate=negate, mask=mask,
                      multiplier=multiplier, zscore=zscore, clip=clip,
                      fill_value=fill_value, orth_to=orth_to)

    _add_signal_analytics(values)

    with set_ns("live"):
        # fix series name so it can be logged
        add_node(loc, "values", args=(values, LIVE_TIME), series_name="values")

    return values
