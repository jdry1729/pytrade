from typing import Optional, Dict, Collection
from typing import Union, Any, Set

import pandas as pd
from pytrade.data import write_data, read_data
from pytrade.data.arctic import ArcticRef, ensure_lib_exists, add_to_snap, list_snaps
from pytrade.data.arctic import has_symbol
from pytrade.graph import ExecutorType, NodeRefOrStr, n, \
    GraphRun, get_active_graph, Graph, add_node, NodeRef
from pytrade.utils.collections import ensure_set, is_tuple_of, inv_many_to_one
from pytrade.utils.functions import partial

_LIBRARY_MAP: Dict[Graph, str] = {}

# assets is a pandas index giving the assets in the universe
ASSETS = n("/assets")

# asset decimals stores number of decimals to round to when computing positions/
# trades; may be integer or series with different value for each asset
ASSET_DECIMALS = n("/asset_decimals")

# times stores all trading/ sample times between start and end times
TIMES = n("/times")

# start time corresponds to the start time specified when running graph
START_TIME = n("/start_time")

# end time corresponds to the end time specified when running graph
END_TIME = n("/end_time")

# live time corresponds to the current trading time
LIVE_TIME = n("/live_time")

# wall time is the time the graph is run
WALL_TIME = n("/wall_time")

PRICES = n("/prices")
RETURNS = n("/returns")

# fum currently must be a float
FUM = n("/fum")

ANN_FACTOR = n("/ann_factor")

LIBRARY = n("/library")


def _write_fn(node: NodeRef, data: Any, library: str,
              aliases: Dict[NodeRef, Collection[NodeRef]],
              snap: Optional[str] = None) -> None:
    # don't write output if node name starts with underscore
    if not node.name.startswith("_") and data is not None:
        # if tuple of dataframes/ series, each element is written to a separate symbol
        if is_tuple_of(data, (pd.DataFrame, pd.Series)):
            # use square brackets below to guarantee there won't be a name conflict with
            # a different node (since square brackets aren't allowed in node names)
            refs = tuple([ArcticRef(f"{node.path}[{i}]", library, as_of=snap)
                          for i in range(len(data))])
            for i, ref in enumerate(refs):
                version = write_data(library, ref.symbol, data[i])
                if snap is not None:
                    add_to_snap(ref.symbol, snap, library, version=version,
                                create=True)
            data = refs

        version = write_data(library, node.path, data)
        if snap is not None:
            add_to_snap(node.path, snap, library, version=version, create=True)

        # also write arctic refs for aliases for convenience
        if node in aliases:
            arctic_ref = ArcticRef(node.path, library, as_of=snap)
            for alias in aliases[node]:
                version = write_data(library, alias.path, arctic_ref)
                if snap is not None:
                    add_to_snap(alias.path, snap, library, version=version)


def _read_fn(node: NodeRef, library: str, snap: Optional[str] = None):
    if snap is not None and snap not in list_snaps(library):
        return None
    if has_symbol(library, node.path, as_of=snap):
        data = read_data(library, node.path, as_of=snap)
        if is_tuple_of(data, ArcticRef):
            data = tuple([read_data(library, e.symbol, as_of=snap) for e in data])
        return data
    return None


def set_library(library: str) -> None:
    _LIBRARY_MAP[get_active_graph()] = library
    add_node(library, LIBRARY)


def run_graph(node: Union[NodeRefOrStr, Set[NodeRefOrStr]],
              resume: bool = False,
              exclude: Optional[Union[NodeRefOrStr, Set[NodeRefOrStr]]] = None,
              zap: Optional[Union[NodeRefOrStr, Set[NodeRefOrStr]]] = None,
              write: bool = False,
              executor_type: ExecutorType = ExecutorType.SYNC,
              max_workers: Optional[int] = None,
              snap: Optional[str] = None) -> GraphRun:
    graph = get_active_graph()
    aliases = inv_many_to_one(graph.aliases)

    node = ensure_set(node)

    library = _LIBRARY_MAP.get(graph)
    if library is not None:
        ensure_lib_exists(library)

    if resume and library is None:
        raise ValueError("Cannot resume graph; no library specified")

    if write and library is None:
        raise ValueError("Cannot write graph data; no library specified")

    write_fn = partial(_write_fn, library=library, snap=snap,
                       aliases=aliases) if write else None
    read_fn = partial(_read_fn, library=library, snap=snap) if resume else None

    return graph.run(node, resume, exclude=exclude, zap=zap, read_fn=read_fn,
                     write_fn=write_fn, executor_type=executor_type,
                     max_workers=max_workers)
