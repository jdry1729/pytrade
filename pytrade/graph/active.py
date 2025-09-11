from contextlib import contextmanager
from functools import partial
from typing import Iterable, Union, Tuple, List, Optional, Set, Callable, Any, \
    Collection

from pytrade.graph.core import Graph, NodeRef, NodeRefOrStr, ExecutorType, GraphRun

_GRAPH_STACK: List[Graph] = [Graph()]


@contextmanager
def set_active_graph(g: Graph):
    _GRAPH_STACK.append(g)
    yield g
    _GRAPH_STACK.pop()


# TODO: rename to get_graph
def get_active_graph() -> Graph:
    return _GRAPH_STACK[-1]


def clear_graph() -> None:
    get_active_graph().clear()


def plot_graph(figsize: Tuple[int, int] = (12, 3),
               font_color: str = "red",
               font_weight: str = "bold",
               edge_color: Optional[Union[str, Tuple]] = (0, 0, 0, 0.5)):
    get_active_graph().plot(figsize=figsize, font_color=font_color,
                            font_weight=font_weight, edge_color=edge_color)


def add_node(fn: Any, path: Optional[str] = None, args: Tuple[Any, ...] = (),
             **kwargs) -> NodeRef:
    """
    Adds a node to the current active graph.
    """
    return get_active_graph().add_node(fn, path=path, args=args, **kwargs)


def add_placeholder(path: NodeRefOrStr) -> NodeRef:
    return get_active_graph().add_placeholder(path)


def add_flag(nodes: Collection[NodeRef], path: str) -> NodeRef:
    flag = add_node(True, path)
    for node in nodes:
        add_edge(node, flag)
    return flag


def add_alias(src: NodeRefOrStr, alias: NodeRefOrStr) -> NodeRef:
    """
    Adds an alias to the current active graph.
    """
    return get_active_graph().add_alias(src, alias)


def search(pattern: Union[str, Iterable[str]]) -> Set[NodeRef]:
    return get_active_graph().search(pattern)


# TODO: allow constant as well as partial?
def add_path(items: Iterable[Union[partial, Tuple[partial, str]]]):
    """
    Adds a path to the active graph.
    """
    ref = None
    for i, e in enumerate(items):
        p = e[0] if isinstance(e, Tuple) else e
        name = e[1] if isinstance(e, Tuple) else None
        if i > 0:
            p = partial(p, *(ref,) + p.args)
        ref = add_node(p.func, path=name, args=p.args, **p.keywords)

    return ref


def add_edge(u: NodeRefOrStr, v: NodeRefOrStr):
    """
    Adds an edge between u and v. Can be used to add an implicit dependency between
    nodes.
    """
    return get_active_graph().add_edge(u, v)


def get_nodes() -> Set[NodeRef]:
    return get_active_graph().nodes


def node_exists(node: NodeRef) -> bool:
    return node in get_active_graph().nodes


def run_graph(node: Union[NodeRefOrStr, Set[NodeRefOrStr]],
              resume: bool = False,
              exclude: Optional[Union[NodeRefOrStr, Set[NodeRefOrStr]]] = None,
              zap: Optional[Union[NodeRefOrStr, Set[NodeRefOrStr]]] = None,
              read_fn: Optional[Callable] = None,
              write_fn: Optional[Callable] = None,
              raise_if_error: bool = True,
              executor_type: ExecutorType = ExecutorType.SYNC,
              max_workers: Optional[int] = None) -> GraphRun:
    return get_active_graph().run(
        node=node, resume=resume, exclude=exclude, zap=zap, read_fn=read_fn,
        write_fn=write_fn, raise_if_error=raise_if_error, executor_type=executor_type,
        max_workers=max_workers)
