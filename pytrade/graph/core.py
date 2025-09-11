import fnmatch
import itertools
import logging
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any
from typing import Optional, Callable
from typing import Set
from typing import Union, Iterable, Tuple, List

import networkx as nx
from matplotlib import pyplot as plt
from pytrade.utils.collections import ensure_list, ensure_set
from pytrade.utils.functions import partial, map_args_and_kwargs
from pytrade.utils.random import generate_uid

logger = logging.getLogger(__name__)

# allow "=" and "." below since useful to set namespace to "alpha=0.5" when
# doing experiments
NODE_PATH_REGEX = "^[a-zA-Z0-9_\.=]+(\/[a-zA-Z0-9_\.=]+)*$"

NodeRefOrStr = Union["NodeRef", str]

# TODO: change to just allow \w word characters?
NAMESPACE_REGEX = "^[a-zA-Z0-9_\.=-]+$"

_NAMESPACE_STACK: List[str] = []


class UnknownRefException(Exception):
    pass


class ExecutorType(Enum):
    SYNC = 0
    PROC = 2


@contextmanager
def set_ns(namespaces: Union[str, Iterable[str]]):
    global _NAMESPACE_STACK
    namespaces = ensure_list(namespaces)
    namespaces = [y for x in namespaces for y in x.split("/")]
    for ns in namespaces:
        if not re.match(NAMESPACE_REGEX, ns):
            raise ValueError(f"Error creating namespace: {ns};"
                             f" namespace can only contain alphanumeric"
                             f" characters, \"=\", \"_\", and \".\" ")

    prev_stack = _NAMESPACE_STACK.copy()
    try:
        _NAMESPACE_STACK.extend(namespaces)
        yield
    finally:
        _NAMESPACE_STACK = prev_stack


def get_ns() -> Tuple[str]:
    return tuple(_NAMESPACE_STACK)


def fully_qualified_name(name: str, namespace: Tuple[str, ...]):
    return "/".join(list(namespace) + [name])


@dataclass(frozen=True)
class NodeRef:
    name: str
    # namespace must be tuple for NodeRef to be hashable
    namespace: Tuple[str, ...] = ()

    def __post_init__(self):
        if not isinstance(self.namespace, tuple):
            raise ValueError(
                "Error creating NodeRef; namespace must be a tuple")

    @property
    def path(self):
        return fully_qualified_name(self.name, self.namespace)


def path_to_ref(path: str, namespace: Tuple[str, ...] = ()):
    # TODO: validate path/ namespace
    while path.startswith("../"):
        path = path.removeprefix("../")
        namespace = namespace[:-1]

    split = tuple(path.split("/"))
    if path.startswith("/"):
        return NodeRef(name=split[-1], namespace=split[1:-1])
    return NodeRef(name=split[-1], namespace=namespace + split[:-1])


def n(path: str):
    # the node ref you get depends on current namespace
    namespace = tuple(_NAMESPACE_STACK)
    return path_to_ref(path, namespace)


def _deref_arg(arg: Any, res: Dict[NodeRef, Any], aliases: Dict[NodeRef, NodeRef]):
    _deref_fn = partial(_deref_arg, res=res, aliases=aliases)
    if isinstance(arg, NodeRef):
        return res[aliases.get(arg, arg)]
    elif isinstance(arg, Dict):
        return {_deref_fn(k): _deref_fn(v) for k, v in arg.items()}
    elif isinstance(arg, Tuple):
        return tuple(_deref_fn(x) for x in arg)
    elif isinstance(arg, List):
        return list(_deref_fn(x) for x in arg)
    return arg


def _extract_refs(*args, **kwargs) -> Set[NodeRef]:
    def _extract(node_refs: Set, arg: Any) -> None:
        if isinstance(arg, NodeRef):
            node_refs.add(arg)
        elif isinstance(arg, Dict):
            for x in itertools.chain(arg.keys(), arg.values()):
                _extract(node_refs, x)
        # must use (Tuple, List) instead of Iterable below since strings are
        # instances of Iterable
        elif isinstance(arg, (Tuple, List)):
            for x in arg:
                _extract(node_refs, x)

    node_refs = set()
    _extract(node_refs, args)
    _extract(node_refs, kwargs)
    return node_refs


def _str_to_node_ref(x: Any):
    if isinstance(x, str):
        return n(x)
    return x


def _get_nodes_from_patterns(
        graph: "Graph", patterns: Set[NodeRefOrStr]) -> Set[NodeRef]:
    nodes = set()
    nodes.update([x for x in patterns if isinstance(x, NodeRef)])
    nodes.update(graph.search([x for x in patterns if isinstance(x, str)]))
    return nodes


def _eval_node(graph: "Graph", nodes: Set[NodeRef], node: NodeRef,
               out: Dict[NodeRef, Any], dirty_nodes: Set[NodeRef],
               read_fn: Optional[Callable] = None,
               write_fn: Optional[Callable] = None, raise_if_error: bool = True):
    if node in out:
        return

    attrs = graph._nx.nodes[node]
    if read_fn is not None and node not in dirty_nodes:
        logger.debug(f"Trying to read: {node.path}")
        t0 = time.time()
        data = read_fn(node)
        t1 = time.time()
        if data is not None:
            logger.info(f"Read: {node.path} ({t1 - t0:.2f}s)")
            out[node] = data
            return
        else:
            logger.debug(f"No data found for: {node.path}")

    for predecessor in graph.predecessors(node):
        _eval_node(graph, nodes, predecessor, out, dirty_nodes, read_fn, write_fn,
                   raise_if_error)
    # node have been set to None if predecessor errored and raise_if_error=False
    if node in out:
        return

    logger.info(f"Computing: {node.path}")
    node_args, node_kwargs = map_args_and_kwargs(
        partial(_deref_arg, res=out, aliases=graph.aliases),
        attrs["args"], attrs["kwargs"])

    try:
        t0 = time.time()
        data = attrs["fn"](*node_args, **node_kwargs)
        t1 = time.time()
        logger.info(f"Computed: {node.path} ({t1 - t0:.2f}s)")

        out[node] = data
        if write_fn is not None:
            write_fn(node, data)
    except Exception:
        if raise_if_error:
            logger.critical(f"Error computing node {node}", exc_info=True)
            raise
        logger.warning(f"Error computing node {node}; proceeding to compute"
                       f" non-descendant nodes since raise_if_error=False",
                       exc_info=True)
        out[node] = None
        for descendant in graph.descendants(node):
            out[descendant] = None

    for predecessor in graph.predecessors(node):
        # clear predecessor data if they have no more successors
        if predecessor not in nodes and all(
                x in out for x in graph.successors(predecessor)):
            logger.info(f"Clearing: {predecessor.path}")
            out[predecessor] = None


class GraphRun:
    def __init__(self, res: Dict):
        self._res = res

    def __getitem__(self, item):
        if isinstance(item, str):
            item = n(item)
        return self._res[item]

    def __eq__(self, other):
        return self._res == other._res

    # TODO: accept pattern rather than regex like in graph.search?
    def list_nodes(self, regex: Optional[str] = None) -> Set[str]:
        nodes = [x.path for x in self._res.keys()]
        if regex is not None:
            nodes = [x for x in nodes if re.search(regex, x)]
        return set(nodes)


class Graph:

    def __init__(self):
        self._nx = nx.DiGraph()
        self._aliases: Dict[NodeRef, NodeRef] = {}
        self._placeholders: Set[NodeRef] = set()

    def add_node(self, fn: Any, path: Optional[NodeRefOrStr] = None,
                 args: Tuple[Any, ...] = (), **kwargs) -> NodeRef:
        if path is None:
            path = generate_uid(6)

        if not callable(fn):
            if args or kwargs:
                raise ValueError("Error adding node; args and kwargs must be empty if"
                                 " fn isn't callable")
            const = fn
            # must return const below rather than fn!
            fn = lambda: const  # noqa: E731

        node_ref = path
        if isinstance(node_ref, str):
            node_ref = n(node_ref)

        node_attrs = dict(fn=fn, args=args, kwargs=kwargs)
        if node_ref in self._nx.nodes:
            if node_ref not in self._placeholders:
                raise ValueError(f"Node already exists with ref: {node_ref}")
            self._placeholders.remove(node_ref)

        if node_ref not in self._nx.nodes:
            self._nx.add_node(node_ref)

        self._nx.nodes[node_ref].update(node_attrs)
        for arg_ref in _extract_refs(*args, **kwargs):
            arg_ref_ = self._aliases.get(arg_ref, arg_ref)
            if arg_ref_ not in self._nx.nodes:
                raise UnknownRefException(
                    f"Error adding node: {node_ref.path}; {arg_ref.path} doesn't"
                    f" exist")
            self._nx.add_edge(arg_ref_, node_ref)
        return node_ref

    def add_placeholder(self, path: NodeRefOrStr) -> NodeRef:
        """
        Adds a placeholder node.
        """
        if isinstance(path, str):
            path = n(path)
        if path in self._placeholders or path in self.nodes:
            return
        self._nx.add_node(path)
        self._placeholders.add(path)

    def add_alias(self, src: NodeRefOrStr, alias: NodeRefOrStr) -> NodeRef:
        """
        Adds an alias for a node.

        Parameters
        ----------
        src
            Node to alias. May be an alias itself.
        alias
            Alias to create.

        Returns
        -------
        alias
        """
        if isinstance(src, str):
            src = n(src)
        if isinstance(alias, str):
            alias = n(alias)

        if src == alias:
            return alias

        if alias in self.nodes:
            if alias not in self._placeholders:
                raise ValueError("Error adding alias; alias already exists")

        if src not in self.nodes:
            raise UnknownRefException(
                f"Error adding alias: {alias.path}; {src.path} doesn't"
                f" exist")

        if src in self._aliases:
            src = self._aliases[src]

        if alias in self._placeholders:
            for edge in self._nx.out_edges(alias):
                self._nx.add_edge(src, edge[1])
            # below removes node and all edges
            self._nx.remove_node(alias)
            self._placeholders.remove(alias)

        self._aliases[alias] = src
        return alias

    def add_edge(self, u: NodeRefOrStr, v: NodeRefOrStr) -> None:
        """
        Can be used to add implicit dependency between nodes.
        """
        # TODO: add nodes if unknown
        if isinstance(u, str):
            u = n(u)
        if isinstance(v, str):
            v = n(v)
        u = self._aliases.get(u, u)
        v = self._aliases.get(v, v)
        self._nx.add_edge(u, v)

    @property
    def aliases(self) -> Dict[NodeRef, NodeRef]:
        return self._aliases.copy()

    @property
    def nodes(self) -> Set[NodeRef]:
        nodes = set(self._nx.nodes)
        aliases = set(self._aliases.keys())
        return nodes.union(aliases)

    def search(self, pattern: Union[str, Iterable[str]]) -> Set[NodeRef]:
        nodes = set()
        paths_str = set(x.path for x in self.nodes)
        for p in ensure_set(pattern):
            nodes.update([n(f"/{path}") for path in fnmatch.filter(paths_str, p)])
        return nodes

    def ancestors(self, node: NodeRef) -> Set[NodeRef]:
        """
        Gets node ancestors, i.e., dependencies.
        """
        node = self._aliases.get(node, node)
        return nx.ancestors(self._nx, node)

    def predecessors(self, node: NodeRef) -> Set[NodeRef]:
        """
        Gets node predecessors, i.e., direct dependencies of the node.
        """
        node = self._aliases.get(node, node)
        return set(self._nx.predecessors(node))

    def descendants(self, node: NodeRef) -> Set[NodeRef]:
        """
        Gets node descendants, i.e., nodes which depend on the node.
        """
        node = self._aliases.get(node, node)
        return nx.descendants(self._nx, node)

    def successors(self, node: NodeRef) -> Set[NodeRef]:
        """
        Gets node successors i.e., nodes which directly depend on the node.
        """
        node = self._aliases.get(node, node)
        return set(self._nx.successors(node))

    def subgraph(self, node: Union[NodeRef, Set[NodeRef]]):
        g = Graph()
        nodes = set(self._aliases.get(x, x) for x in ensure_set(node))
        ancestors = set()
        for node in nodes:
            ancestors.update(self.ancestors(node))

        aliases = {k: v for k, v in self._aliases.items() if v in ancestors}
        g._nx = self._nx.subgraph(ancestors)
        g._aliases = aliases
        return g

    def plot(
            self,
            figsize: Tuple[int, int] = (12, 3),
            font_color: str = "red",
            font_weight: str = "bold",
            edge_color: Optional[Union[str, Tuple]] = (0, 0, 0, 0.5)
    ):
        """
        Plots the graph.

        Notes
        -----
        See: https://networkx.org/documentation/stable/auto_examples/graph
        /plot_dag_layout.html
        """
        plt.figure(figsize=figsize)
        g = self._nx.copy()
        for alias in self._aliases:
            g.add_node(alias)
            g.add_edge(self._aliases[alias], alias)
        for layer, nodes in enumerate(nx.topological_generations(g)):
            for node in nodes:
                g.nodes[node]["layer"] = layer
        colors = ["#ffb399" if x in self._aliases else "#99ccff" for x in g.nodes]
        pos = nx.multipartite_layout(g, subset_key="layer")
        labels = {x: x.path for x in g.nodes}
        nx.draw_networkx(g, pos=pos, arrows=True, labels=labels, node_color=colors,
                         font_color=font_color, font_weight=font_weight,
                         edge_color=edge_color)

    def clear(self) -> None:
        self._nx.clear()
        self._aliases.clear()

    def _run_sync(self, nodes: Set[NodeRef], dirty_nodes: Set[NodeRef],
                  read_fn: Optional[Callable] = None,
                  write_fn: Optional[Callable] = None,
                  raise_if_error: bool = True) -> Dict[NodeRef, Any]:
        out = {}
        nodes_ = set(self._aliases.get(x, x) for x in nodes)
        for node in nodes_:
            _eval_node(self, nodes_, node, out, dirty_nodes, read_fn, write_fn,
                       raise_if_error=raise_if_error)
        return {k: out[k] for k in nodes if k in out}

    def run(self, node: Union[NodeRefOrStr, Set[NodeRefOrStr]],
            resume: bool = False,
            exclude: Optional[Union[NodeRefOrStr, Set[NodeRefOrStr]]] = None,
            zap: Optional[Union[NodeRefOrStr, Set[NodeRefOrStr]]] = None,
            read_fn: Optional[Callable] = None,
            write_fn: Optional[Callable] = None,
            raise_if_error: bool = True,
            allow_unfilled_placeholders: bool = True,
            executor_type: ExecutorType = ExecutorType.SYNC,
            max_workers: Optional[int] = None) -> GraphRun:
        logger.info("Running graph")

        nodes = _get_nodes_from_patterns(self, ensure_set(node))
        excludes = set()
        if exclude is not None:
            excludes = _get_nodes_from_patterns(self, ensure_set(exclude))

        if self._placeholders:
            if not allow_unfilled_placeholders:
                raise ValueError(f"Error running graph; unfilled placeholders:"
                                 f" {', '.join(x.path for x in self._placeholders)}")
            placeholder_excludes = set(self._placeholders)
            for placeholder in self._placeholders:
                placeholder_excludes.update(self.descendants(placeholder))
            logger.warning(f"Excluding nodes due to unfilled placeholders:"
                           f" {', '.join(x.path for x in placeholder_excludes)}")
            excludes.update(placeholder_excludes)

        nodes = nodes.difference(excludes)

        zaps = (set() if zap is None else
                _get_nodes_from_patterns(self, ensure_set(zap)))
        if zaps:
            logger.info(f"Zapping nodes: {', '.join([x.path for x in zaps])}")

        nodes_ = set(self._aliases.get(x, x) for x in nodes)
        zaps_ = set(self._aliases.get(x, x) for x in zaps)

        dirty_nodes_ = self._nx.nodes
        if resume:
            dirty_nodes_ = set(zaps_)
            for node in zaps_:
                dirty_nodes_.update(self.descendants(node))

        kwargs = {"nodes": nodes_, "dirty_nodes": dirty_nodes_,
                  "read_fn": read_fn, "write_fn": write_fn,
                  "raise_if_error": raise_if_error}
        if executor_type in [ExecutorType.PROC]:
            kwargs["max_workers"] = max_workers

        if executor_type == ExecutorType.SYNC:
            res = self._run_sync(**kwargs)
        else:
            raise ValueError(f"Executor type must be: {ExecutorType.SYNC}")

        return GraphRun({node: res[self._aliases.get(node, node)] for node in nodes})
