import itertools
import logging
import operator
from collections import defaultdict
from collections.abc import MutableMapping
from datetime import datetime
from typing import (Any, Callable, Dict, Iterable, List, Set, Tuple, Sequence,
                    Optional, Collection, Union)

import networkx as nx
from networkx import NetworkXUnfeasible

logger = logging.getLogger(__name__)


def reverse(d: Dict):
    return {v: k for k, v in d.items()}


def inv_many_to_one(d: Dict) -> Dict:
    res = defaultdict(list)
    for k, v in d.items():
        res[v].append(k)
    return dict(res)


def ensure_list(x) -> List:
    if isinstance(x, List):
        return x
    elif isinstance(x, tuple):
        return list(x)
    return [x]


def ensure_set(x) -> Set:
    if isinstance(x, set):
        return x
    elif isinstance(x, (tuple, list)):
        return set(x)
    return {x}


def all_none(l: Collection) -> bool:
    return all(x is None for x in l)


def is_tuple_of(t, item_type):
    if isinstance(t, Tuple):
        return all(isinstance(x, item_type) for x in t)
    return False


# TODO: mark deprecated since will exhaust generator
def is_iterable_of(l, item_type) -> bool:
    if isinstance(l, Iterable):
        # TODO: below will exhaust l if it's a generator?
        return all(isinstance(x, item_type) for x in l)
    return False


def is_collection_of(l: Collection, item_type) -> bool:
    if isinstance(l, Collection):
        return all(isinstance(x, item_type) for x in l)
    return False


def find(iterable: Iterable[Any], condition: Callable) -> Any:
    """
    Returns the first item in an iterable that satisfies a condition. Returns
    None if no item found.
    """
    try:
        return next(x for x in iterable if condition(x))
    except StopIteration:
        return None


def groupby(iterable: Iterable[Dict], key: str) -> Dict[Any, Any]:
    result = {}
    for x in iterable:
        result.setdefault(x[key], []).append(x)
    return result


# TODO: tidy! looks overly complicated
def rename_keys(d: Dict[str, Any], mapper: Dict[str, str], recursive=True):
    """
    Renames dictionary keys.
    """
    for old_key, new_key in mapper.items():
        if isinstance(old_key, str):
            value = d.pop(old_key)
            if recursive and isinstance(value, Dict):
                value = rename_keys(value, mapper)
            d[new_key] = value
    return d


def flatten(dictionary, parent_key=False, separator='.'):
    items = []
    for key, value in dictionary.items():
        new_key = str(parent_key) + separator + key if parent_key else key
        if isinstance(value, MutableMapping):
            items.extend(flatten(value, new_key, separator).items())
        elif isinstance(value, list):
            for k, v in enumerate(value):
                items.extend(flatten({str(k): v}, new_key).items())
        else:
            items.append((new_key, value))
    return dict(items)


def get(d: Dict, keys: Iterable[Any]):
    """
    Gets a nested item from a dict.
    """
    for key in keys:
        d = d[key]
    return d


def get_item(s: Sequence, i: int, default: Optional = None):
    try:
        return s[i]
    except IndexError:
        return default


def intersection(l: Collection, m: Collection) -> List:
    return [x for x in l if x in m]


def difference(l: Collection, m: Collection) -> List:
    return [x for x in l if x not in m]


def contains_any(l: Collection, values: Collection) -> bool:
    return any(v in l for v in values)


def replace(l: Collection, to_replace: Dict) -> List:
    res = []
    for x in l:
        if x in to_replace:
            res.append(to_replace[x])
        else:
            res.append(x)
    return res


def contains_duplicates(l: Collection) -> bool:
    return len(l) != len(set(l))


def get_first(obj, keys, value: Any = None):
    """
    Returns value of first key which is in in obj.
    """
    for key in keys:
        if key in obj:
            return obj[key]
    return value


def _get_first_index_greater_than(low, high, target, value_fn,
                                  or_equal_to: bool = False) -> int:
    i = 0
    op = operator.lt if or_equal_to else operator.le
    while low != high:
        i += 1
        mid = (low + high) // 2
        value = value_fn(mid)
        if op(value, target):
            low = mid + 1
        else:
            high = mid
    value = value_fn(low)
    if op(value, target):
        error_msg = "No values greater than"
        if or_equal_to:
            error_msg += " or equal to"
        raise ValueError(f"{error_msg} {target}")
    logger.debug(f"Found index in {i} steps")
    return high


def get_first_index_greater_than(low: int, high: int, target, value_fn) -> int:
    return _get_first_index_greater_than(low, high, target, value_fn)


def get_first_index_greater_than_or_equal_to(low: int, high: int, target,
                                             value_fn) -> int:
    return _get_first_index_greater_than(low, high, target, value_fn,
                                         or_equal_to=True)


def get_at_time(l: Union[Any, List[Dict]], time: datetime) -> Any:
    for i in range(len(l)):
        if l[i]["start_time"] <= time < l[i]["end_time"]:
            return l[i]["element"]
    raise ValueError("Error getting element")


def topological_sort(*args) -> List:
    g = nx.DiGraph()
    edges = {}
    nodes = set()

    # Build the graph and track the priority of each edge
    for priority, l in enumerate(args):
        nodes.update(l)
        for i in range(len(l) - 1):
            u, v = l[i], l[i + 1]
            if not g.has_edge(u, v):
                g.add_edge(u, v)
                edges[(u, v)] = priority

    while True:
        try:
            return list(nx.topological_sort(g))
        except nx.NetworkXUnfeasible:
            cycle = nx.find_cycle(g, orientation="original")
            to_remove = max(cycle, key=lambda e: edges[(e[0], e[1])])
            g.remove_edge(to_remove[0], to_remove[1])


def dedupe(l):
    res = []
    seen = set()
    for x in l:
        if x not in seen:
            seen.add(x)
            res.append(x)
    return res
