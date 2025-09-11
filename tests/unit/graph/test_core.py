from typing import Dict, Any, Iterable, Union

import pytest

from pytrade.graph import n, GraphRun
from pytrade.graph.core import Graph


def graph_1():
    g = Graph()
    g.add_node(2, "a")
    g.add_node(lambda x: x * 3, "b", x=n("a"))
    g.add_node(2, "c")
    g.add_node(5, "d")
    g.add_node(lambda x, y: x * y, "e", x=n("c"), y=n("d"))
    g.add_node(lambda x, y: x + y, "f", x=n("c"), y=n("d"))
    g.add_node(lambda x, y: x * y, "g", args=("e",), y=n("c"))
    return g


@pytest.mark.parametrize(
    ["graph", "node", "expected"],
    [
        pytest.param(
            graph_1(),
            "a",
            GraphRun({
                n("a"): 2
            })
        ),
        pytest.param(
            graph_1(),
            ["f", "e", "b", "g"],
            GraphRun({
                n("f"): 7,
                n("e"): 10,
                n("b"): 6,
                n("g"): "ee",
            })
        ),

    ]
)
def test_graph_run(graph: Graph, node: Union[str, Iterable[str]],
                   expected: Dict[str, Any]):
    actual = graph.run(node)
    assert actual == expected
