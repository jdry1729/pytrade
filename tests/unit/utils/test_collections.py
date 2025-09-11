from contextlib import nullcontext

import pytest

from pytrade.utils.collections import _get_first_index_greater_than


@pytest.mark.parametrize(
    ["low", "high", "target", "value_fn", "or_equal_to", "expected"],
    [
        pytest.param(
            0,
            10,
            5,
            lambda x: x,
            False,
            6,
            id="target_in_range_gt"
        ),
        pytest.param(
            0,
            10,
            9,
            lambda x: x,
            False,
            10,
            id="target_at_end_of_range_gt"
        ),
        pytest.param(
            0,
            10,
            -5,
            lambda x: x,
            False,
            0,
            id="target_less_than_range_gt"
        ),
        pytest.param(
            0,
            10,
            12,
            lambda x: x,
            False,
            ValueError(),
            id="target_greater_than_range_gt"
        ),
        pytest.param(
            0,
            10,
            5,
            lambda x: x,
            True,
            5,
            id="target_in_range_ge"
        ),
        pytest.param(
            0,
            10,
            9,
            lambda x: x,
            True,
            9,
            id="target_at_end_of_range_ge"
        ),
        pytest.param(
            0,
            10,
            -2,
            lambda x: x,
            True,
            0,
            id="target_less_than_range_ge"
        ),
        pytest.param(
            0,
            10,
            11,
            lambda x: x,
            True,
            ValueError(),
            id="target_greater_than_range_ge"
        )
    ]
)
def test__get_first_index_greater_than(low, high, target, value_fn, or_equal_to,
                                       expected):
    error = False

    ctx = nullcontext()
    if isinstance(expected, Exception):
        error = True
        ctx = pytest.raises(type(expected))

    with ctx:
        res = _get_first_index_greater_than(low, high, target, value_fn,
                                            or_equal_to=or_equal_to)
    if not error:
        assert res == expected
