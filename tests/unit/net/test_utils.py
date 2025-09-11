from contextlib import nullcontext

import pytest
from pytrade.net.utils import time_period_to_page_range

PAGE_SPAN_DATA_1 = [(i * 10 + 1, (i + 1) * 10) for i in range(0, 100)]
# PAGE_SPAN_DATA_2 = [(991, 1000), (981, 990), (971, 980), ..., (11, 20), (1, 10)]
PAGE_SPAN_DATA_2 = PAGE_SPAN_DATA_1[::-1]


@pytest.mark.parametrize(
    ["num_pages", "span_fn", "start_time", "end_time", "ascending", "expected"],
    [
        # | 750 |
        # | ... | page 26
        # | 741 |
        #
        #   ...
        #
        # | 250 |
        # | ... | page 76
        # | 241 |
        pytest.param(
            100,
            lambda i: PAGE_SPAN_DATA_2[i - 1],
            250,
            750,
            False,
            (26, 76),
            id="time_period_subset_of_data_period_desc"
        ),
        # | 1000 |
        # | ...  | page 1
        # | 991  |
        #
        #   ...
        pytest.param(
            100,
            lambda i: PAGE_SPAN_DATA_2[i - 1],
            995,
            1200,
            False,
            (1, 1),
            id="time_period_overlaps_end_of_data_period_desc"
        ),
        #   ...
        #
        # | 10  |
        # | ... | page 100
        # | 1   |
        pytest.param(
            100,
            lambda i: PAGE_SPAN_DATA_2[i - 1],
            -100,
            5,
            False,
            (100, 100),
            id="time_period_overlaps_start_of_data_period_desc"
        ),
        pytest.param(
            100,
            lambda i: PAGE_SPAN_DATA_2[i - 1],
            1100,
            1200,
            False,
            ValueError(),
            id="time_period_after_data_period_desc"
        ),
        pytest.param(
            100,
            lambda i: PAGE_SPAN_DATA_2[i - 1],
            -100,
            -50,
            False,
            ValueError(),
            id="time_period_before_data_period_desc"
        )
    ]
)
def test_time_period_to_page_range(num_pages, span_fn, start_time, end_time,
                                   ascending, expected):
    error = False

    ctx = nullcontext()
    if isinstance(expected, Exception):
        error = True
        ctx = pytest.raises(type(expected))

    with ctx:
        res = time_period_to_page_range(num_pages, span_fn, start_time, end_time,
                                        ascending=ascending)
    if not error:
        assert res == expected
