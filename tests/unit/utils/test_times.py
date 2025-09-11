from datetime import datetime, timedelta

import pytest
from pytrade.utils.time import get_equally_spaced_times


@pytest.mark.parametrize(
    ["start_time", "end_time", "period", "expected"],
    [
        pytest.param(
            datetime(2024, 1, 4),
            datetime(2024, 1, 7),
            timedelta(days=2),
            [datetime(2024, 1, 4), datetime(2024, 1, 6)]
        ),
        pytest.param(
            datetime(2024, 1, 4, 13),
            datetime(2024, 1, 7),
            timedelta(days=8),
            [datetime(2024, 1, 4, 13)]
        )
    ]
)
def test_get_equally_spaced_times(start_time, end_time, period, expected):
    actual = get_equally_spaced_times(start_time, end_time, period)
    assert actual == expected
