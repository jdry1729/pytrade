import pandas as pd
import pytest
from pytrade.data.datasets import _get_load_intervals
from pytrade.utils.pandas import str_to_pandas, empty_df

spans_1 = str_to_pandas("""
               id  start_time  end_time  creation_time
                0           0         9             10
                1           7        19             20
      """, index_col="id")

spans_2 = str_to_pandas("""
               id  start_time  end_time  creation_time
                0           0         9             10
                1           7        19             20
                2           5        10             30
      """, index_col="id")


@pytest.mark.parametrize(
    ["spans", "load_time", "start_time", "end_time", "reload", "settling_period",
     "buffer_period", "expected"],
    [
        pytest.param(
            spans_1,
            20,
            0,
            19,
            False,
            None,
            None,
            empty_df(columns=["start", "end"]),
            id="no_reload_settling_period_or_buffer"
        ),
        pytest.param(
            spans_1,
            20,
            0,
            19,
            True,
            None,
            0,
            str_to_pandas("""
                start  end
                    0   19
            """),
            id="reload_with_zero_buffer"
        ),
        pytest.param(
            spans_1,
            20,
            0,
            19,
            True,
            None,
            None,
            str_to_pandas("""
                start  end
                    0   19
            """),
            id="reload_without_buffer"
        ),
        pytest.param(
            spans_1,
            20,
            0,
            19,
            True,
            None,
            1,
            str_to_pandas("""
                start  end
                    0    7
            """),
            id="reload_with_buffer"
        ),
        pytest.param(
            spans_1,
            20,
            0,
            19,
            False,
            3,
            None,
            str_to_pandas("""
                start  end
                   17   19
            """),
            id="settling_period_without_buffer"
        ),
        pytest.param(
            spans_1,
            20,
            0,
            19,
            False,
            3,
            1,
            empty_df(columns=["start", "end"]),
            id="settling_period_with_buffer"
        ),
        pytest.param(
            spans_1,
            22,
            0,
            21,
            False,
            3,
            None,
            str_to_pandas("""
                start  end
                   17   21
            """),
            id="new_data_with_settling_period"
        ),
        pytest.param(
            spans_1,
            22,
            0,
            21,
            False,
            3,
            3,
            str_to_pandas("""
                start  end
                   19   21
            """),
            id="new_data_with_settling_period_and_buffer"
        ),
        pytest.param(
            spans_2,
            31,
            0,
            19,
            True,
            3,
            2,
            str_to_pandas("""
                start  end
                    0    5
                   10   19
            """),
            id="reload_with_settling_period_and_buffer"
        ),
    ]
)
def test__get_load_intervals(spans, load_time, start_time, end_time, reload,
                             settling_period, buffer_period, expected):
    actual = _get_load_intervals(spans, load_time, start_time, end_time,
                                 reload=reload, settling_period=settling_period,
                                 buffer_period=buffer_period)
    return pd.testing.assert_frame_equal(actual, expected)
