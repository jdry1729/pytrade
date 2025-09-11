import pandas as pd
import pytest
from pytrade.risk.models.har import fit_har_cholesky_risk_model
from pytrade.utils.pandas import str_to_pandas


@pytest.mark.parametrize(
    ["realized_cov", "lags", "expected"],
    [
        pytest.param(
            str_to_pandas(
                """  
                                      
                time           asset  PF_XBTUSD  PF_ETHUSD                   
                2024-02-10 PF_XBTUSD   0.000834   0.000530
                2024-02-10 PF_ETHUSD   0.000530   0.000580
                2024-02-11 PF_XBTUSD   0.000229   0.000153
                2024-02-11 PF_ETHUSD   0.000153   0.000194
                2024-02-12 PF_XBTUSD   0.000268   0.000173
                2024-02-12 PF_ETHUSD   0.000173   0.000229
                2024-02-13 PF_XBTUSD   0.000698   0.000469
                2024-02-13 PF_ETHUSD   0.000469   0.000685
                2024-02-14 PF_XBTUSD   0.000708   0.000608
                2024-02-14 PF_ETHUSD   0.000608   0.000855
                2024-02-15 PF_XBTUSD   0.000672   0.000426
                2024-02-15 PF_ETHUSD   0.000426   0.000629
                2024-02-16 PF_XBTUSD   0.000687   0.000661
                2024-02-16 PF_ETHUSD   0.000661   0.001020
                2024-02-17 PF_XBTUSD   0.000414   0.000428
                2024-02-17 PF_ETHUSD   0.000428   0.000667
                2024-02-18 PF_XBTUSD   0.000312   0.000274
                2024-02-18 PF_ETHUSD   0.000274   0.000328
                2024-02-19 PF_XBTUSD   0.000251   0.000298
                2024-02-19 PF_ETHUSD   0.000298   0.000541
                2024-02-20 PF_XBTUSD   0.000274   0.000349
                2024-02-20 PF_ETHUSD   0.000349   0.000708
                2024-02-21 PF_XBTUSD   0.000704   0.000750
                2024-02-21 PF_ETHUSD   0.000750   0.001166
                2024-02-22 PF_XBTUSD   0.000444   0.000502
                2024-02-22 PF_ETHUSD   0.000502   0.000903
                2024-02-23 PF_XBTUSD   0.000507   0.000622
                2024-02-23 PF_ETHUSD   0.000622   0.001086
                2024-02-24 PF_XBTUSD   0.000306   0.000363
                2024-02-24 PF_ETHUSD   0.000363   0.000570
                2024-02-25 PF_XBTUSD   0.000107   0.000126
                2024-02-25 PF_ETHUSD   0.000126   0.000234
                """,
                index_col=("time", "asset"),
                parse_dates=["time"]
            ),
            (1, 5),
            str_to_pandas(
                """
                                     
                time           asset  PF_XBTUSD  PF_ETHUSD                    
                2024-02-10 PF_XBTUSD        NaN        NaN
                2024-02-10 PF_ETHUSD        NaN        NaN
                2024-02-11 PF_XBTUSD        NaN        NaN
                2024-02-11 PF_ETHUSD        NaN        NaN
                2024-02-12 PF_XBTUSD        NaN        NaN
                2024-02-12 PF_ETHUSD        NaN        NaN
                2024-02-13 PF_XBTUSD        NaN        NaN
                2024-02-13 PF_ETHUSD        NaN        NaN
                2024-02-14 PF_XBTUSD        NaN        NaN
                2024-02-14 PF_ETHUSD        NaN        NaN
                2024-02-15 PF_XBTUSD   0.000413   0.000452
                2024-02-15 PF_ETHUSD   0.000452   0.000775
                2024-02-16 PF_XBTUSD   0.000433   0.000524
                2024-02-16 PF_ETHUSD   0.000524   0.000905
                2024-02-17 PF_XBTUSD   0.000309   0.000385
                2024-02-17 PF_ETHUSD   0.000385   0.000670
                2024-02-18 PF_XBTUSD   0.000264   0.000307
                2024-02-18 PF_ETHUSD   0.000307   0.000489
                2024-02-19 PF_XBTUSD   0.000284   0.000340
                2024-02-19 PF_ETHUSD   0.000340   0.000606
                2024-02-20 PF_XBTUSD   0.000341   0.000395
                2024-02-20 PF_ETHUSD   0.000395   0.000698
                2024-02-21 PF_XBTUSD   0.000516   0.000521
                2024-02-21 PF_ETHUSD   0.000521   0.000807
                2024-02-22 PF_XBTUSD   0.000442   0.000446
                2024-02-22 PF_ETHUSD   0.000446   0.000723
                2024-02-23 PF_XBTUSD   0.000456   0.000470
                2024-02-23 PF_ETHUSD   0.000470   0.000740
                2024-02-24 PF_XBTUSD   0.000369   0.000352
                2024-02-24 PF_ETHUSD   0.000352   0.000505
                2024-02-25 PF_XBTUSD   0.000283   0.000260
                2024-02-25 PF_ETHUSD   0.000260   0.000395
                """,
                index_col=("time", "asset"),
                parse_dates=["time"]
            )
        )
    ],
)
def test_fit_har_cholesky_risk_model(realized_cov, lags, expected):
    model = fit_har_cholesky_risk_model(realized_cov, lags)
    actual = model.predict(realized_cov)
    pd.testing.assert_frame_equal(actual, expected, atol=1e-6, rtol=0,
                                  check_names=False)
