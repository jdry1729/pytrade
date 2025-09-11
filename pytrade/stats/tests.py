import pandas as pd
import scipy.stats


def ttest_1samp(x, popmean=0, alternative="greater"):
    res = scipy.stats.ttest_1samp(x, popmean=popmean, alternative=alternative,
                                  nan_policy="omit")
    return pd.Series({"pvalue": res.pvalue, "tstat": res.statistic})
