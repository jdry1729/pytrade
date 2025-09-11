import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from matplotlib.cm import get_cmap
from matplotlib.ticker import FuncFormatter


def fcf_plot(data: pd.DataFrame, ax: plt.Axes, padding) -> plt.Axes:
    years = data.index
    n_years = len(years)

    colors = get_cmap("tab10").colors

    bar_width = 0.175
    x = np.arange(n_years) * 0.75

    for i, year in enumerate(years):
        offset = x[i]

        nopat = data.loc[year, "nopat"]
        if not np.isnan(nopat):
            ax.bar(
                offset,
                nopat,
                width=bar_width,
                color=colors[0],
                label="NOPAT" if i == 0 else "",
            )
            nopat_label_y = nopat / 2
            ax.text(
                offset,
                nopat_label_y,
                f"{int(nopat / 1e6)}",
                ha='center',
                va='center',
                color='black',
                fontsize=8,
            )

            net_investment = data.loc[year, "net_investment"]
            if not np.isnan(net_investment):
                op_exp_base = nopat
                ax.bar(
                    offset + bar_width,
                    -net_investment,
                    width=bar_width,
                    bottom=op_exp_base,
                    color=colors[1],
                    label="Net Investment" if i == 0 else "",
                )
                net_investment_label_y = op_exp_base - net_investment / 2
                ax.text(
                    offset + bar_width,
                    net_investment_label_y,
                    f"{int(net_investment / 1e6)}",
                    ha='center',
                    va='center',
                    color='black',
                    fontsize=8,
                )

                fcf = data.loc[year, "fcf"]
                ax.bar(
                    offset + 2 * bar_width,
                    fcf,
                    width=bar_width,
                    color=colors[2],
                    label="FCF" if i == 0 else "",
                )
                fcf_label_y = fcf / 2
                ax.text(
                    offset + 2 * bar_width,
                    fcf_label_y,
                    f"{int(fcf / 1e6)}",
                    ha='center',
                    va='center',
                    color='black',
                    fontsize=8,
                )

    ax.set_xticks(x + 1 * bar_width)
    ax.set_xticklabels([f"FY{str(x)[2:-2]}" for x in years])
    ax.set_axisbelow(True)
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda y, _: f"{'-' if y < 0 else ''}${abs(int(y / 1e6))}M")
    )
    padding = data[["nopat", "fcf"]].abs().max().max() * padding
    ax.set_ylim(min(data[["nopat", "fcf"]].min().min(), 0) - padding,
                max(data[["nopat", "fcf"]].max().max(), 0) + padding)
    ax.grid(zorder=1)
    return ax


def roic_plot(data: pd.DataFrame, ax: plt.Axes) -> plt.Axes:
    years = data.index
    n_years = len(years)
    x = np.arange(n_years)

    # Set bar width
    bar_width = 0.35

    # Plot NOPAT Margin bars
    bars_nopat = ax.bar(
        x - bar_width / 2,
        data['nopat_margin'] * 100,
        width=bar_width,
        label='NOPAT Margin'
    )

    # Plot ROIC bars
    bars_roic = ax.bar(
        x + bar_width / 2,
        data['roic'] * 100,
        width=bar_width,
        label='ROIC'
    )

    # Add labels inside bars
    for bars in [bars_nopat, bars_roic]:
        for bar in bars:
            height = bar.get_height()
            xpos = bar.get_x() + bar.get_width() / 2
            ypos = bar.get_y() + height / 2
            ax.text(
                xpos,
                ypos,
                f"{height:.1f}",
                ha='center',
                va='center',
                color='black',
                fontsize=8,
            )

    # Axis formatting
    ax.set_xticks(x)
    ax.set_xticklabels([f"FY{str(x)[2:-2]}" for x in years])
    ax.set_axisbelow(True)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax.grid()
