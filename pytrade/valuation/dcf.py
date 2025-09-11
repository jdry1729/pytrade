from dataclasses import dataclass
from typing import Union, Sequence, Optional

import numpy as np


@dataclass
class EnterpriseDCF:
    """
    Attributes
    ----------
    invested_capital
        Invested capital. Equal to net PP&E + working capital, or, equivalently,
        operating assets - operating liabilities.
    roic
        Return on invested capital. Equal to NOPAT/ invested capital.
    """
    revenues: np.ndarray
    operating_profit: np.ndarray
    taxes: np.ndarray
    nopat: np.ndarray
    fcff: np.ndarray
    reinvestment: np.ndarray
    reinvestment_rate: np.ndarray
    enterprise_value: float
    terminal_value: float
    discounted_terminal_value: float
    discounted_fcff: np.ndarray
    discount_factor: np.ndarray
    value_of_operations: float
    equity_value: float
    equity_value_per_share: Optional[float] = None
    # invested capital for period i represents invested capital after making
    # reinvestments necessary to achieve growth in period i + 1
    invested_capital: Optional[np.ndarray] = None
    # roic is calculated as nopat for period i divided by invested capital for period
    # i - 1
    roic: Optional[np.ndarray] = None


def compute_taxes(pre_tax_income: np.ndarray, tax_rate: Union[float, np.ndarray],
                  allow_nol_carryforwards: bool = True,
                  init_nol: float = 0) -> np.ndarray:
    cum_nol = init_nol
    N = len(pre_tax_income)
    taxes = np.zeros(N)

    for i in range(N):
        pre_tax_income_ = pre_tax_income[i]
        if pre_tax_income_ >= 0:
            offset = 0
            if allow_nol_carryforwards:
                offset = min(pre_tax_income_, cum_nol)
            taxes[i] = (pre_tax_income_ - offset) * tax_rate[i]
            cum_nol -= offset
        else:
            taxes[i] = 0
            cum_nol += abs(pre_tax_income_)

    return taxes


def compute_terminal_value(nopat: float, growth_rate: float,
                           discount_rate: float,
                           ronic: Optional[float] = None) -> float:
    if ronic is None:
        ronic = discount_rate
    return nopat * (1 - growth_rate / ronic) / (discount_rate - growth_rate)


def reinvestment_from_sales_to_capital(
        revenues: Sequence[float], sales_to_capital: Union[float, Sequence[float]]
) -> np.ndarray:
    N = len(revenues)
    if isinstance(sales_to_capital, float):
        sales_to_capital = np.full(N, sales_to_capital)

    revenues = np.array(revenues)
    sales_to_capital = np.array(sales_to_capital)

    return np.diff(revenues, append=np.nan) / sales_to_capital


def enterprise_dcf(
        revenues: Sequence[float],
        operating_margin: Sequence[float],
        reinvestment: Sequence[float],
        *,
        tax_rate: Sequence[float],
        discount_rate: Sequence[float],
        shares_outstanding: Optional[int] = None,
        init_book_debt: float = 0,
        init_book_equity: Optional[float] = None,
        init_excess_cash: float = 0,
        init_nol: float = 0,
        terminal_growth_rate: float = 0,
        terminal_ronic: Optional[float] = None,
        init_options_value: float = 0,
        failure_pct: float = 0,
        liquidation_value: float = 0,
        use_mid_year_convention: bool = False,
) -> EnterpriseDCF:
    """
    Values a company using enterprise discounted cash flow. Assumes company has
    no non-operating assets other than excess cash.

    Parameters
    ----------
    revenues
        Company revenues. Must be an array of length N + 1, where N is length
        of projection period.
    operating_margin
        Operating margin. Must be an array of length N + 1.
    reinvestment
        Reinvestment. Must be an array of length N + 1. This equals capex -
        depreciation + change in working capital. If a company repairs its existing
        assets to cover their depreciation, doesn't change its working capital, and
        doesn't make any extra capital expenditures, its reinvestment will be 0.
    tax_rate
        Tax rate. Must be an array of length N + 1.
    discount_rate
        Discount rate. Must be an array of length N + 1.
    shares_outstanding
        Current shares outstanding of company.
    init_book_debt
        Total initial long/ short term interest-bearing debt of company.
    init_book_equity
        Total initial shareholder's equity. Required to compute invested capital
        and ROIC.
    init_nol
        Initial net operating loss used to shield tax.
    terminal_growth_rate
        Terminal growth rate.
    terminal_ronic
        Terminal RONIC. Assumed equal to terminal discount rate if not specified.
    init_excess_cash
        Initial excess cash of company.
    init_options_value
        Initial value of outstanding options.
    failure_pct
        Probability of failure.
    liquidation_value
        Liquidation value.
    use_mid_year_convention
        Whether to discount using mid-year convention.

    Returns
    -------
    Enterprise DCF.

    Notes
    -----
    Terminal value is computed based on the FCF and discount rate of the last period.
    The present value of this terminal value, however, is computed using the discount
    rate corresponding to the second last period. This is necessary because the
    terminal value is computed as of the second last period.

    If you want to do a 10 year DCF and compute a terminal value for future years,
    you should pass arrays of length 11 for revenue, operating_margin, reinvestment,
    tax_rate and discount_rate.

    Approach prior to changing to McKinsey terminal value calculation can be
    followed by setting terminal_ronic = terminal sales/ capital ratio x terminal
    nopat margin.

    # TODO: phrase below better
    If terminal_ronic equals the terminal discount rate, it's assumed that the
    company's new investments after the explicit forecast period earn a return equal to
    the cost of capital, and thus add no value. Thus you should set the length of
    the explicit forecast period to the number of years you believe the company
    will be able to carry on making positive NPV investments.
    """
    revenues = np.array(revenues)
    operating_margin = np.array(operating_margin)
    reinvestment = np.array(reinvestment)
    tax_rate = np.array(tax_rate)
    discount_rate = np.array(discount_rate)

    operating_profit = revenues * operating_margin
    # TODO: set allow_nol_carryforwards to False so differentiable?
    taxes = compute_taxes(operating_profit, tax_rate,
                          allow_nol_carryforwards=True,
                          init_nol=init_nol)
    nopat = operating_profit - taxes
    fcff = nopat - reinvestment

    discount_factor = 1 + discount_rate
    if use_mid_year_convention:
        discount_factor[0] = discount_factor[0] ** 0.5
    discount_factor = np.cumprod(1.0 / discount_factor)

    # terminal value below is value as of last year in projection period
    terminal_value = compute_terminal_value(
        nopat[-1], terminal_growth_rate, discount_rate[-1], terminal_ronic)
    discounted_terminal_value = terminal_value * discount_factor[-2]
    discounted_fcff = fcff * discount_factor
    value_of_operations = np.sum(discounted_fcff[:-1]) + discounted_terminal_value
    value_of_operations = ((1 - failure_pct) * value_of_operations +
                           failure_pct * liquidation_value)

    enterprise_value = value_of_operations + init_excess_cash
    equity_value = enterprise_value - init_book_debt - init_options_value
    equity_value_per_share = None
    if shares_outstanding is not None:
        equity_value_per_share = equity_value / shares_outstanding

    # TODO: subtract value of other non-operating assets below
    invested_capital = None
    roic = None
    if init_book_equity is not None:
        # TODO: need to subtract value of other non-operating assets from invested
        #  capital.. ideally just want one other argument in addition to
        #  init_book_debt.. tricky!
        init_invested_capital = init_book_debt + init_book_equity - init_excess_cash
        invested_capital = init_invested_capital + np.cumsum(reinvestment)
        roic = np.concatenate([[nopat[0] / init_invested_capital],
                               nopat[1:] / invested_capital[:-1]])

    return EnterpriseDCF(
        revenues=revenues,
        operating_profit=operating_profit,
        taxes=taxes,
        nopat=nopat,
        fcff=fcff,
        reinvestment=reinvestment,
        reinvestment_rate=reinvestment / nopat,
        discount_factor=discount_factor,
        enterprise_value=enterprise_value,
        value_of_operations=value_of_operations,
        terminal_value=terminal_value,
        equity_value=equity_value,
        equity_value_per_share=equity_value_per_share,
        roic=roic,
        invested_capital=invested_capital,
        discounted_terminal_value=discounted_terminal_value,
        discounted_fcff=discounted_fcff,
    )
