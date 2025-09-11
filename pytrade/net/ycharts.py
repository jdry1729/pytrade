import io
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional, Dict

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from pytrade.fundamental.consts import FinancialStatementType
from pytrade.net.constants import USER_AGENT
from pytrade.net.http import HttpMethod, HttpRequest, _send_request, is_http_error
from pytrade.utils.functions import partial
from pytrade.utils.retry import retry
from requests import Session
from tqdm import tqdm

pd.set_option("future.no_silent_downcasting", True)

logger = logging.getLogger(__name__)

BASE_URL = "https://ycharts.com"

FINANCIALS_NAME_MAP = {
    FinancialStatementType.INCOME_STATEMENT: "income_statement",
    FinancialStatementType.BALANCE_SHEET: "balance_sheet",
    FinancialStatementType.CASH_FLOW_STATEMENT: "cash_flow",
}

FINANCIALS_TABLE_MATCHES = {
    FinancialStatementType.BALANCE_SHEET: [
        "Total Assets",
        "Total Liabilities",
        # mustn't use "Shareholders Equity" as match below since string exists
        # in liabilities table too
        "Retained Earnings",
        "Book Value",
        "Originally Reported",
    ],
    FinancialStatementType.INCOME_STATEMENT: [
        "Revenue",
        "EBITDA",
        "EPS Basic",
        "Dividend Per Share",
        "Shares Outstanding",
        "Originally Reported"
    ],
    FinancialStatementType.CASH_FLOW_STATEMENT: [
        "Cash from Operations",
        "Cash from Investing",
        "Cash from Financing",
        "Ending Cash",
        "Repurchase of Capital Stock",
        "Originally Reported"
    ]
}

FINANCIALS_EXTRACT_LINKS_MATCHES = {
    FinancialStatementType.BALANCE_SHEET: ["Originally Reported"],
    FinancialStatementType.INCOME_STATEMENT: ["Originally Reported"],
    FinancialStatementType.CASH_FLOW_STATEMENT: ["Originally Reported"],
}

COLUMN_NAMES = {
    FinancialStatementType.INCOME_STATEMENT: {
        "As Originally Reported": "SEC Filing URL",
        "Restated": "SEC Filing URL (Restated)"
    },
    FinancialStatementType.BALANCE_SHEET: {
        "As Originally Reported": "SEC Filing URL",
        "Restated": "SEC Filing URL (Restated)"
    },
    FinancialStatementType.CASH_FLOW_STATEMENT: {
        "As Originally Reported": "SEC Filing URL",
        "Restated": "SEC Filing URL (Restated)"
    }
}

INCOME_STATEMENT_COLUMN_TYPES = {
    "Actual Release Date": "datetime64[ns]",
    "Operating Revenue": float,
    "Revenue": float,
    "Cost of Goods Sold": float,
    "Gross Profit": float,
    "Research and Development Expense": float,
    "SG&A Expense": float,
    "General and Administrative Expense": float,
    "Sales and Marketing Expense": float,
    "Rent and Landing Expense": float,
    "Income Statement Depreciation": float,
    "Reconciled Depreciation": float,
    "Amortization Expense": float,
    "Amortization of Intangibles": float,
    "Provision for Doubtful Accounts": float,
    "Other Operating Expenses": float,
    "Total Operating Expenses": float,
    "Operating Income": float,
    "Other Income and Expenses": float,
    "Special Income and Charges": float,
    "Net Interest Income": float,
    "Income from Continuing Operations": float,
    "Income from Discontinued Operations": float,
    "Extraordinary Items, Income Statement": float,
    "Investment Write Off": float,
    "Non-Operating Income": float,
    "Non-Operating Interest Income": float,
    "Non-Operating Interest Expense": float,
    "Net Non-Operating Interest Income Expense": float,
    "Pre-Tax Income": float,
    "Excise Taxes": float,
    "Non Income Taxes": float,
    "Provision for Income Taxes": float,
    "Income Attributable to Minority Interest": float,
    "Net Income": float,
    "Normalized Income": float,
    "Total Expenses": float,
    "EBIT": float,
    "EBITDA": float,
    "EPS Basic from Accounting Change": float,
    "EPS Basic from Continuing Operations": float,
    "EPS Basic from Discontinued Operations": float,
    "EPS Basic from Extraordinaries": float,
    "EPS Basic from Other Gains / Loss": float,
    "EPS Basic from Tax Loss Carryforward": float,
    "EPS Basic": float,
    "EPS Diluted from Accounting Change": float,
    "EPS Diluted from Continuing Operations": float,
    "EPS Diluted from Discontinued Operations": float,
    "EPS Diluted from Extraordinaries": float,
    "EPS Diluted from Other Gain/Loss": float,
    "EPS Diluted from Tax Loss Carryforward": float,
    "EPS Diluted": float,
    "Normalized Basic EPS": float,
    "Normalized Diluted EPS": float,
    "Dividend Per Share": float,
    "Preferred Stock Dividend": float,
    "Average Basic Shares Outstanding": float,
    "Average Diluted Shares Outstanding": float,
    "SEC Filing URL": "object",
    "SEC Filing URL (Restated)": "object"
}

BALANCE_SHEET_COLUMN_TYPES = {
    "Actual Release Date": "datetime64[ns]",
    "Cash": float,
    "Cash and Equivalents": float,
    "Short Term Investments": float,
    "Cash and Short Term Investments": float,
    "Accounts Receivable": float,
    "Notes Receivable": float,
    "Loans Receivable": float,
    "Other Receivables": float,
    "Total Receivables": float,
    "Raw Materials Inventory": float,
    "Work in Process Inventory": float,
    "Finished Goods Inventory": float,
    "Other Inventory": float,
    "Inventories": float,
    "Restricted Cash": float,
    "Prepaid Expenses": float,
    "Current Deferred Tax Assets": float,
    "Other Current Assets": float,
    "Total Current Assets": float,
    "Properties": float,
    "Land and Improvements": float,
    "Buildings and Improvements": float,
    "Leases": float,
    "Machine, Furniture & Equipment": float,
    "Construction in Progress": float,
    "Other Properties": float,
    "Gross PP&E": float,
    "Accumulated D&A": float,
    "Net PP&E": float,
    "Goodwill": float,
    "Other Intangible Assets": float,
    "Goodwill and Intangibles": float,
    "Gross Loans": float,
    "Allow for Loan/Lease Loss": float,
    "Unearned Income": float,
    "Net Loan Assets": float,
    "Long Term Investments": float,
    "Long Term Deferred Assets": float,
    "Long Term Deferred Tax Assets": float,
    "Long Term Notes Receivable": float,
    "Long Term Receivables": float,
    "Derivative Instruments": float,
    "Pension Asset": float,
    "Other Long Term Assets": float,
    "Total Long Term Assets": float,
    "Total Assets": float,
    "Accounts Payable": float,
    "Current Tax Payable": float,
    "Dividends Payable": float,
    "Other Payables": float,
    "Total Payables": float,
    "Accrued Expenses": float,
    "Payables and Accrued Expenses": float,
    "Notes Payable": float,
    "Liability on Credit Line": float,
    "Commercial Paper Liability": float,
    "Current Portion of Long Term Debt": float,
    "Current Capital Lease Obligation": float,
    "Other Current Borrowings": float,
    "Current Debt & Capital Lease Obligation": float,
    "Current Deferred Revenue": float,
    "Current Deferred Tax Liability": float,
    "Current Deferred Liabilities": float,
    "Current Provisions - Legal & Other": float,
    "Other Current Liability": float,
    "Total Current Liabilities": float,
    "Long Term Provisions - Legal & Other": float,
    "Long Term Cap Lease Obligation": float,
    "Non-Current Accrued Expenses": float,
    "Non-Current Portion of Long Term Debt": float,
    "Long Term Deferred Tax Liabilities": float,
    "Non-Current Deferred Revenue": float,
    "Non-Current Deferred Liabilities": float,
    "Total Deposits": float,
    "Security Sold Not Yet Repurchased": float,
    "Unpaid Loss Reserve": float,
    "Unearned Premium on Insurance Contract": float,
    "Pension Liability": float,
    "Minority Interest Ownership": float,
    "Long Term Deferred Charges": float,
    "Restricted Common Stock": float,
    "Preferred Securities out of Shareholders Equity": float,
    "Derivative Contract Liabilities": float,
    "Other Long Term Liabilities": float,
    "Total Long Term Liabilities": float,
    "Total Liabilities": float,
    "Total Capital Stock": float,
    "Treasury Stock": float,
    "Minimum Pension Liabilities": float,
    "Retained Earnings": float,
    "Additional Paid In Capital": float,
    "Unrealized Gain or Loss - Total": float,
    "Other Equity Adjustments": float,
    "Adjustments for Foreign Currency Translation": float,
    "Preferred Stock": float,
    "Accrued Comprehensive Inc": float,
    "Shareholders Equity": float,
    "Ordinary Shares Number": float,
    "Tangible Book Value": float,
    "Book Value": float,
    "Total Equity Including Minority Interest": float,
    "SEC Filing URL": "object",
    "SEC Filing URL (Restated)": "object"
}

CASH_FLOW_STATEMENT_COLUMN_TYPES = {
    'Actual Release Date': "datetime64[ns]",
    'Net Income': float,
    'Gain/Loss on Sale Business': float,
    'Net Foreign Currency Exchange Gain/Loss': float,
    'Pension and Employee Expense': float,
    'Gain and Loss on Sale of PPE': float,
    'Gain (Loss) on Investment Securities': float,
    'Earnings Loss from Eq. Investments': float,
    'Operating Gains Losses': float,
    'Amortization Expense CF': float,
    'Depreciation Expense': float,
    'Total Depreciation and Amortization': float,
    'Total Depreciation, Amortization, Depletion': float,
    'Deferred Taxes': float,
    'Amortization of Securities': float,
    'Asset Impairment Charge': float,
    'Stock Based Compensation': float,
    'Excess Tax Benefit from Stock Compensation': float,
    'Other Noncash Items': float,
    'Change in Receivables': float,
    'Change in Inventories': float,
    'Change in Prepaid Assets': float,
    'Change in Payables and Accrued Expense': float,
    'Change in Other Current Assets': float,
    'Change in Other Current Liabilities': float,
    'Change in Other Working Cap': float,
    'Change in Taxes Payable': float,
    'Changes in Working Capital': float,
    'Other Cash from Operations': float,
    'Cash from Operations': float,
    'Net Change in Capital Expenditures': float,
    'Sale of PPE': float,
    'Net Change in PP&E': float,
    'Net Change in Intangibles': float,
    'Net Divestitures (Acquisitions)': float,
    'Total Net Change in Investments': float,
    'Net Other Investing Changes': float,
    'Cash from Investing': float,
    'Net Change in Long Term Debt': float,
    'Net Change in Short Term Debt': float,
    'Net Debt Issuance': float,
    'Common Stock Issuance': float,
    'Common Stock Payments': float,
    'Net Common Equity Issued (Purchased)': float,
    'Preferred Stock Issuance': float,
    'Preferred Stock Payments': float,
    'Net Preferred Equity Issued (Purchased)': float,
    'Total Common Dividends Paid': float,
    'Total Preferred Dividends Paid': float,
    'Total Dividends Paid': float,
    'Proceeds from Stock Option Exercised': float,
    'Cash from Other Financing Activities': float,
    'Cash from Financing': float,
    'Beginning Cash': float,
    'Change in Cash': float,
    'Cash Foreign Exchange Adjustment': float,
    'Ending Cash': float,
    'Issuance of Capital Stock': float,
    'Issuance of Debt': float,
    'Debt Repayment': float,
    'Repurchase of Capital Stock': float,
    'Income Tax Paid Supplemental Data': float,
    'Interest Paid Supplemental Data': float,
    'Domestic Sales': float,
    'Foreign Sales': float,
    "SEC Filing URL": "object",
    "SEC Filing URL (Restated)": "object"
}

QUARTERLY_INCOME_STATEMENT_COLUMN_TYPES = INCOME_STATEMENT_COLUMN_TYPES.copy()
QUARTERLY_INCOME_STATEMENT_COLUMN_TYPES[
    "Other Comprehensive Income (Quarterly)"] = float
QUARTERLY_INCOME_STATEMENT_COLUMN_TYPES["Total Interest Expense (Quarterly)"] = float
QUARTERLY_INCOME_STATEMENT_COLUMN_TYPES["EBITDA Margin (Quarterly)"] = float

ANNUAL_INCOME_STATEMENT_COLUMN_TYPES = INCOME_STATEMENT_COLUMN_TYPES.copy()
ANNUAL_INCOME_STATEMENT_COLUMN_TYPES["Other Comp Income (Annual)"] = float
ANNUAL_INCOME_STATEMENT_COLUMN_TYPES["Interest Expense (Annual)"] = float
ANNUAL_INCOME_STATEMENT_COLUMN_TYPES["EBITDA Margin (Annual)"] = float

QUARTERLY_BALANCE_SHEET_COLUMN_TYPES = BALANCE_SHEET_COLUMN_TYPES.copy()
QUARTERLY_BALANCE_SHEET_COLUMN_TYPES["Total Long Term Debt (Quarterly)"] = float
QUARTERLY_BALANCE_SHEET_COLUMN_TYPES["Non-Curr LTD & Lease Q"] = float

ANNUAL_BALANCE_SHEET_COLUMN_TYPES = BALANCE_SHEET_COLUMN_TYPES.copy()
ANNUAL_BALANCE_SHEET_COLUMN_TYPES["Total Long Term Debt (Annual)"] = float
ANNUAL_BALANCE_SHEET_COLUMN_TYPES["Non-Curr LTD & Lease Ann"] = float

QUARTERLY_CASH_FLOW_STATEMENT_COLUMN_TYPES = CASH_FLOW_STATEMENT_COLUMN_TYPES.copy()
QUARTERLY_CASH_FLOW_STATEMENT_COLUMN_TYPES["Unreali Gain/Loss Inv Q"] = float
QUARTERLY_CASH_FLOW_STATEMENT_COLUMN_TYPES["Cash Disc Ops Adjust Q"] = float
QUARTERLY_CASH_FLOW_STATEMENT_COLUMN_TYPES["Stock Buybacks (Quarterly)"] = float

ANNUAL_CASH_FLOW_STATEMENT_COLUMN_TYPES = CASH_FLOW_STATEMENT_COLUMN_TYPES.copy()
ANNUAL_CASH_FLOW_STATEMENT_COLUMN_TYPES["Unreali Gain/Loss Inv Ann"] = float
ANNUAL_CASH_FLOW_STATEMENT_COLUMN_TYPES["Cash Disc Ops Adjust Ann"] = float
ANNUAL_CASH_FLOW_STATEMENT_COLUMN_TYPES["Stock Buyback (Annual)"] = float

# column types are different for quarterly/ annual periods
FINANCIALS_COLUMN_TYPES = {
    "quarterly": {
        FinancialStatementType.INCOME_STATEMENT:
            QUARTERLY_INCOME_STATEMENT_COLUMN_TYPES,
        FinancialStatementType.BALANCE_SHEET:
            QUARTERLY_BALANCE_SHEET_COLUMN_TYPES,
        FinancialStatementType.CASH_FLOW_STATEMENT:
            QUARTERLY_CASH_FLOW_STATEMENT_COLUMN_TYPES,
    },
    "annual": {
        FinancialStatementType.INCOME_STATEMENT:
            ANNUAL_INCOME_STATEMENT_COLUMN_TYPES,
        FinancialStatementType.BALANCE_SHEET:
            ANNUAL_BALANCE_SHEET_COLUMN_TYPES,
        FinancialStatementType.CASH_FLOW_STATEMENT:
            ANNUAL_CASH_FLOW_STATEMENT_COLUMN_TYPES,
    },
}


class Metric(Enum):
    SHORT_INTEREST = 0


@dataclass
class Credentials:
    username: str
    password: str


def get_company_metadata(symbol: str) -> Dict:
    req = HttpRequest(
        base_url=BASE_URL,
        endpoint=f"/companies/{symbol}",
        method=HttpMethod.GET,
        headers={
            "User-Agent": USER_AGENT,
        }
    )
    res = retry(_send_request, e=partial(
        is_http_error, status_code=(429,)), args=(req,))
    soup = BeautifulSoup(res.content, "html.parser")
    data = soup.find("ycn-security-header-control")["security"]
    return json.loads(data)


def login(session: Session, credentials: Credentials):
    req1 = HttpRequest(
        base_url=BASE_URL,
        endpoint="/login",
        method=HttpMethod.GET,
        headers={
            "User-Agent": USER_AGENT,
        }
    )
    res1 = _send_request(req1, session=session, raise_for_status=False)
    csrftoken = session.cookies.get("csrftoken", domain="ycharts.com")
    soup = BeautifulSoup(res1.content, "html.parser")
    csrfmiddlewaretoken = soup.find("input", {"name": "csrfmiddlewaretoken"})["value"]

    req2 = HttpRequest(
        base_url=BASE_URL,
        endpoint="/login",
        method=HttpMethod.POST,
        data={
            "account_login_view-current_step": "auth",
            "auth-username": credentials.username,
            "auth-password": credentials.password,
            "auth-remember_me": "on",
            "csrfmiddlewaretoken": csrfmiddlewaretoken
        },
        headers={
            "User-Agent": USER_AGENT,
            "Content-Type": "application/x-www-form-urlencoded",
            "Origin": BASE_URL,
            "Referer": "https://ycharts.com/login",
            "cookie": f"csrftoken={csrftoken}",
        }
    )
    _send_request(req2, session=session)
    logger.info("Successfully logged in to YCharts")


def str_to_float(x):
    if x == "--":
        return np.nan
    # you can have -1.42K%, so need to allow two of BMK{}
    if re.match(r"^-?\d+(\.\d+)?[BMK%]{0,2}$", x):
        if x.endswith("%"):
            x = x[:-1]
        if x.endswith("B"):
            return float(x[:-1]) * 1e9
        elif x.endswith("M"):
            return float(x[:-1]) * 1e6
        elif x.endswith("K"):
            return float(x[:-1]) * 1e3
        return float(x)
    return x


def get_short_interest(session: Session, company_id: int,
                       start_time: datetime = datetime(2000, 1, 1),
                       end_time: Optional[datetime] = None) -> pd.DataFrame:
    """
    Gets short interest for a company.

    Parameters
    ----------
    session
        Session. Must be logged in.
    company_id
        ID of company.
    start_time
        Start time.
    end_time
        End time.

    Returns
    -------
    Short interest.
    """
    if end_time is None:
        end_time = datetime.utcnow()

    start_time_str = start_time.strftime("%m/%d/%Y")
    end_time_str = end_time.strftime("%m/%d/%Y")

    def send_request_(page_num: int):
        params = {
            "startDate": start_time_str,
            "endDate": end_time_str,
            "pageNum": page_num,

        }
        req = HttpRequest(
            base_url=BASE_URL,
            endpoint=f"/companies/{company_id}/short_interest.json",
            params=params,
            headers={
                "User-Agent": USER_AGENT,
            },
        )
        return retry(_send_request, args=(req,), e=partial(
            is_http_error, status_code=(429,)), session=session).json()

    data = []
    num_pages = send_request_(1)["last_page_num"]
    for i in tqdm(range(num_pages)):
        data_ = send_request_(i + 1)["data_table_html"]
        data.append(pd.read_html(io.StringIO(data_))[0])

    data = pd.concat(data)

    data["Date"] = pd.to_datetime(data["Date"])
    data = data.set_index("Date")

    data = data.replace(
        {"--": np.nan, "M": "e6", "%": ""}, regex=True).astype(float)
    data[["% of Float Short", "% of SO Short"]] /= 100.0
    return data.sort_index()


def get_metric(session: Session, company_id: int, metric: Metric,
               start_time: datetime = datetime(2000, 1, 1),
               end_time: Optional[datetime] = None) -> pd.DataFrame:
    if metric == Metric.SHORT_INTEREST:
        return get_short_interest(session, company_id, start_time, end_time)


def change_financial_statement_period(session: Session,
                                      statement_type: FinancialStatementType,
                                      period: str = "quarterly"):
    name = FINANCIALS_NAME_MAP[statement_type]
    csrftoken = session.cookies.get("csrftoken", domain="ycharts.com")
    req = HttpRequest(
        method=HttpMethod.PUT,
        base_url=BASE_URL,
        endpoint="/accounts/preferences",
        headers={
            "User-Agent": USER_AGENT,
            "origin": BASE_URL,
            "x-csrftoken": csrftoken,
        },
        json={f"stock_financials_{name}_format": period},
    )
    res = retry(_send_request, args=(req, session))
    return res


def _update_column_period(column: str, period: str):
    return column.replace("(Quarterly)", f"({period.title()})")


def get_financials(
        session: Session, symbol: str,
        statement_type: FinancialStatementType
        = FinancialStatementType.INCOME_STATEMENT,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        period: str = "quarterly",
):
    page_num = 1
    num_pages = None
    extract_links_matches = FINANCIALS_EXTRACT_LINKS_MATCHES[statement_type]

    change_financial_statement_period(session, statement_type, period)
    column_types = FINANCIALS_COLUMN_TYPES[period][statement_type]

    data = []
    while True:
        page_num_str = f"page {page_num}"
        if num_pages is not None:
            page_num_str += f" of {num_pages}"

        logger.info(f"Getting {statement_type.name} data ({page_num_str})")
        path = FINANCIALS_NAME_MAP[statement_type]
        req = HttpRequest(
            base_url=BASE_URL,
            endpoint=f"/companies/{symbol}/financials/{path}/{page_num}",
            headers={
                "User-Agent": USER_AGENT,
            },
        )
        res = retry(_send_request, args=(req,), e=partial(
            is_http_error, status_code=(429,)), session=session)

        if num_pages is None:
            num_pages = 1
            soup = BeautifulSoup(res.content, "html.parser")
            pagination = soup.find("div", {"class": "panel-pagination"})
            a_tag = pagination.find(
                "a", string=lambda text: text and "First Period" in text)
            if a_tag is not None:
                num_pages = int(a_tag["href"].split("/")[-1])

        data_ = []
        res = io.StringIO(res.text)
        for match in FINANCIALS_TABLE_MATCHES[statement_type]:
            extract_links = "body" if match in extract_links_matches else None
            table = pd.read_html(res, header=0, match=match, index_col=0,
                                 extract_links=extract_links)[0]
            if extract_links:
                # must convert link to str!
                table = table.map(lambda x: str(x[1]) if x[1] is not None else x[0])
                table.index = table.index.map(lambda x: x[0])

            table = table[[x for x in table.columns if not x.startswith("Unnamed")]].T
            table.index = pd.to_datetime(table.index)
            table = table.map(str_to_float)
            table.index.name = "Time"
            table.columns.name = None
            data_.append(table)

        data_ = pd.concat(data_, axis=1)
        data_ = data_.loc[:, ~data_.columns.duplicated()]
        data.append(data_)

        page_num += 1
        if page_num > num_pages or (
                start_time is not None and data_.index.min() < start_time):
            break

    data = pd.concat(data).rename(columns=COLUMN_NAMES[statement_type])
    data = data.reindex(columns=column_types).astype(column_types).sort_index()

    if start_time is not None:
        data = data.loc[start_time:]
    if end_time is not None:
        data = data.loc[:end_time]
    return data
