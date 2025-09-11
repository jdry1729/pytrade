import logging
from datetime import datetime
from functools import lru_cache
from typing import List, Iterable, Optional, Dict

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

from pytrade.fundamental.consts import FinancialStatementType
from pytrade.net.constants import USER_AGENT, USER_AGENT_2
from pytrade.net.http import HttpRequest, _send_request
from pytrade.utils.constants import MIN_TIME
from pytrade.utils.pandas import empty_df, empty_time_idx
from pytrade.utils.retry import retry

BASE_URL_1 = "https://query1.finance.yahoo.com"
BASE_URL_2 = "https://query2.finance.yahoo.com"

# using additional uk subdomain seems to avoid rate limits
BASE_URL_3 = "https://uk.finance.yahoo.com"

METADATA_MODULES = ["summaryProfile", "price"]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_2) "
                  "AppleWebKit/601.3.9 (KHTML, like Gecko) Version/9.0.2 "
                  "Safari/601.3.9"}

HISTORY_INDEX = ["time"]
HISTORY_COLUMNS = ["open", "high", "low", "close", "volume"]

SPLIT_DTYPES = {
    "numerator": float,
    "denominator": float,
    "splitRatio": "string",
}

BALANCE_SHEET_TYPES = {
    "TreasurySharesNumber",
    "PreferredSharesNumber",
    "OrdinarySharesNumber",
    "ShareIssued",
    "NetDebt",
    "TotalDebt",
    "TangibleBookValue",
    "InvestedCapital",
    "WorkingCapital",
    "NetTangibleAssets",
    "CapitalLeaseObligations",
    "CommonStockEquity",
    "PreferredStockEquity",
    "TotalCapitalization",
    "TotalEquityGrossMinorityInterest",
    "MinorityInterest",
    "StockholdersEquity",
    "OtherEquityInterest",
    "GainsLossesNotAffectingRetainedEarnings",
    "OtherEquityAdjustments",
    "FixedAssetsRevaluationReserve",
    "ForeignCurrencyTranslationAdjustments",
    "MinimumPensionLiabilities",
    "UnrealizedGainLoss",
    "TreasuryStock",
    "RetainedEarnings",
    "AdditionalPaidInCapital",
    "CapitalStock",
    "OtherCapitalStock",
    "CommonStock",
    "PreferredStock",
    "TotalPartnershipCapital",
    "GeneralPartnershipCapital",
    "LimitedPartnershipCapital",
    "TotalLiabilitiesNetMinorityInterest",
    "TotalNonCurrentLiabilitiesNetMinorityInterest",
    "OtherNonCurrentLiabilities",
    "LiabilitiesHeldforSaleNonCurrent",
    "RestrictedCommonStock",
    "PreferredSecuritiesOutsideStockEquity",
    "DerivativeProductLiabilities",
    "EmployeeBenefits",
    "NonCurrentPensionAndOtherPostretirementBenefitPlans",
    "NonCurrentAccruedExpenses",
    "DuetoRelatedPartiesNonCurrent",
    "TradeandOtherPayablesNonCurrent",
    "NonCurrentDeferredLiabilities",
    "NonCurrentDeferredRevenue",
    "NonCurrentDeferredTaxesLiabilities",
    "LongTermDebtAndCapitalLeaseObligation",
    "LongTermCapitalLeaseObligation",
    "LongTermDebt",
    "LongTermProvisions",
    "CurrentLiabilities",
    "OtherCurrentLiabilities",
    "CurrentDeferredLiabilities",
    "CurrentDeferredRevenue",
    "CurrentDeferredTaxesLiabilities",
    "CurrentDebtAndCapitalLeaseObligation",
    "CurrentCapitalLeaseObligation",
    "CurrentDebt",
    "OtherCurrentBorrowings",
    "LineOfCredit",
    "CommercialPaper",
    "CurrentNotesPayable",
    "PensionandOtherPostRetirementBenefitPlansCurrent",
    "CurrentProvisions",
    "PayablesAndAccruedExpenses",
    "CurrentAccruedExpenses",
    "InterestPayable",
    "Payables",
    "OtherPayable",
    "DuetoRelatedPartiesCurrent",
    "DividendsPayable",
    "TotalTaxPayable",
    "IncomeTaxPayable",
    "AccountsPayable",
    "TotalAssets",
    "TotalNonCurrentAssets",
    "OtherNonCurrentAssets",
    "DefinedPensionBenefit",
    "NonCurrentPrepaidAssets",
    "NonCurrentDeferredAssets",
    "NonCurrentDeferredTaxesAssets",
    "DuefromRelatedPartiesNonCurrent",
    "NonCurrentNoteReceivables",
    "NonCurrentAccountsReceivable",
    "FinancialAssets",
    "InvestmentsAndAdvances",
    "OtherInvestments",
    "InvestmentinFinancialAssets",
    "HeldToMaturitySecurities",
    "AvailableForSaleSecurities",
    "FinancialAssetsDesignatedasFairValueThroughProfitorLossTotal",
    "TradingSecurities",
    "LongTermEquityInvestment",
    "InvestmentsinJointVenturesatCost",
    "InvestmentsInOtherVenturesUnderEquityMethod",
    "InvestmentsinAssociatesatCost",
    "InvestmentsinSubsidiariesatCost",
    "InvestmentProperties",
    "GoodwillAndOtherIntangibleAssets",
    "OtherIntangibleAssets",
    "Goodwill",
    "NetPPE",
    "AccumulatedDepreciation",
    "GrossPPE",
    "Leases",
    "ConstructionInProgress",
    "OtherProperties",
    "MachineryFurnitureEquipment",
    "BuildingsAndImprovements",
    "LandAndImprovements",
    "Properties",
    "CurrentAssets",
    "OtherCurrentAssets",
    "HedgingAssetsCurrent",
    "AssetsHeldForSaleCurrent",
    "CurrentDeferredAssets",
    "CurrentDeferredTaxesAssets",
    "RestrictedCash",
    "PrepaidAssets",
    "Inventory",
    "InventoriesAdjustmentsAllowances",
    "OtherInventories",
    "FinishedGoods",
    "WorkInProcess",
    "RawMaterials",
    "Receivables",
    "ReceivablesAdjustmentsAllowances",
    "OtherReceivables",
    "DuefromRelatedPartiesCurrent",
    "TaxesReceivable",
    "AccruedInterestReceivable",
    "NotesReceivable",
    "LoansReceivable",
    "AccountsReceivable",
    "AllowanceForDoubtfulAccountsReceivable",
    "GrossAccountsReceivable",
    "CashCashEquivalentsAndShortTermInvestments",
    "OtherShortTermInvestments",
    "CashAndCashEquivalents",
    "CashEquivalents",
    "CashFinancial",
}

# when you load up fundamentals in yahoo, data for all the types below prefixed
# with "trailing" is also requested
INCOME_STATEMENT_TYPES = {
    "TaxEffectOfUnusualItems",
    "TaxRateForCalcs",
    "NormalizedEBITDA",
    "NormalizedDilutedEPS",
    "NormalizedBasicEPS",
    "TotalUnusualItems",
    "TotalUnusualItemsExcludingGoodwill",
    "NetIncomeFromContinuingOperationNetMinorityInterest",
    "ReconciledDepreciation",
    "ReconciledCostOfRevenue",
    "EBITDA",
    "EBIT",
    "NetInterestIncome",
    "InterestExpense",
    "InterestIncome",
    "ContinuingAndDiscontinuedDilutedEPS",
    "ContinuingAndDiscontinuedBasicEPS",
    "NormalizedIncome",
    "NetIncomeFromContinuingAndDiscontinuedOperation",
    "TotalExpenses",
    "RentExpenseSupplemental",
    "ReportedNormalizedDilutedEPS",
    "ReportedNormalizedBasicEPS",
    "TotalOperatingIncomeAsReported",
    "DividendPerShare",
    "DilutedAverageShares",
    "BasicAverageShares",
    "DilutedEPS",
    "DilutedEPSOtherGainsLosses",
    "TaxLossCarryforwardDilutedEPS",
    "DilutedAccountingChange",
    "DilutedExtraordinary",
    "DilutedDiscontinuousOperations",
    "DilutedContinuousOperations",
    "BasicEPS",
    "BasicEPSOtherGainsLosses",
    "TaxLossCarryforwardBasicEPS",
    "BasicAccountingChange",
    "BasicExtraordinary",
    "BasicDiscontinuousOperations",
    "BasicContinuousOperations",
    "DilutedNIAvailtoComStockholders",
    "AverageDilutionEarnings",
    "NetIncomeCommonStockholders",
    "OtherunderPreferredStockDividend",
    "PreferredStockDividends",
    "NetIncome",
    "MinorityInterests",
    "NetIncomeIncludingNoncontrollingInterests",
    "NetIncomeFromTaxLossCarryforward",
    "NetIncomeExtraordinary",
    "NetIncomeDiscontinuousOperations",
    "NetIncomeContinuousOperations",
    "EarningsFromEquityInterestNetOfTax",
    "TaxProvision",
    "PretaxIncome",
    "OtherIncomeExpense",
    "OtherNonOperatingIncomeExpenses",
    "SpecialIncomeCharges",
    "GainOnSaleOfPPE",
    "GainOnSaleOfBusiness",
    "OtherSpecialCharges",
    "WriteOff",
    "ImpairmentOfCapitalAssets",
    "RestructuringAndMergernAcquisition",
    "SecuritiesAmortization",
    "EarningsFromEquityInterest",
    "GainOnSaleOfSecurity",
    "NetNonOperatingInterestIncomeExpense",
    "TotalOtherFinanceCost",
    "InterestExpenseNonOperating",
    "InterestIncomeNonOperating",
    "OperatingIncome",
    "OperatingExpense",
    "OtherOperatingExpenses",
    "OtherTaxes",
    "ProvisionForDoubtfulAccounts",
    "DepreciationAmortizationDepletionIncomeStatement",
    "DepletionIncomeStatement",
    "DepreciationAndAmortizationInIncomeStatement",
    "Amortization",
    "AmortizationOfIntangiblesIncomeStatement",
    "DepreciationIncomeStatement",
    "ResearchAndDevelopment",
    "SellingGeneralAndAdministration",
    "SellingAndMarketingExpense",
    "GeneralAndAdministrativeExpense",
    "OtherGandA",
    "InsuranceAndClaims",
    "RentAndLandingFees",
    "SalariesAndWages",
    "GrossProfit",
    "CostOfRevenue",
    "TotalRevenue",
    "ExciseTaxes",
    "OperatingRevenue",
}

# when you load up fundamentals in yahoo, data for all the types below prefixed
# with "trailing" is also requested
CASH_FLOW_TYPES = {
    "ForeignSales",
    "DomesticSales",
    "AdjustedGeographySegmentData",
    "FreeCashFlow",
    "RepurchaseOfCapitalStock",
    "RepaymentOfDebt",
    "IssuanceOfDebt",
    "IssuanceOfCapitalStock",
    "CapitalExpenditure",
    "InterestPaidSupplementalData",
    "IncomeTaxPaidSupplementalData",
    "EndCashPosition",
    "OtherCashAdjustmentOutsideChangeinCash",
    "BeginningCashPosition",
    "EffectOfExchangeRateChanges",
    "ChangesInCash",
    "OtherCashAdjustmentInsideChangeinCash",
    "CashFlowFromDiscontinuedOperation",
    "FinancingCashFlow",
    "CashFromDiscontinuedFinancingActivities",
    "CashFlowFromContinuingFinancingActivities",
    "NetOtherFinancingCharges",
    "InterestPaidCFF",
    "ProceedsFromStockOptionExercised",
    "CashDividendsPaid",
    "PreferredStockDividendPaid",
    "CommonStockDividendPaid",
    "NetPreferredStockIssuance",
    "PreferredStockPayments",
    "PreferredStockIssuance",
    "NetCommonStockIssuance",
    "CommonStockPayments",
    "CommonStockIssuance",
    "NetIssuancePaymentsOfDebt",
    "NetShortTermDebtIssuance",
    "ShortTermDebtPayments",
    "ShortTermDebtIssuance",
    "NetLongTermDebtIssuance",
    "LongTermDebtPayments",
    "LongTermDebtIssuance",
    "InvestingCashFlow",
    "CashFromDiscontinuedInvestingActivities",
    "CashFlowFromContinuingInvestingActivities",
    "NetOtherInvestingChanges",
    "InterestReceivedCFI",
    "DividendsReceivedCFI",
    "NetInvestmentPurchaseAndSale",
    "SaleOfInvestment",
    "PurchaseOfInvestment",
    "NetInvestmentPropertiesPurchaseAndSale",
    "SaleOfInvestmentProperties",
    "PurchaseOfInvestmentProperties",
    "NetBusinessPurchaseAndSale",
    "SaleOfBusiness",
    "PurchaseOfBusiness",
    "NetIntangiblesPurchaseAndSale",
    "SaleOfIntangibles",
    "PurchaseOfIntangibles",
    "NetPPEPurchaseAndSale",
    "SaleOfPPE",
    "PurchaseOfPPE",
    "CapitalExpenditureReported",
    "OperatingCashFlow",
    "CashFromDiscontinuedOperatingActivities",
    "CashFlowFromContinuingOperatingActivities",
    "TaxesRefundPaid",
    "InterestReceivedCFO",
    "InterestPaidCFO",
    "DividendReceivedCFO",
    "DividendPaidCFO",
    "ChangeInWorkingCapital",
    "ChangeInOtherWorkingCapital",
    "ChangeInOtherCurrentLiabilities",
    "ChangeInOtherCurrentAssets",
    "ChangeInPayablesAndAccruedExpense",
    "ChangeInAccruedExpense",
    "ChangeInInterestPayable",
    "ChangeInPayable",
    "ChangeInDividendPayable",
    "ChangeInAccountPayable",
    "ChangeInTaxPayable",
    "ChangeInIncomeTaxPayable",
    "ChangeInPrepaidAssets",
    "ChangeInInventory",
    "ChangeInReceivables",
    "ChangesInAccountReceivables",
    "OtherNonCashItems",
    "ExcessTaxBenefitFromStockBasedCompensation",
    "StockBasedCompensation",
    "UnrealizedGainLossOnInvestmentSecurities",
    "ProvisionandWriteOffofAssets",
    "AssetImpairmentCharge",
    "AmortizationOfSecurities",
    "DeferredTax",
    "DeferredIncomeTax",
    "DepreciationAmortizationDepletion",
    "Depletion",
    "DepreciationAndAmortization",
    "AmortizationCashFlow",
    "AmortizationOfIntangibles",
    "Depreciation",
    "OperatingGainsLosses",
    "PensionAndEmployeeBenefitExpense",
    "EarningsLossesFromEquityInvestments",
    "GainLossOnInvestmentSecurities",
    "NetForeignCurrencyExchangeGainLoss",
    "GainLossOnSaleOfPPE",
    "GainLossOnSaleOfBusiness",
    "NetIncomeFromContinuingOperations",
    "CashFlowsfromusedinOperatingActivitiesDirect",
    "TaxesRefundPaidDirect",
    "InterestReceivedDirect",
    "InterestPaidDirect",
    "DividendsReceivedDirect",
    "DividendsPaidDirect",
    "ClassesofCashPayments",
    "OtherCashPaymentsfromOperatingActivities",
    "PaymentsonBehalfofEmployees",
    "PaymentstoSuppliersforGoodsandServices",
    "ClassesofCashReceiptsfromOperatingActivities",
    "OtherCashReceiptsfromOperatingActivities",
    "ReceiptsfromGovernmentGrants",
    "ReceiptsfromCustomers",
}

VALUATION_TYPES = {
    "quarterlyMarketCap",
    "trailingMarketCap",
    "quarterlyEnterpriseValue",
    "trailingEnterpriseValue",
    "quarterlyPeRatio",
    "trailingPeRatio",
    "quarterlyForwardPeRatio",
    "trailingForwardPeRatio",
    "quarterlyPegRatio",
    "trailingPegRatio",
    "quarterlyPsRatio",
    "trailingPsRatio",
    "quarterlyPbRatio",
    "trailingPbRatio",
    "quarterlyEnterprisesValueRevenueRatio",
    "trailingEnterprisesValueRevenueRatio",
    "quarterlyEnterprisesValueEBITDARatio",
    "trailingEnterprisesValueEBITDARatio"
}

METADATA_FIELDS = [
    "currency",
    "symbol",
    "exchangeName",
    "fullExchangeName",
    "instrumentType",
    "firstTradeDate",
    "exchangeTimezoneName"
]

STATEMENT_TYPE_TO_FUNDAMENTALS_MAP = {
    FinancialStatementType.BALANCE_SHEET: BALANCE_SHEET_TYPES,
    FinancialStatementType.INCOME_STATEMENT: INCOME_STATEMENT_TYPES,
    FinancialStatementType.CASH_FLOW_STATEMENT: CASH_FLOW_TYPES,
}

logger = logging.getLogger(__name__)


def get_metadata(symbols: List[str]) -> pd.DataFrame:
    results = [_get_symbol_metadata(x) for x in symbols]
    return pd.DataFrame(results).set_index("symbol")


def get_stock_splits(
        ticker: str, start_time: datetime, end_time: Optional[datetime] = None
) -> pd.DataFrame:
    req = HttpRequest(
        base_url="https://query1.finance.yahoo.com",
        endpoint=f"/v8/finance/chart/{ticker}",
        params={
            "events": "capitalGain|div|split",
            "interval": "1d",
            "period1": int(start_time.timestamp()),
            "period2": int(end_time.timestamp()),
        },
        headers={
            "User-Agent": USER_AGENT_2,
        },
    )
    res = retry(_send_request, args=(req,)).json()
    res = res["chart"]["result"][0]
    if "events" in res and "splits" in res["events"]:
        data = list(res["events"]["splits"].values())
        if data:
            data = pd.DataFrame(data)
            data["date"] = pd.to_datetime(data["date"], unit="s")
            return data.set_index("date").reindex(
                columns=list(SPLIT_DTYPES.keys())).astype(
                SPLIT_DTYPES)
    return empty_df(empty_time_idx("date"), columns=list(SPLIT_DTYPES.keys())).astype(
        SPLIT_DTYPES)


def get_price_history(symbol: str, start_time: datetime,
                      end_time: datetime, freq: str = "1d") -> pd.DataFrame:
    """

    Parameters
    ----------
    symbol
    start_time
    end_time
    freq
        May be 1d, 1m, 2m, 5m or 15m.

    Returns
    -------
    Price history. Index is open time in UTC.
    """
    url = f"{BASE_URL_2}/v8/finance/chart/{symbol}"
    params = {"period1": int(start_time.timestamp()),
              "period2": int(end_time.timestamp()),
              "interval": freq, "includePrePost": False}
    res = requests.get(url, params, headers=HEADERS)
    json = res.json()
    error = json["chart"]["error"]
    if error is not None:
        raise ValueError(f"Error getting data for: {symbol}; {error['description']}")
    data = _clean_chart_data(json).sort_index()
    return data[HISTORY_COLUMNS]


def get_description(symbol: str):
    """
    Gets description for company.

    Parameters
    ----------
    symbol
        Symbol.

    Returns
    -------
    Description of company.
    """
    endpoint = f"/quote/{symbol}/profile/"
    req = HttpRequest(base_url=BASE_URL_3, endpoint=endpoint,
                      headers={"User-Agent": USER_AGENT})
    res = retry(_send_request, args=(req,), max_tries=3)
    soup = BeautifulSoup(res.text, features="html.parser")
    span = soup.find("span", string="Description")
    if span is None:
        raise ValueError(f"Error getting description for: {symbol}; not found")
    return span.parent.parent.find("p").text


@lru_cache
def search(query):
    res = requests.get(f"{BASE_URL_2}/v1/finance/search", params={"q": query},
                       headers=HEADERS)
    return res.json()["quotes"]


@lru_cache
def get_index_constituents(symbol):
    # not all indices are supported
    res = requests.get(f"{BASE_URL_2}/v10/finance/quoteSummary/{symbol}",
                       params={"modules": "components"},
                       headers=HEADERS)
    result = res.json()["quoteSummary"]["result"]
    return result[0]["components"]["components"]


@lru_cache
def _get_symbol_metadata(symbol: str):
    # TODO: below gives "invalid crumb" error
    res1 = requests.get(f"{BASE_URL_2}/v10/finance/quoteSummary/{symbol}",
                        params={"modules": ",".join(METADATA_MODULES)},
                        headers=HEADERS)
    res2 = requests.get(f"{BASE_URL_2}/v7/finance/quote",
                        params={"symbols": symbol},
                        headers=HEADERS)
    data1 = res1.json()["quoteSummary"]["result"][0]
    data2 = res2.json()["quoteResponse"]["result"][0]
    return {
        "industry": data1["summaryProfile"]["industry"],
        "sector": data1["summaryProfile"]["sector"],
        "country": data1["summaryProfile"]["country"],
        "exchange": data1["price"]["exchange"],
        "exchangeName": data1["price"]["exchangeName"],
        "marketState": data1["price"]["marketState"],
        "symbol": data1["price"]["symbol"],
        "shortName": data1["price"]["shortName"],
        "currency": data1["price"]["currency"],
        "exchangeTimezoneName": data2["exchangeTimezoneName"],
        "exchangeTimezoneShortName": data2["exchangeTimezoneShortName"]
    }


def _clean_chart_data(chart_data):
    result = chart_data["chart"]["result"][0]
    index = pd.DatetimeIndex(pd.to_datetime(result["timestamp"], unit="s"),
                             name="time")
    return pd.DataFrame(result["indicators"]["quote"][0], index=index).astype(float)


def get_fundamental_timeseries(symbol: str, types: Iterable[str],
                               start_time: Optional[datetime] = None,
                               end_time: Optional[datetime] = None):
    if start_time is None:
        start_time = MIN_TIME
    if end_time is None:
        end_time = datetime.utcnow()
    period1 = int(start_time.timestamp())
    period2 = int(end_time.timestamp())
    type_: str = ",".join(types)
    req = HttpRequest(
        base_url=BASE_URL_1,
        endpoint=f"/ws/fundamentals-timeseries/v1/finance/timeseries/{symbol}",
        params={
            "merge": "false",
            "padTimeSeries": "true",
            "period1": period1,
            "period2": period2,
            "type": type_,
            "lang": "en-US",
            "region": "US",
        },
        headers=HEADERS
    )
    res = retry(_send_request, args=(req,), max_tries=3).json()

    data = []
    for item in res["timeseries"]["result"]:
        item_type = item["meta"]["type"][0]
        if item_type in item:
            times = item["timestamp"]
            records = item[item_type]
            for i, time in enumerate(times):
                record = {"time": time, "item": item_type, "value": np.nan}
                if records[i] is not None:
                    record["value"] = records[i]["reportedValue"]["raw"]
                data.append(record)
    data = pd.DataFrame(data, columns=["time", "item", "value"])
    data["time"] = pd.to_datetime(data["time"], unit="s")
    # sort index so sorted by time
    return data.set_index(["time", "item"])["value"].sort_index()


def get_financials(
        symbol: str,
        statement_type: FinancialStatementType = FinancialStatementType.BALANCE_SHEET,
        period: str = "annual"
):
    """
    Gets financial data from Yahoo Finance.

    Parameters
    ----------
    symbol
        Symbol to get financials for.
    statement_type
        Financial statement to get.
    period
        Can be annual or quaterly.

    Returns
    -------
    Dataframe containing financials.
    """

    base_types = STATEMENT_TYPE_TO_FUNDAMENTALS_MAP[statement_type]
    types = {f"{period}{x}" for x in base_types}
    if statement_type in [FinancialStatementType.INCOME_STATEMENT,
                          FinancialStatementType.CASH_FLOW_STATEMENT]:
        types.update({f"trailing{x}" for x in base_types})
    return get_fundamental_timeseries(symbol, types)


def get_metadata_v2(symbol: str) -> Dict:
    req = HttpRequest(
        base_url="https://query1.finance.yahoo.com",
        endpoint="/v7/finance/spark",
        params={
            "range": "1d",
            "symbols": symbol
        },
        headers={
            "User-Agent": USER_AGENT,
        }
    )
    data = retry(_send_request, args=(req,)).json()
    data = data["spark"]["result"][0]["response"][0]["meta"]
    data = {k: data[k] for k in METADATA_FIELDS}
    if data["firstTradeDate"] is not None:
        data["firstTradeDate"] = datetime.fromtimestamp(data["firstTradeDate"])
    return data
