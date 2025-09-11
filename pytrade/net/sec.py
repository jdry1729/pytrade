import io
import json
import logging
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Optional, Tuple, Dict, Any, Collection, Union, List, Sequence

import numpy as np
import pandas as pd
from lxml import etree
from lxml.etree import Element, XMLSyntaxError

from pytrade.data.xbrl import read_filing, get_fiscal_period, get_current_context_id, \
    XBRLFiling, LabelType
from pytrade.net.http import HttpMethod, HttpRequest, _send_request
from pytrade.utils.collections import flatten, topological_sort, ensure_list
from pytrade.utils.pandas import get_one_row, empty_df
from pytrade.utils.retry import retry

pd.set_option("future.no_silent_downcasting", True)

logger = logging.getLogger(__name__)

BASE_URL_1 = "https://www.sec.gov"
BASE_URL_2 = "https://data.sec.gov"

USER_AGENT = "Mozilla/5.0 (Dummy dummy@dummy.com)"

XBRL_FACTS_DTYPES = {
    "id": "string",
    "name": str,
    "value": "object",
    "start_time": "datetime64[ns]",
    "end_time": "datetime64[ns]",
    "dimensions": "object",
    "unit": "string"
}

FILINGS_COLS = {
    "acceptanceDateTime": "acceptance_time",
    "accessionNumber": "accession_number",
    "filingDate": "filing_date",
    "reportDate": "report_date",
    "act": "act",
    "form": "form",
    "fileNumber": "file_number",
    "filmNumber": "film_number",
    "items": "items",
    "size": "size",
    "isXBRL": "is_xbrl",
    "isInlineXBRL": "is_inline_xbrl",
    "primaryDocument": "primary_document",
    "primaryDocDescription": "primary_doc_description",
}

HOLDINGS_COLS = {
    "nameOfIssuer": "string",
    "titleOfClass": "string",
    "cusip": "string",
    "value": "Int64",
    "shrsOrPrnAmt.sshPrnamt": "Int64",
    "shrsOrPrnAmt.sshPrnamtType": "string",
    "investmentDiscretion": "string",
    "otherManager": "string",
    "votingAuthority.Sole": "Int64",
    "votingAuthority.Shared": "string",
    "votingAuthority.None": "string"
}

FILINGS_TIME_COLS = ["filing_date", "report_date", "acceptance_time"]

REVENUE_CONCEPTS = [
    "us-gaap:Revenues",
    "us-gaap:SalesRevenueNet",
    "us-gaap:SalesRevenueGoodsNet",
    "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
    "us-gaap:RevenueFromContractWithCustomerIncludingAssessedTax",
    "ifrs-full:RevenueFromContractsWithCustomers",
    "ifrs-full:Revenue",
]


def get_cik(symbol: str) -> int:
    req = HttpRequest(
        base_url=BASE_URL_1,
        endpoint="/cgi-bin/browse-edgar",
        method=HttpMethod.POST,
        data={
            "action": "getcompany",
            "CIK": symbol,
            "count": 10,
            "output": "xml",
        },
        headers={"User-Agent": USER_AGENT}
    )
    res = retry(_send_request, args=(req,))
    root = etree.fromstring(res.content)
    res = root.xpath("//companyInfo//CIK/text()")
    if res:
        return int(res[0])
    raise ValueError(f"Error getting CIK for {symbol}; not found")


def get_filing_url(accession_number: str, file_name: Optional[str] = None,
                   cik: Optional[int] = None) -> str:
    if cik is None:
        cik = get_cik_from_accession_number(accession_number)
    accession_number = accession_number.replace("-", "")
    url = (f"https://www.sec.gov/Archives/edgar/data/{cik}/"
           f"{accession_number}")
    if file_name is not None:
        url += f"/{file_name}"
    return url


def get_filings(cik: int, forms: Optional[Collection[str]] = None) -> pd.DataFrame:
    def _get_data(file_name: str):
        req = HttpRequest(
            base_url=BASE_URL_2,
            endpoint=f"/submissions/{file_name}",
            method=HttpMethod.GET,
            headers={"User-Agent": USER_AGENT}
        )
        return retry(_send_request, args=(req,)).json()

    data = []
    cik = str(cik).zfill(10)
    res = _get_data(f"CIK{cik}.json")
    data.append(pd.DataFrame(res["filings"]["recent"]))
    for file in res["filings"]["files"]:
        res = _get_data(file["name"])
        data.append(pd.DataFrame(res))

    data = pd.concat(data)
    data = data[FILINGS_COLS.keys()]
    # rename columns to match those returned by get_xbrl_facts
    data = data.rename(columns=FILINGS_COLS)
    data[FILINGS_TIME_COLS] = data[FILINGS_TIME_COLS].apply(
        lambda x: pd.to_datetime(x).dt.tz_localize(None))
    data = data.set_index("acceptance_time").sort_index()

    if forms is not None:
        data = data.loc[data["form"].isin(forms)]
    return data


def get_company_concept(cik: int, taxonomy: str, tag: str) -> pd.DataFrame:
    cik = str(cik).zfill(10)
    req = HttpRequest(
        base_url=BASE_URL_2,
        endpoint=f"/api/xbrl/companyconcept/CIK{cik}/{taxonomy}/{tag}.json",
        method=HttpMethod.GET,
        headers={"User-Agent": USER_AGENT}
    )
    res = retry(_send_request, args=(req,))
    return res.json()


def get_cik_from_accession_number(accession_number: str) -> int:
    return int(accession_number[:10])


def _get_filing_content(accession_number: str, file_name: str,
                        cik: Optional[int] = None) -> bytes:
    accession_number = accession_number.replace("-", "")
    # if CIK of entity changes, filings with an accession number starting with the
    # old CIK are migrated to the data directory associated with the new CIK - hence
    # the need for the optional cik arg here
    if cik is None:
        cik = get_cik_from_accession_number(accession_number)
    req = HttpRequest(
        base_url=BASE_URL_1,
        endpoint=f"/Archives/edgar/data/{cik}/{accession_number}/{file_name}",
        headers={"user-agent": USER_AGENT}
    )
    res = retry(_send_request, args=(req,))
    return res.content


def _get_xbrl_units(root: Element) -> pd.DataFrame:
    units = []
    for e in root.findall(".//{*}unit", root.nsmap):
        child = list(e)[0]
        tag = etree.QName(child).localname
        if tag == "measure":
            name = child.text.strip()
        elif tag == "divide":
            names = [child.find(f".//{{*}}unit{x}").find(".//{*}measure").text.strip()
                     for x in ["Numerator", "Denominator"]]
            name = f"{names[0]}/{names[1]}"
        else:
            raise ValueError("Error parsing unit; child tag must be \"measure\" or"
                             " \"divide\"")
        units.append({"id": e.get("id"), "tag": tag, "name": name})
    return pd.DataFrame(units, columns=["id", "tag", "name"]).set_index("id")


def _get_xbrl_contexts(root: Element) -> pd.DataFrame:
    contexts = []
    for e in root.findall(".//{*}context", root.nsmap):
        context_id = e.get('id')
        context: Dict[str, Any] = {"id": context_id}

        if (instant := e.find(".//{*}instant", root.nsmap)) is not None:
            end_time = datetime.strptime(instant.text.strip(), "%Y-%m-%d")
            context["end_time"] = end_time
            context["period_type"] = "instant"
        elif (start_time := e.find(".//{*}startDate", root.nsmap)) is not None:
            start_time = start_time.text.strip()
            end_time = e.find(".//{*}endDate", root.nsmap).text.strip()
            context["start_time"] = datetime.strptime(start_time, "%Y-%m-%d")
            context["end_time"] = datetime.strptime(end_time, "%Y-%m-%d")
            context["period_type"] = "duration"
        else:
            raise ValueError("Unknown period type")

        dimensions = {}
        entity = e.find(".//{*}entity", root.nsmap)
        if (segment := entity.find(".//{*}segment", root.nsmap)) is not None:
            for dimension in segment:
                dimension_name = dimension.get("dimension")
                if "explicitMember" in dimension.tag:
                    dimension_value = dimension.text
                elif "typedMember" in dimension.tag:
                    dimension_value = list(dimension)[0].text
                else:
                    logger.warning(f"Ignoring unknown dimension: {dimension}")
                    continue
                # dimension value text can be None!
                if dimension_value is not None:
                    dimension_value = dimension_value.strip()
                dimensions[dimension_name] = dimension_value
        context["dimensions"] = json.dumps(dimensions, sort_keys=True)
        contexts.append(context)
    return pd.DataFrame(
        contexts, columns=["id", "start_time", "end_time", "period_type",
                           "dimensions"]).set_index("id")


def _str_to_float(s) -> float:
    s = s.lower()
    if s in ["—", "none", "", "no"]:
        return np.nan
    try:
        return int(s)
    except Exception:
        return float(s)


def get_xbrl_facts(accession_number: str, file_name: str,
                   cik: Optional[int] = None) -> pd.DataFrame:
    """
    Gets XBRL facts from an XML XBRL file. Doesn't work with IXBRL.

    Parameters
    ----------
    accession_number
        Accession number. Must be hyphenated version.
    file_name
        File to extract XBRL from.
    cik
        Registrant CIK. Must be passed if filing wasn't filed by the registrant itself.
        Since in that case registrant CIK cannot be inferred from accession number.

    Returns
    -------
    XBRL facts.

    Notes
    -----
    Dimensions are stored in JSON format (i.e., as a string).
    """
    try:
        root = etree.fromstring(_get_filing_content(accession_number, file_name, cik))
    except XMLSyntaxError:
        raise ValueError("Error getting XBRL facts; error parsing file")

    units = _get_xbrl_units(root)
    contexts = _get_xbrl_contexts(root)

    facts = []
    for context_id in contexts.index:
        context = contexts.loc[context_id]
        for fact in root.findall(f".//*[@contextRef='{context_id}']", root.nsmap):
            prefix = fact.prefix
            local_name = etree.QName(fact).localname
            name = f"{prefix}:{local_name}"
            # must use itertext below to handle case where fact tag surrounds inner
            # fact with common value
            value = "".join(fact.itertext())

            unit_name = pd.NA
            unit_ref = fact.get("unitRef")
            if unit_ref is not None:
                unit = units.loc[unit_ref]
                unit_name = unit["name"]
                # all numeric facts must have unit
                if unit["tag"] in ["measure", "divide"]:
                    try:
                        value = _str_to_float(value)
                    except (AttributeError, ValueError):
                        logger.warning(f"Error converting fact value to float;"
                                       f" {name=}, {value=}")
                        pass

            facts.append({
                "id": fact.get("id", pd.NA),
                "name": name,
                "value": value,
                "start_time": context["start_time"],
                "end_time": context["end_time"],
                "dimensions": context["dimensions"],
                "unit": unit_name,
            })

    # don't set index to ID since facts don't always have ID
    return pd.DataFrame(facts, columns=list(XBRL_FACTS_DTYPES)).astype(
        XBRL_FACTS_DTYPES)


def infer_reporting_period(facts: pd.DataFrame) -> \
        Tuple[datetime, datetime]:
    """
    Infers a form's reporting period from its facts.
    """
    fact_names = facts["name"].unique()
    for name in ["dei:DocumentPeriodEndDate", "dei:DocumentFiscalPeriodFocus"]:
        if name not in fact_names:
            raise ValueError(f"Error inferring reporting period; no fact"
                             f" named: {name}")

    # document period end date should always match reporting period end time
    end_time = get_one_row(facts, name="dei:DocumentPeriodEndDate")["value"]
    fiscal_period = get_one_row(facts, name="dei:DocumentFiscalPeriodFocus")["value"]

    period_size = timedelta(days=90)
    if fiscal_period == "FY":
        period_size = timedelta(days=365)

    end_time = datetime.strptime(end_time, "%Y-%m-%d")
    start_times = list(
        facts[(facts["name"].isin(REVENUE_CONCEPTS)) &
              (facts["dimensions"] == "{}") &
              (facts["end_time"] == end_time)]["start_time"].unique())
    period_diff = [abs(end_time - x - period_size) for x in start_times]
    start_time = start_times[np.argmin(period_diff)]

    return start_time, end_time


def get_filing_documents(accession_number: str,
                         cik: Optional[int] = None) -> pd.DataFrame:
    """
    Gets filing documents.

    Parameters
    ----------
    accession_number
        Accession number. Must be hyphenated version.
    cik
        Registrant CIK. Must be passed if filing wasn't filed by the registrant itself.
        Since in that case registrant CIK cannot be inferred from accession number.

    Returns
    -------
    Filing documents.
    """
    if cik is None:
        cik = get_cik_from_accession_number(accession_number)
    req = HttpRequest(
        base_url=BASE_URL_1,
        endpoint=f"/Archives/edgar/data/{cik}/{accession_number.replace('-', '')}"
                 f"/{accession_number}-index.html",
        headers={"User-Agent": USER_AGENT}
    )
    res = _send_request(req)
    data = pd.concat(pd.read_html(io.StringIO(str(res.content)), match="Document"))
    data[["Description", "Document"]] = data[["Description", "Document"]].fillna("")
    return data


def get_xbrl_file_name(accession_number: str, cik: Optional[int] = None) -> str:
    """
    Gets XBRL file name.

    Parameters
    ----------
    accession_number
        Accession number. Must be hyphenated version.
    cik
        Registrant CIK. Must be passed if filing wasn't filed by the registrant itself.
        Since in that case registrant CIK cannot be inferred from accession number.

    Returns
    -------
    XBRL file name.
    """
    if cik is None:
        cik = get_cik_from_accession_number(accession_number)

    logger.debug(f"Getting filing documents; {cik=}, {accession_number=}")
    documents = get_filing_documents(accession_number, cik)
    # must fillna below since description is sometimes empty
    documents = documents.loc[
        documents["Document"].str.match("^(?!.*_(cal|def|lab|pre)\.xml$).*\.xml$") |
        # TODO: maybe don't need condition below?
        documents["Description"].str.contains("INSTANCE DOCUMENT|INSTANCE FILE")
        ]
    if documents.empty:
        raise ValueError("Error getting XBRL file name; no candidate files found")
    if len(documents) > 1:
        raise ValueError("Error getting XBRL file name; multiple candidate files"
                         " found")
    return documents.iloc[0]["Document"]


def get_13f_file_name(accession_number: str, cik: Optional[int] = None) -> str:
    if cik is None:
        cik = get_cik_from_accession_number(accession_number)

    logger.debug(f"Getting filing documents; {cik=}, {accession_number=}")
    documents = get_filing_documents(accession_number, cik)
    # must fillna below since description is sometimes empty
    documents = documents.loc[
        documents["Document"].str.match("^.*\.xml$") &
        # TODO: maybe don't need condition below?
        documents["Type"].str.contains("INFORMATION TABLE")
        ]
    if documents.empty:
        raise ValueError("Error getting 13F-HR file name; no candidate files found")
    if len(documents) > 1:
        raise ValueError("Error getting 13F-HR file name; multiple candidate files"
                         " found")
    return documents.iloc[0]["Document"]


def parse_element(element):
    if len(element):
        return {etree.QName(child.tag).localname: parse_element(child) for child in
                element}
    else:
        return element.text


def get_13f_holdings(accession_number: str, file_name: str,
                     cik: Optional[int] = None):
    try:
        root = etree.fromstring(_get_filing_content(accession_number, file_name, cik))
    except XMLSyntaxError:
        raise ValueError("Error getting 13F holdings; error parsing file")

    data = []
    for info in root.findall('.//{*}infoTable', root.nsmap):
        record = parse_element(info)
        data.append(record)
    data = [flatten(x) for x in data]

    if len(data):
        return pd.DataFrame(data).reindex(
            columns=list(HOLDINGS_COLS.keys())).astype(HOLDINGS_COLS)
    return empty_df(columns=list(HOLDINGS_COLS.keys())).astype(HOLDINGS_COLS)


@lru_cache
def get_xbrl_filing(accession_number: str, cik: Optional[int] = None) -> XBRLFiling:
    if cik is None:
        cik = get_cik_from_accession_number(accession_number)
    return read_filing(get_filing_url(
        accession_number, get_xbrl_file_name(accession_number, cik), cik=cik))


def get_table_for_contexts(
        cik: int, role_uri: Union[str, List[str]],
        context_ids: Dict[
            str, Union[Collection[str], Dict[str, Union[str, Sequence[str]]]]],
        label_type: Optional[LabelType] = LabelType.STANDARD,
) -> pd.DataFrame:
    accession_numbers = list(context_ids.keys())
    role_uris = ensure_list(role_uri)

    tables = []
    for accession_number in accession_numbers:
        filing = get_xbrl_filing(accession_number, cik=cik)
        role_uris_ = filing.get_roles(referenced_only=True)["uri"].unique()
        role_uris_ = [x for x in role_uris if x in role_uris_]
        if len(role_uris_) == 0:
            logging.warning(f"Error extracting table from filing: {accession_number};"
                            f" no role found")
            continue

        context_ids_ = context_ids[accession_number]
        if not isinstance(context_ids_, dict):
            context_ids_ = dict(zip(context_ids_, context_ids_))

        table = filing.get_table(
            role_uris_[0],
            context_id=tuple(context_ids_.keys()),
            label_type=label_type,
        ).rename(columns=context_ids_)

        tables.append(table)

    res = pd.concat(tables, axis=1)

    try:
        index = topological_sort(*[x.index for x in tables])
    except ValueError:
        logger.warning("Error topologically sorting table concepts; infeasible")
    else:
        res = res.reindex(index)

    return res


def get_table_over_time(
        cik: int, role_uri: Union[str, List[str]], *,
        forms: Optional[Collection[str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
) -> pd.DataFrame:
    """
    Gets table over time.

    Parameters
    ----------
    cik
    role_uri
        Can be a list if multiple URIs are used over time for referring to the
        same table.
    forms
        Forms to look for table in.
    start_time
        Start time.
    end_time
        End time.

    Returns
    -------
    Table.
    """
    filings = get_filings(cik)
    role_uris = ensure_list(role_uri)

    filings = filings.loc[filings["is_xbrl"] == 1]
    if forms is not None:
        filings = filings.loc[filings["form"].isin(forms)]
    if start_time is not None:
        filings = filings.loc[start_time:]
    if end_time is not None:
        filings = filings.loc[:end_time]

    logger.info(f"Extracting table from {len(filings)} filings")

    tables = []
    for i in range(len(filings)):
        accession_number = filings.iloc[i]["accession_number"]

        filing = get_xbrl_filing(accession_number, cik=cik)
        role_uris_ = filing.get_roles(referenced_only=True)["uri"].unique()
        role_uris_ = [x for x in role_uris if x in role_uris_]
        if len(role_uris_) == 0:
            logging.warning(f"Error extracting table from filing: {accession_number};"
                            f" no role found")
            continue

        period_context = get_current_context_id(filing, "period")
        instant_context = get_current_context_id(filing, "instant")
        table = filing.get_table(
            role_uris_[0],
            context_id=[period_context, instant_context],
        )

        invalid_mask = (~table.isnull()).sum(axis=1) > 1
        if invalid_mask.any():
            invalid_facts = table.loc[invalid_mask].index.tolist()
            logger.warning("Ignoring facts since non-nan values for both period and"
                           f" instant contexts: {', '.join(invalid_facts)}")
            table = table.loc[~invalid_mask]

        # to get the value of a fact, use whichever of the period or instant value
        # is non-nan, or nan if both are nan
        table = table.bfill(axis=1).iloc[:, 0]

        table.name = "".join(map(str, get_fiscal_period(filing)))
        tables.append(table)

    res = pd.concat(tables, axis=1)

    try:
        index = topological_sort(*[x.index for x in tables])
    except ValueError:
        logger.warning("Error topologically sorting table concepts; infeasible")
    else:
        res = res.reindex(index)

    return res
