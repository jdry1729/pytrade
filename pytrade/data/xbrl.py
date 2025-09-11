import enum
import os
from datetime import timedelta, datetime
from functools import lru_cache
from typing import Dict, Collection, List, Tuple, Optional, Union, \
    Literal, Set, Iterable

import numpy as np
import pandas as pd
from arelle import Cntlr, XbrlConst, XmlUtil
from arelle.ModelDtsObject import ModelConcept
from arelle.ModelRelationshipSet import ModelRelationshipSet
from arelle.ModelValue import QName as QName_
from arelle.ModelXbrl import ModelContext, ModelXbrl

from pytrade.utils.collections import ensure_list
from pytrade.utils.time import date_to_datetime


def get_prefix_and_local_name(qname: str) -> Tuple[Optional[str], str]:
    if ":" in qname:
        return qname.split(":")
    return None, qname


class LabelType(enum.Enum):
    STANDARD = 0
    TERSE = 1


_LABEL_TYPE_MAP = {
    LabelType.STANDARD: XbrlConst.standardLabel,
    LabelType.TERSE: XbrlConst.terseLabel,
}


def _get_dimensions(c: ModelContext) -> Dict[str, str]:
    dims = {}
    for k in sorted(c.qnameDims, key=lambda x: str(x)):
        v = c.qnameDims[k]
        if v.isExplicit:
            v = v.memberQname
        else:
            v = XmlUtil.innerText(v.typedMember)
        dims[str(k)] = v
    return dims


def _extract(concept: ModelConcept, rset: ModelRelationshipSet,
             order: Tuple[int, ...], out: List[Tuple[int, str]]):
    out.append((order, str(concept.qname)))
    for rel in rset.fromModelObject(concept):
        _extract(rel.toModelObject, rset, order + (int(rel.order),), out)


class XBRLFiling:
    def __init__(self, model_xbrl: ModelXbrl):
        self._model_xbrl = model_xbrl

        self._contexts = self._get_contexts()
        self._facts = self._get_facts()
        self._roles = self._get_roles()
        self._referenced_role_uris = self._get_referenced_role_uris()

    @lru_cache
    def _get_contexts(self) -> pd.DataFrame:
        res = []
        contexts: Iterable[ModelContext] = self._model_xbrl.contexts.values()
        for context in contexts:
            # endDate attribute gives raw end date specified in period element
            # startDatetime gives raw start datetime specified in period element
            res.append({
                "id": context.id, "start_time": context.startDatetime,
                "end_time": date_to_datetime(context.endDate),
                "dimensions": _get_dimensions(context)
            })
        return pd.DataFrame(res).set_index("id")

    @lru_cache
    def _get_facts(self, deduplicate: bool = True) -> pd.DataFrame:
        res = []
        for fact in self._model_xbrl.facts:
            res.append(
                {"id": fact.id, "qname": str(fact.qname), "context_ref": fact.contextID,
                 "unit_ref": fact.unitID, "value": fact.value})
        res = pd.DataFrame(res)
        if deduplicate:
            res = res.loc[~res.duplicated(
                subset=["qname", "context_ref", "unit_ref", "value"])]
        return res

    @lru_cache
    def _get_roles(self) -> pd.DataFrame:
        res = []
        for role_uri, role_types_ in self._model_xbrl.roleTypes.items():
            role_type = list(role_types_)[0]
            res.append(
                {"id": role_type.id, "definition": role_type.definition,
                 "uri": role_type.roleURI})
        return pd.DataFrame(res)

    def get_contexts(self) -> pd.DataFrame:
        return self._contexts.copy(deep=False)

    def get_facts(self) -> pd.DataFrame:
        return self._facts.copy(deep=False)

    def get_table(
            self,
            role_uri: str,
            *,
            context_id: Optional[Union[str, Collection[str]]] = None,
            label_type: Optional[LabelType] = LabelType.STANDARD,
    ) -> pd.DataFrame:
        facts = self._facts.copy(deep=False)

        if context_id is not None:
            context_id = ensure_list(context_id)
            facts = facts.loc[facts["context_ref"].isin(context_id)]

        presentation = self.get_presentation(role_uri)
        facts = facts.loc[facts["qname"].isin(presentation["qname"].unique())]

        facts["label"] = facts["qname"]
        if label_type is not None:
            facts["label"] = facts["label"].apply(self._get_label, type_=label_type)

        facts = facts.loc[~facts.duplicated(subset=["qname", "context_ref"],
                                            keep="first")]
        facts = pd.merge(facts, presentation, on="qname").sort_values("order")

        table = pd.pivot(facts, columns="context_ref", index="label", values="value")
        table.columns = table.columns.rename(None)
        table.index = table.index.rename("fact")

        # reindexing to labels ensures correct row ordering
        return table.reindex(index=facts["label"].unique(), columns=context_id)

    def get_one_fact(self, qname: str,
                     context_id: Optional[str] = None) -> pd.Series:
        facts = self._facts.loc[self._facts["qname"] == qname]
        if context_id is not None:
            facts = facts.loc[facts["context_ref"] == context_id]

        if facts.empty:
            raise ValueError("Error getting fact; no facts found")

        if len(facts) > 1:
            raise ValueError("Error getting fact; multiple facts found")

        return facts.iloc[0]

    def get_roles(self, *, pattern: Optional[str] = None,
                  referenced_only: bool = True) -> pd.DataFrame:
        roles = self._roles.copy(deep=False)
        if referenced_only:
            roles = roles.loc[roles["uri"].isin(self._referenced_role_uris)]
        if pattern is not None:
            roles = roles.loc[roles["definition"].str.match(pattern)]
        return roles.reset_index(drop=True)

    @lru_cache
    def get_presentation(self, role_uri: str, deduplicate: bool = True) -> pd.DataFrame:
        rset = self._model_xbrl.relationshipSet(
            XbrlConst.parentChild, linkrole=role_uri)

        res = []
        for i, concept in enumerate(rset.rootConcepts):
            _extract(concept, rset, (i,), res)

        res = pd.DataFrame(res, columns=["order", "qname"])
        if deduplicate:
            res = res.loc[~res.duplicated("qname", keep="last")]
        return res

    @property
    def nsmap(self) -> pd.Series:
        return pd.Series(self._model_xbrl.prefixedNamespaces)

    @lru_cache
    def _get_referenced_role_uris(self) -> Set[str]:
        return set(self._model_xbrl.relationshipSet(XbrlConst.parentChild).linkRoleUris)

    @lru_cache
    def _get_label(self, qname: str,
                   type_: LabelType = LabelType.STANDARD) -> str:
        prefix, local_name = get_prefix_and_local_name(qname)

        qname_ = QName_(prefix=prefix,
                        namespaceURI=self.nsmap[prefix],
                        localName=local_name)
        concept = self._model_xbrl.qnameConcepts[qname_]
        return concept.label(preferredLabel=_LABEL_TYPE_MAP[type_])


def read_filing(path: str) -> XBRLFiling:
    """
    Reads a filing.

    Parameters
    ----------
    path
        Path to XBRL instance document. Can be local path or URL.

    Returns
    -------
    Filing.
    """
    cntlr = Cntlr.Cntlr(logFileName=os.devnull)
    model_xbrl = cntlr.modelManager.load(path)
    return XBRLFiling(model_xbrl=model_xbrl)


def get_fiscal_period(filing: XBRLFiling) -> Tuple[int, str]:
    fiscal_year = int(filing.get_one_fact("dei:DocumentFiscalYearFocus")["value"])
    fiscal_period = filing.get_one_fact("dei:DocumentFiscalPeriodFocus")["value"]
    return fiscal_year, fiscal_period


def infer_reporting_period(filing: XBRLFiling) -> Tuple[datetime, datetime]:
    fiscal_period = filing.get_one_fact("dei:DocumentFiscalPeriodFocus")["value"]
    end_time = datetime.strptime(filing.get_one_fact(
        "dei:DocumentPeriodEndDate")["value"], "%Y-%m-%d")

    period_size = timedelta(days=90)
    if fiscal_period == "FY":
        period_size = timedelta(days=365)

    contexts = filing.get_contexts()
    contexts = contexts.loc[~contexts["start_time"].isnull()]
    start_times = list(
        contexts.loc[(contexts["start_time"] is not None) &
                     (contexts["end_time"] == end_time)]["start_time"].unique()
    )

    period_diff = [abs(end_time - x - period_size) for x in start_times]
    start_time = start_times[np.argmin(period_diff)]
    if isinstance(start_time, pd.Timestamp):
        start_time = start_time.to_pydatetime()

    return start_time, end_time


def get_current_context_id(
        filing: XBRLFiling,
        mode: Literal["period", "instant"] = "period") -> str:
    if mode not in ["period", "instant"]:
        raise ValueError("Error getting current context ID; mode must be \"period\""
                         " or \"instant\"")

    reporting_period = infer_reporting_period(filing)
    contexts = filing.get_contexts()
    contexts = contexts.loc[
        (((contexts["start_time"] == reporting_period[0]) if mode == "period" else
          contexts["start_time"].isnull()) &
         (contexts["end_time"] == reporting_period[1]) &
         (contexts["dimensions"] == {}))
    ]

    if len(contexts) == 0:
        raise ValueError("Error getting current context; no contexts found")

    if len(contexts) > 1:
        raise ValueError("Error getting current context; multiple contexts found")

    return contexts.iloc[0].name
