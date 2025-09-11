from edgar.xbrl import XBRLS

from pytrade.fundamental.consts import FinancialStatementType
from pytrade.utils.collections import topological_sort

STATEMENT_TYPE_MAP = {
    FinancialStatementType.INCOME_STATEMENT: "income_statement",
    FinancialStatementType.BALANCE_SHEET: "balance_sheet",
    FinancialStatementType.CASH_FLOW_STATEMENT: "cashflow_statement",
}


def get_statement(xbrls: XBRLS, statement_type: FinancialStatementType):
    stmt_type = STATEMENT_TYPE_MAP[statement_type]
    concepts = topological_sort(
        *[
            getattr(x.statements, stmt_type)()
            .to_dataframe(standard=False)["concept"]
            .values
            for x in xbrls.xbrl_list
        ]
    )
    stmt = getattr(xbrls.statements, stmt_type)(
        max_periods=100, standardize=False
    ).to_dataframe()
    stmt = stmt.set_index(["label", "concept"]).stack()
    stmt.index = stmt.index.rename("time", level=2)
    stmt = stmt.sort_index(level="time")
    labels = {x[1]: x[0] for x in stmt.index}
    stmt = stmt.droplevel("label")
    stmt = stmt.unstack()
    stmt = stmt.reindex(index=concepts)
    stmt = stmt.loc[~stmt.isnull().all(axis=1)]
    stmt.index = stmt.index.map(labels)
    return stmt
