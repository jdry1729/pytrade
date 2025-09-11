import enum

# TODO: create enum?
ACCOUNTING_PERIODS = ["annual", "quarterly"]


class FinancialStatementType(enum.Enum):
    BALANCE_SHEET = 0
    INCOME_STATEMENT = 1
    CASH_FLOW_STATEMENT = 2


FINANCIAL_STATEMENT_TYPES = [
    FinancialStatementType.BALANCE_SHEET,
    FinancialStatementType.INCOME_STATEMENT,
    FinancialStatementType.CASH_FLOW_STATEMENT,
]
