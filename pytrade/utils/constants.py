import enum
from datetime import datetime

MIN_TIME = datetime(1900, 1, 1)
MAX_TIME = datetime(2100, 1, 1)
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
