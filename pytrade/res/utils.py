import logging
from typing import Dict

import pandas as pd

logger = logging.getLogger(__name__)


def write_portfolio(name: str, weights: pd.DataFrame, metadata: Dict[str]):
    logger.info("Saving portfolio:")
