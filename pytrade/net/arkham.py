import json
import logging
from typing import List, Dict, Optional

import pandas as pd
from pytrade.net.webdriver import get_performance_logs, get_xhr_requests
from pytrade.utils.time import sleep
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
import time

logger = logging.getLogger(__name__)

BASE_URL = "https://platform.arkhamintelligence.com"
API_BASE_URL = "https://api.arkhamintelligence.com"

TRANSACTIONS_PAG_CONTAINER_XPATH = """
(//div[contains(@class, 'Transactions_container')])[1]
//div[contains(@class, 'Transactions_paginationContainer')]
"""

TRANSACTIONS_INPUT_XPATH = (
    f"{TRANSACTIONS_PAG_CONTAINER_XPATH}"
    f"//div[contains(@class, 'Input_container')]/input"
)

TRANSACTIONS_NUM_PAGES_XPATH = (
    f"{TRANSACTIONS_PAG_CONTAINER_XPATH}"
    f"//span[contains(@class, 'Transactions_headerPageInfo')]"
)

TRANSACTION_COLUMNS = [
    "txid", "blockHeight", "blockTimestamp", "unitValue", "historicalUSD",
    "fromCoinbase", "sending", "toAddress", "toEntityName", "toEntityId",
    "toLabelName", "fromValue", "fromAddress", "fromEntityName",
    "fromEntityId", "toValue", "fromLabelName"
]


def _wait_for_num_pages_element(driver, timeout: int = 10) -> None:
    t0 = time.time()
    WebDriverWait(driver, timeout).until(EC.visibility_of_element_located(
        (By.XPATH, TRANSACTIONS_NUM_PAGES_XPATH)))
    t1 = time.time()
    logger.info(f"Waited for {(t1 - t0):.2f}s for num pages element to be visible")


def _get_total_pages(driver) -> int:
    page_info = driver.find_element(By.XPATH, TRANSACTIONS_NUM_PAGES_XPATH)
    return int(page_info.text[2:])


def _update_transactions_page(driver, page_num: int) -> None:
    page_input = driver.find_element(By.XPATH, TRANSACTIONS_INPUT_XPATH)
    page_input.send_keys(Keys.CONTROL + "a")
    page_input.send_keys(Keys.DELETE)
    page_input.send_keys(page_num)
    page_input.send_keys(Keys.RETURN)


def get_entity_transactions(driver, entity: str, start_page: int = 1,
                            end_page: Optional[int] = None,
                            page_delay: float = 0.5) -> List[Dict]:
    """

    Parameters
    ----------
    driver
    entity
    start_page
    end_page
    page_delay

    Returns
    -------
    Transactions.

    Notes
    -----
    If a single transaction sends data to multiple different wallets associated with
    the same entity, the returned dataframe will contain multiple rows.
    """
    transactions = []
    # call get_performance_logs to clear any existing logs
    get_performance_logs(driver)
    driver.get(f"{BASE_URL}/explorer/entity/{entity}")
    _wait_for_num_pages_element(driver)
    total_pages = _get_total_pages(driver)
    end_page = total_pages if end_page is None else min(end_page, total_pages)
    logger.info(f"Getting {entity} transactions for pages {start_page} to {end_page}")
    for i in range(start_page, end_page + 1):
        logger.info(f"Getting {entity} transactions for page: {i}")
        logs = get_performance_logs(driver)
        if i > 1:
            _update_transactions_page(driver, i)
            # noinspection PyTypeChecker
            # sleep so presence of num pages element prior to update isn't detected
            sleep(page_delay)
            _wait_for_num_pages_element(driver)
            logs = get_performance_logs(driver)
        # TODO: check offset in request URL is what we expect it to be?
        reqs = get_xhr_requests(logs, base_url=API_BASE_URL, endpoint="/transfers")
        if len(reqs) == 0:
            raise ValueError("Error getting entity transactions; no XHR request found")
        # TODO: ok to just use first request if multiple matching?
        request_id = reqs[0]["params"]["requestId"]
        data = driver.execute_cdp_cmd("Network.getResponseBody",
                                      {"requestId": request_id})
        transactions.extend(json.loads(data["body"])["transfers"])
    return transactions


def convert_entity_transactions_to_df(transactions: List[Dict]) -> pd.DataFrame:
    """
    Converts entity transactions into a dataframe.

    Parameters
    ----------
    transactions
        Transactions.

    Returns
    -------
    Transactions dataframe.

    Notes
    -----
    If toAddress field exists, then at least one of the output UTXOs is
    associated with the entity address; in this case the fromAddresses field
    will exist rather than the fromAddress field, and it will give the set of
    addresses associated with the input UTXOs of the transaction. The value
    field in each fromAddress gives the amount each from address contributed.
    The sum of these values (i.e., the total sum of the inputs) is given by the root
    fromValue field. Unit value will be the amount that the entity
    (e.g., Marathon Digital) received.

    If toAddress field doesn't exist, then at least one of the input UTXOs is
    associated with the entity address, and the fromAddress field will exist.
    Unit value will be amount that entity's balance will decrease by due to the
    transaction. The root toValue field equals the total sum of the outputs.

    In the Arkham UI TOTAL VALUE gives the sum of the outputs. The inputs will sum
    to TOTAL VALUE + FEE. E.g.,
    https://platform.arkhamintelligence.com/explorer/tx/47a57d6ad06040d529ba94f75863c80fd5dfe63ee40177fa4bbd6e9143cc648d
    """
    records = []
    for tx in transactions:
        sending = True
        if "toAddress" in tx:
            sending = False
            entity_address = tx["toAddress"]
        else:
            entity_address = tx["fromAddress"]
        chain = entity_address["chain"]
        # TODO: allow other chains
        if chain != "bitcoin":
            continue

        from_coinbase = tx.get("fromCoinbase", False)
        base_record = {
            "txid": tx["txid"],
            "blockHeight": tx["blockHeight"],
            "blockTimestamp": tx["blockTimestamp"],
            "unitValue": tx["unitValue"],
            "historicalUSD": tx["historicalUSD"],
            "fromCoinbase": from_coinbase,
            "sending": sending,
        }

        prefix = "from" if sending else "to"
        foreign_prefix = "to" if sending else "from"
        base_record[f"{prefix}Address"] = entity_address["address"]
        base_record[f"{prefix}EntityName"] = entity_address["arkhamEntity"]["name"]
        base_record[f"{prefix}EntityId"] = entity_address["arkhamEntity"]["id"]
        if "arkhamLabel" in entity_address:
            base_record[f"{prefix}LabelName"] = entity_address["arkhamLabel"]["name"]
        if not sending and from_coinbase:
            records.append(base_record.copy())
        else:
            for address in tx[f"{foreign_prefix}Addresses"]:
                record = base_record.copy()
                record[f"{foreign_prefix}Value"] = address["value"]
                foreign_address = address["address"]
                record[f"{foreign_prefix}Address"] = foreign_address["address"]
                if "arkhamEntity" in foreign_address:
                    record[f"{foreign_prefix}EntityName"] = foreign_address[
                        "arkhamEntity"]["name"]
                    record[f"{foreign_prefix}EntityId"] = foreign_address[
                        "arkhamEntity"]["id"]
                if "arkhamLabel" in foreign_address:
                    record[f"{foreign_prefix}LabelName"] = foreign_address[
                        "arkhamLabel"]["name"]
                records.append(record)
    transactions = pd.DataFrame(records, columns=TRANSACTION_COLUMNS)
    transactions["blockTimestamp"] = pd.to_datetime(transactions["blockTimestamp"])
    transactions["blockTimestamp"] = transactions["blockTimestamp"].dt.tz_localize(None)
    return transactions
