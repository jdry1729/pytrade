import logging
import os
from dataclasses import dataclass
from datetime import timedelta
from typing import Optional

import pandas as pd
from pytrade.net.webdriver import element_exists
from pytrade.utils.files import wait_for_file
from pytrade.utils.time import sleep
from selenium.webdriver.chrome.webdriver import WebDriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait

logger = logging.getLogger(__name__)


@dataclass
class Credentials:
    email: str
    password: str


def login(driver: WebDriver, credentials: Credentials):
    driver.get("https://instrack.app/login")

    email_xpath = "//input[@id='login-email']"
    WebDriverWait(driver, 10).until(EC.element_to_be_clickable(
        (By.XPATH, email_xpath)))

    driver.find_element(By.XPATH, email_xpath).send_keys(credentials.email)

    password_xpath = "//input[@id='login-password']"
    driver.find_element(By.XPATH, password_xpath).send_keys(credentials.password)

    submit_xpath = "//button[@type='submit']"
    driver.find_element(By.XPATH, submit_xpath).click()


def delete_tracked_accounts(driver: WebDriver):
    logger.info("Deleting tracked accounts")
    driver.get("https://instrack.app")

    modal_xpath = "//div[contains(@class, 'modal-content')]"
    if element_exists(driver, (By.XPATH, modal_xpath), timeout=3):
        close_xpath = "//button[@class='close']"
        driver.find_element(By.XPATH, close_xpath).click()

    prev_num_accounts = None
    button_xpath = "//tr[@role='row']/td/button"
    while True:
        delete_buttons = driver.find_elements(By.XPATH, button_xpath)
        if delete_buttons:
            num_accounts = len(delete_buttons)
            if prev_num_accounts is not None and num_accounts >= prev_num_accounts:
                raise ValueError("Error deleting tracked accounts")
            prev_num_accounts = num_accounts
            driver.execute_script("arguments[0].click();", delete_buttons[0])
            sleep(2)
        else:
            break


def get_history(driver: WebDriver, username: str, period: str = "All Time",
                driver_download_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Gets history for a particular brand. Assumes driver is already logged in, and
    that number of tracked accounts is less than the account's maximum.
    """
    if driver_download_dir is None:
        driver_download_dir = os.path.expanduser("~/selenium")
    if not os.path.exists(driver_download_dir):
        os.makedirs(driver_download_dir)

    # use execute_script approach for clicking on elements instead of calling
    # click method directly (since the latter approach gives interactable errors)
    params = {"behavior": "allow", "downloadPath": driver_download_dir}
    driver.execute_cdp_cmd("Page.setDownloadBehavior", params)

    file_path = os.path.join(driver_download_dir,
                             f"{username}-historical-stats.csv")
    driver.get(f"https://instrack.app/instagram/{username}")

    dropdown_xpath = "//button[contains(@id, '__BVID__')]"
    dropdown = driver.find_element(By.XPATH, dropdown_xpath)
    driver.execute_script("arguments[0].click();", dropdown)

    period_xpath = f"//button[contains(text(), '{period}')]"
    period_element = driver.find_element(By.XPATH, period_xpath)
    driver.execute_script("arguments[0].click();", period_element)

    sleep(1)

    history = ("//ul[contains(@class,'navbar-nav')]//div[contains(text(),"
               " 'History')]")
    history = driver.find_element(By.XPATH, history)
    driver.execute_script("arguments[0].click();", history)

    # sleep so page updates with full history rather than just past 30 days
    sleep(7)

    export_xpath = "//div[contains(@id,'export')]"
    export = WebDriverWait(driver, 3).until(EC.presence_of_element_located(
        (By.XPATH, export_xpath)))
    driver.execute_script("arguments[0].click();", export)

    wait_for_file(file_path, timeout=timedelta(seconds=10))
    data = pd.read_csv(file_path, index_col="date", parse_dates=True).sort_index()
    os.remove(file_path)
    return data
