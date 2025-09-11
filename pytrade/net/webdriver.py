import json
import logging
import tempfile
import time
import zipfile
from dataclasses import dataclass
from typing import Optional, Dict, List, Any, Tuple

from pytrade.net.constants import USER_AGENT
from pytrade.net.http import parse_url
from selenium import webdriver
from selenium.common import TimeoutException
from selenium.webdriver.chrome.webdriver import WebDriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait

logger = logging.getLogger(__name__)

MANIFEST_JSON = """
{
    "version": "1.0.0",
    "manifest_version": 2,
    "name": "Chrome Proxy",
    "permissions": [
        "proxy",
        "tabs",
        "unlimitedStorage",
        "storage",
        "<all_urls>",
        "webRequest",
        "webRequestBlocking"
    ],
    "background": {
        "scripts": ["background.js"]
    },
    "minimum_chrome_version":"22.0.0"
}
"""

BACKGROUND_JS = """
var config = {
        mode: "fixed_servers",
        rules: {
        singleProxy: {
            scheme: "http",
            host: "%s",
            port: parseInt(%s)
        },
        bypassList: ["localhost"]
        }
    };

chrome.proxy.settings.set({value: config, scope: "regular"}, function() {});

function callbackFn(details) {
    return {
        authCredentials: {
            username: "%s",
            password: "%s"
        }
    };
}

chrome.webRequest.onAuthRequired.addListener(
            callbackFn,
            {urls: ["<all_urls>"]},
            ['blocking']
);
"""


@dataclass
class Proxy:
    host: str
    port: int
    username: Optional[str] = None
    password: Optional[str] = None


def _create_proxy_extension_file(file, proxy: Proxy) -> None:
    background_js = BACKGROUND_JS % (proxy.host, proxy.port, proxy.username,
                                     proxy.password)
    with zipfile.ZipFile(file, "w") as zp:
        zp.writestr("manifest.json", MANIFEST_JSON)
        zp.writestr("background.js", background_js)


def chromedriver(headless: bool = True,
                 executable_path: Optional[str] = None,
                 binary_location: Optional[str] = None,
                 proxy: Optional[Proxy] = None,
                 user_data_dir: Optional[str] = None,
                 data_path: Optional[str] = None,
                 disk_cache_dir: Optional[str] = None,
                 profile_dir: Optional[str] = None,
                 single_process: bool = False,
                 no_zygote: bool = False,
                 exclude_switches: Tuple[str] = (),
                 disable_extensions: bool = False,
                 disable_dev_tools: bool = False,
                 user_agent_override: bool = False,
                 navigator_undefined: bool = False,
                 logging_prefs: Optional[Dict] = None) -> WebDriver:
    """
    Creates a Chrome webdriver.

    Parameters
    ----------
    headless
        Boolean indicating whether driver should be headless.
    executable_path
        Path to chromedriver executable.
    binary_location
        Path to chrome binary.
    proxy
        Optional proxy to use.
    user_data_dir
        Where to store user data. Can be used to reuse cookies, etc., from
        session to session. See https://stackoverflow.com/a/48665557/16136775
        for more details.
    data_path
        Data path.
    disk_cache_dir
        Disk cache dir.
    profile_dir
        Profile directory. Should be something like "Default", or "MyProfile".
    single_process
        Whether to run Chrome using a single process.
    no_zygote
        Whether to run chromedriver with --no-zygote flag.
    exclude_switches
        Switches to exclude.
    logging_prefs
        Logging preferences.

    Returns
    -------
    A WebDriver instance.

    Notes
    -----
    If using this function to create a Chromedriver which will run on AWS lambda,
    where you can only write to the /tmp/ directory, you must set user_data_dir,
    data_path and disk_cache_dir to a directory within /tmp. This is easily done
    using tempfile.mkdtemp. Moreover, you should set single_process and no_zygote
    to True.

    The user data, data, disk cache, profile and download directories must exist
    prior to calling this function - otherwise the webdriver will crash.

    Most of the arguments below have come from:
     https://stackoverflow.com/a/53040904/23987403exclude_switches.
    """
    options = webdriver.ChromeOptions()
    if binary_location is not None:
        options.binary_location = binary_location

    # see: https://stackoverflow.com/questions/50642308/webdriverexception
    # -unknown-error-devtoolsactiveport-file-doesnt-exist-while-t
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-dev-shm-usage")

    if disable_dev_tools:
        options.add_argument("--disable-dev-tools")

    if disable_extensions:
        options.add_argument("--disable-extensions")

    options.add_argument('--disable-blink-features=AutomationControlled')

    options.add_argument('--ignore-ssl-errors=yes')
    options.add_argument('--ignore-certificate-errors')

    # need to specify exclude switches?
    if exclude_switches:
        options.add_experimental_option("excludeSwitches", exclude_switches)

    options.add_experimental_option('useAutomationExtension', False)
    options.add_argument(f"--user-agent={USER_AGENT}")

    options.add_argument("--window-size=1920x1080")
    # set debugging port to avoid "DevToolsActivePort file doesn't exist"
    # error (see: https://stackoverflow.com/a/56638103)
    options.add_argument("--remote-debugging-port=9222")

    if single_process:
        options.add_argument("--single-process")

    if no_zygote:
        # some suggest passing --no-zygote flag is needed to run selenium on
        # AWS lambda (see: https://groups.google.com/a/chromium.org/g/headless-dev/
        # c/qqbZVZ2IwEw/m/01g5QIkRCAAJ)
        options.add_argument("--no-zygote")

    if headless:
        # use "new" headless mode by passing "--headless=new" for better performance
        # versus just passing "--headless"
        options.add_argument("--headless=new")

    if logging_prefs is not None:
        # set logging prefs using set_capability as of selenium 4.10
        # (see: https://stackoverflow.com/a/76795435/23987403)
        options.set_capability("goog:loggingPrefs", logging_prefs)

    if user_data_dir is not None:
        options.add_argument(f"--user-data-dir={user_data_dir}")

    if data_path is not None:
        options.add_argument(f"--data-path={data_path}")

    if disk_cache_dir is not None:
        options.add_argument(f"--disk-cache-dir={disk_cache_dir}")

    if profile_dir is not None:
        options.add_argument(f"--profile-directory={profile_dir}")

    if proxy is not None:
        with tempfile.NamedTemporaryFile(delete=False, dir="/tmp") as f:
            _create_proxy_extension_file(f, proxy)
            options.add_extension(f.name)

    service = webdriver.ChromeService(executable_path)
    driver = webdriver.Chrome(options=options, service=service)

    if user_agent_override:
        driver.execute_cdp_cmd('Network.setUserAgentOverride', {
            "userAgent": USER_AGENT})

    if navigator_undefined:
        driver.execute_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

    return driver


def get_performance_logs(driver) -> List[Dict]:
    """
    Gets all logs since last time driver.get_log was called.
    """
    logs = driver.get_log("performance")
    return [json.loads(x["message"])["message"] for x in logs]


def get_fetch_xhr_requests(logs: List[Dict], *, base_url: Optional[str] = None,
                           endpoint: Optional[str] = None):
    """
    Gets Fetch/ XHR requests from performance logs.
    """
    res = []
    for log in logs:
        params = log["params"]
        # add Document in type in case making direct request to API endpoint
        if (log["method"] == "Network.requestWillBeSent" and
                params["type"] in ("XHR", "Fetch", "Document")):
            url = params["request"]["url"]
            parsed_url = parse_url(url)
            if base_url is not None:
                if parsed_url.base_url != base_url:
                    continue
            if endpoint is not None:
                if parsed_url.endpoint != endpoint:
                    continue
            res.append(log)
    return res


def get_fetch_xhr_responses(logs: List[Dict], *, base_url: Optional[str] = None,
                            endpoint: Optional[str] = None):
    """
    Gets Fetch/ XHR responses from performance logs.
    """
    res = []
    for log in logs:
        params = log["params"]
        # add Document in type in case making direct request to API endpoint
        if (log["method"] == "Network.responseReceived" and
                params["type"] in ("XHR", "Fetch", "Document")):
            url = params["response"]["url"]
            parsed_url = parse_url(url)
            if base_url is not None:
                if parsed_url.base_url != base_url:
                    continue
            if endpoint is not None:
                if parsed_url.endpoint != endpoint:
                    continue
            res.append(log)
    return res


# deprecated, use get_fetch_xhr_requests instead
def get_xhr_requests(logs: List[Dict], *, base_url: Optional[str] = None,
                     endpoint: Optional[str] = None):
    return get_fetch_xhr_requests(logs, base_url=base_url, endpoint=endpoint)


def element_exists(driver: WebDriver, locator: Any, timeout=10) -> bool:
    try:
        WebDriverWait(driver, timeout).until(EC.presence_of_element_located(locator))
        return True
    except TimeoutException:
        return False


def perimeterx_click_and_hold(driver) -> None:
    captcha_xpath = "//div[contains(@class, 'px-captcha-header')]"
    if element_exists(driver, (By.XPATH, captcha_xpath), timeout=5):
        logger.info("Attempting to solve PerimeterX challenge")
        action = ActionChains(driver)
        action.send_keys(Keys.TAB)
        action.pause(1)
        action.key_down(Keys.ENTER)
        action.pause(10)
        action.key_up(Keys.ENTER)
        action.perform()
        time.sleep(20)
