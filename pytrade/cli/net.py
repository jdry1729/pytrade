import logging

import click

logger = logging.getLogger(__name__)


@click.group()
def net():
    pass


@net.group()
def cookies():
    pass


@cookies.command()
@click.option("--out", type=str)
@click.option("--user-data-dir", type=str)
@click.option("--profile-dir", type=str)
def dump(out: str, user_data_dir: str, profile_dir: str):
    from pytrade.data.fs import write_json
    from pytrade.net.utils import get_chrome_cookies, \
        convert_chrome_cookies_to_webdriver_format

    cookies = get_chrome_cookies(f"{user_data_dir}/{profile_dir}/Cookies")
    cookies = convert_chrome_cookies_to_webdriver_format(cookies)

    write_json(out, cookies)
