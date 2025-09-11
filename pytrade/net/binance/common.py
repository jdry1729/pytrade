import os
from dataclasses import dataclass
from typing import Optional

BINANCE_API_KEY_ENV_VAR = "PYTRADE_BINANCE_API_KEY"
BINANCE_API_SECRET_ENV_VAR = "PYTRADE_BINANCE_API_SECRET"


@dataclass
class Credentials:
    api_key: str
    api_secret: str


def resolve_credentials(credentials: Optional[Credentials] = None):
    if credentials is not None:
        return credentials

    api_key = os.getenv(BINANCE_API_KEY_ENV_VAR)
    api_secret = os.getenv(BINANCE_API_KEY_ENV_VAR)
    if api_key is not None and api_secret is not None:
        return Credentials(api_key, api_secret)

    raise ValueError("Error resolving Binance credentials")
