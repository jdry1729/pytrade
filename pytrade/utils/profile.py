import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import yaml

PYTRADE_CONFIG_PATH_ENV_VAR = "PYTRADE_CONFIG_PATH"
PYTRADE_PROFILE_ENV_VAR = "PYTRADE_PROFILE"

DEFAULT_PYTRADE_DIR = os.path.expanduser("~/.pytrade")
DEFAULT_PYTRADE_CONFIG_PATH = os.path.join(DEFAULT_PYTRADE_DIR, "config.yml")

POSTGRES_HOST_ENV_VAR = "PYTRADE_POSTGRES_HOST"
POSTGRES_PORT_ENV_VAR = "PYTRADE_POSTGRES_PORT"
POSTGRES_USER_ENV_VAR = "PYTRADE_POSTGRES_USER"
POSTGRES_PASSWORD_ENV_VAR = "PYTRADE_POSTGRES_PASSWORD"
POSTGRES_DATABASE_ENV_VAR = "PYTRADE_POSTGRES_DATABASE"
ARCTIC_URI_ENV_VAR = "PYTRADE_ARCTIC_URI"
ARCTIC_DEFAULT_LIB_ENV_VAR = "PYTRADE_ARCTIC_DEFAULT_LIB"
S3_ACCESS_KEY_ENV_VAR = "PYTRADE_S3_ACCESS_KEY"
S3_SECRET_ENV_VAR = "PYTRADE_S3_SECRET"
S3_ENDPOINT_ENV_VAR = "PYTRADE_S3_ENDPOINT"
IB_GATEWAY_HOST_ENV_VAR = "PYTRADE_IB_GATEWAY_HOST"
IB_GATEWAY_PORT_ENV_VAR = "PYTRADE_IB_GATEWAY_PORT"
IB_ACCOUNT_ID_ENV_VAR = "PYTRADE_IB_ACCOUNT_ID"
OPSGENIE_API_KEY_ENV_VAR = "PYTRADE_OPSGENIE_API_KEY"

logger = logging.getLogger(__name__)

# arctic S3 URI example
# s3://{host}:{bucket}?access={access_key}&secret={secret}&port={port}


ENV_VAR_MAP = {
    "postgres_host": POSTGRES_HOST_ENV_VAR,
    "postgres_port": POSTGRES_PORT_ENV_VAR,
    "postgres_user": POSTGRES_USER_ENV_VAR,
    "postgres_password": POSTGRES_PASSWORD_ENV_VAR,
    "postgres_database": POSTGRES_DATABASE_ENV_VAR,
    "arctic_uri": ARCTIC_URI_ENV_VAR,
    "arctic_default_lib": ARCTIC_DEFAULT_LIB_ENV_VAR,
    "s3_access_key": S3_ACCESS_KEY_ENV_VAR,
    "s3_secret": S3_SECRET_ENV_VAR,
    "s3_endpoint": S3_ENDPOINT_ENV_VAR,
    "ib_gateway_host": IB_GATEWAY_HOST_ENV_VAR,
    "ib_gateway_port": IB_GATEWAY_PORT_ENV_VAR,
    "ib_account_id": IB_ACCOUNT_ID_ENV_VAR,
    "opsgenie_api_key": OPSGENIE_API_KEY_ENV_VAR,
}


@dataclass
class Profile:
    """
    arctic_uri
        Arctic URI. E.g., s3://{host}:{bucket}?access={access_key}&secret=
        {secret}&port={port}.
    """
    name: Optional[str] = None

    # TODO: replace with postgres_uri?
    postgres_host: Optional[str] = "localhost"
    postgres_port: Optional[int] = 5432
    postgres_user: Optional[str] = "postgres"
    postgres_password: Optional[str] = "password"
    postgres_database: Optional[str] = "trading"

    # TODO: might need to have s3 "section" with access key/ secret
    s3_endpoint: Optional[str] = None
    s3_access_key: Optional[str] = None
    s3_secret: Optional[str] = None

    # TODO: constrain arctic to s3 for which credentials are specified above?
    #  then we could just have options here to specify path to arctic directory
    #  within the bucket
    arctic_uri: Optional[str] = "lmdb://tmp/arctic"
    arctic_default_lib: Optional[str] = None

    ib_gateway_host: Optional[str] = "localhost"
    ib_gateway_port: Optional[int] = 5000
    ib_account_id: Optional[str] = None

    opsgenie_api_key: Optional[str] = None

    @property
    def ib_gateway_uri(self):
        return f"https://{self.ib_gateway_host}:{self.ib_gateway_port}"


def get_pytrade_config_path():
    return os.environ.get(PYTRADE_CONFIG_PATH_ENV_VAR,
                          DEFAULT_PYTRADE_CONFIG_PATH)


@lru_cache(maxsize=None)
def get_pytrade_config():
    config_path = get_pytrade_config_path()
    if os.path.exists(config_path):
        with open(config_path, "rb") as f:
            return yaml.safe_load(f)
    raise ValueError(f"Config file doesn't exist: {config_path}")


def _load_profile_from_config_file(name: str) -> Profile:
    config = get_pytrade_config()
    if name in config:
        kwargs = {}
        logger.debug(f"Loading profile: {name}")
        config = config[name]
        for arg, env_var in ENV_VAR_MAP.items():
            # TODO: don'y use env variables if loading named profile?
            kwargs[arg] = os.environ.get(env_var, config.get(arg))
        return Profile(name=name, **kwargs)
    raise ValueError(f"Error loading profile: {name}; profile"
                     f" doesn't exist")


def _load_profile_from_env() -> Profile:
    kwargs = {}
    for arg, env_var in ENV_VAR_MAP.items():
        kwargs[arg] = os.environ.get(env_var)
    return Profile(**kwargs)


def load_profile(name: Optional[str] = None) -> Profile:
    if name is None:
        name = os.getenv(PYTRADE_PROFILE_ENV_VAR)
    if name is None:
        return _load_profile_from_env()
    return _load_profile_from_config_file(name)
