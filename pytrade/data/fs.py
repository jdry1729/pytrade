import json
from typing import Optional, List, Dict, Union, Iterable

import pandas as pd
import s3fs

from pytrade.utils.profile import load_profile


def _get_storage_options(profile: Optional[str] = None) -> Dict:
    profile = load_profile(profile)
    return {
        "key": profile.s3_access_key,
        "secret": profile.s3_secret,
        "client_kwargs": {"endpoint_url": profile.s3_endpoint}}


def read_lines(path: str, profile: Optional[str] = None):
    if path.startswith("s3://"):
        s3 = s3fs.S3FileSystem(**_get_storage_options(profile))
        with s3.open(path, "r") as f:
            return f.read().splitlines()
    with open(path) as f:
        return f.read().splitlines()


def write_lines(path: str, l: List, profile: Optional[str] = None):
    if path.startswith("s3://"):
        s3 = s3fs.S3FileSystem(**_get_storage_options(profile))
        with s3.open(path, "w") as f:
            return f.writelines([f"{x}\n" for x in l])
    with open(path, "w") as f:
        f.writelines([f"{x}\n" for x in l])


def read_json(path: str):
    with open(path, "r") as file:
        return json.load(file)


def write_json(path: str, data):
    with open(path, "w") as file:
        json.dump(data, file, indent=4)


def read_csv(path: str, profile: Optional[str] = None, **kwargs):
    if path.startswith("s3://"):
        return pd.read_csv(path, storage_options=_get_storage_options(profile),
                           **kwargs)
    return pd.read_csv(path, **kwargs)


def write_csv(path: str, data: pd.DataFrame, profile: Optional[str] = None, **kwargs):
    if path.startswith("s3://"):
        return data.to_csv(path, storage_options=_get_storage_options(profile),
                           **kwargs)
    return data.to_csv(path, **kwargs)


def read_excel(path: str, sheet_name: Union[str, Iterable[str]] = 0,
               profile: Optional[str] = None, squeeze: bool = False, **kwargs):
    if path.startswith("s3://"):
        data = pd.read_excel(path, sheet_name,
                             storage_options=_get_storage_options(profile),
                             **kwargs)
    else:
        data = pd.read_excel(path, sheet_name, **kwargs)
    if squeeze:
        data = data.squeeze(axis=1)
    return data


def write_excel(path: str, data: pd.DataFrame,
                sheet_name: Union[str, Iterable[str]] = 0,
                profile: Optional[str] = None, **kwargs):
    if path.startswith("s3://"):
        return data.to_excel(path, sheet_name,
                             storage_options=_get_storage_options(profile),
                             **kwargs)
    return data.to_excel(path, sheet_name, **kwargs)
