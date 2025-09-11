import hashlib
import logging
import os
import shutil
import time
from datetime import timedelta
from glob import glob
from pathlib import Path
from typing import Optional, List

from pytrade.data.fs import write_lines, read_lines

logger = logging.getLogger(__name__)


def compute_sha256_file_hash(path: str):
    with open(path, "rb") as f:
        file_hash = hashlib.sha256()
        while chunk := f.read(8192):
            file_hash.update(chunk)
    return file_hash.hexdigest()


def get_file_paths(glob_pattern: str):
    file_paths = []
    for file_path in sorted(glob(glob_pattern, recursive=True)):
        file_path = os.path.abspath(file_path)
        if os.path.isfile(file_path):
            file_paths.append(file_path)
    return file_paths


def ensure_dir_exists(path: str):
    if not os.path.exists(path):
        logger.info(f"Creating directory: {path}")
        os.makedirs(path)


def delete_dir_if_exists(path: str):
    if os.path.exists(path):
        logger.info(f"Deleting directory: {path}")
        shutil.rmtree(path)


def get_file_name(file_path):
    return Path(file_path).name


# TODO: deprecetate
def read_file_to_list(path: str):
    return read_lines(path)


# TODO: deprecetate
def write_list_to_file(path: str, l: List):
    write_lines(path, l)


def wait_for_file(path: str, timeout: Optional[timedelta] = None,
                  poll_freq: timedelta = timedelta(seconds=0.5)) -> None:
    time_slept = timedelta()
    poll_seconds = poll_freq.total_seconds()

    while True:
        if os.path.exists(path):
            break
        time.sleep(poll_seconds)
        time_slept += poll_freq
        if timeout is not None and time_slept >= timeout:
            raise ValueError("Error waiting for file; time out")
