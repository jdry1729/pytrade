import logging
from typing import Iterable, Optional

import ray


@ray.remote
def wait(*args) -> Iterable:
    return args


# below ensures logs from child tasks are created
def _worker_setup_fn():
    logging.basicConfig(
        format="[%(asctime)s] %(levelname)-4s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO)


def ray_init(address: Optional[str] = None, *,
             num_cpus: Optional[int] = None,
             py_modules: Optional[Iterable[str]] = None,
             pip: Optional[Iterable[str]] = None):
    # runtime env below will be merged with any runtime env specified in
    # ray job submit
    ray.init(address, num_cpus=num_cpus, runtime_env={
        "worker_process_setup_hook": _worker_setup_fn,
        "py_modules": py_modules,
        "pip": pip
    })
