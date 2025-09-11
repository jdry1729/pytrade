import multiprocessing

import dill


def dill_mp_setup_fn():
    """
    Causes multiprocessing library to use dill to pickle objects. One of the
    benefits of this is that it allows multiprocessing to work with lambda
    functions.
    """
    dill.Pickler.dumps, dill.Pickler.loads = dill.dumps, dill.loads
    multiprocessing.reduction.ForkingPickler = dill.Pickler
    multiprocessing.reduction.dump = dill.dump
    multiprocessing.queues._ForkingPickler = dill.Pickler
