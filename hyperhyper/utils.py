"""
utility functions for i/o and other general funtionality
"""

import logging
import os
import pickle
from collections import defaultdict
from concurrent import futures

import numpy as np
from scipy.sparse import csr_matrix, load_npz, save_npz
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _default_workers():
    """
    Number of usable CPUs, respecting affinity/cgroup limits where available.
    """
    return getattr(os, "process_cpu_count", os.cpu_count)() or 1


def save_arrays(f, a1, a2):
    """
    Save two numpy arrays to a single compressed ``.npz`` file at ``f``.

    Used to persist the SVD outputs ``(ut, s)``. ``f`` may be a path-like
    object; numpy appends the ``.npz`` extension. Read back with ``load_arrays``.
    """
    if not isinstance(f, str):
        f = str(f)
    np.savez_compressed(f, a1=a1, a2=a2)


def load_arrays(f):
    """
    Load the two arrays written by ``save_arrays`` from ``f``.

    Accepts a path with or without the ``.npz`` extension (it is appended if
    missing) and returns the arrays as a ``(a1, a2)`` tuple.
    """
    if not isinstance(f, str):
        f = str(f)
    if not f.endswith(".npz"):
        f += ".npz"
    loader = np.load(f)
    return loader["a1"], loader["a2"]


def save_matrix(f, m):
    """
    Save a scipy sparse matrix ``m`` to a compressed ``.npz`` file at ``f``.

    ``f`` may be a path-like object. Read back with ``load_matrix``.
    """
    if not isinstance(f, str):
        f = str(f)
    save_npz(f, m, compressed=True)


def load_matrix(f):
    """
    Load a scipy sparse matrix from ``f`` (as written by ``save_matrix``).

    Accepts a path with or without the ``.npz`` extension (appended if missing).
    First tries scipy's ``load_npz``; if that fails with ``ValueError`` (an
    older file written in the legacy, hand-rolled CSR layout with separate
    ``data``/``indices``/``indptr``/``shape`` arrays), it falls back to
    reconstructing a ``csr_matrix`` from those arrays.
    """
    if not isinstance(f, str):
        f = str(f)
    if not f.endswith(".npz"):
        f += ".npz"
    try:
        return load_npz(f)
    except ValueError:
        # fall back to the legacy, hand-rolled layout (CSR only)
        loader = np.load(f)
        return csr_matrix(
            (loader["data"], loader["indices"], loader["indptr"]),
            shape=loader["shape"],
        )


def chunks(seq, n):
    """
    Yield successive n-sized chunks from seq.
    """
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def map_pool(array, fun, total=None, desc=None, process_chunksize=100):
    """
    Apply `fun` to every item of `array` across a process pool.

    `process_chunksize` is the batching that matters: `Executor.map` hands each
    worker that many items per task and flattens the results incrementally, so
    batching the input by hand beforehand only bundles twice and serialises the
    work back onto one worker.
    """
    with futures.ProcessPoolExecutor(_default_workers()) as executor:
        if desc is None:
            return list(executor.map(fun, array, chunksize=process_chunksize))
        return list(
            tqdm(
                executor.map(fun, array, chunksize=process_chunksize),
                total=len(array) if total is None else total,
                desc=desc,
            )
        )


def delete_folder(pth):
    """
    Recursively delete the directory ``pth`` and everything inside it.

    ``pth`` is a ``pathlib.Path``. Walks the tree, unlinking files and
    recursing into subdirectories, then removes the now-empty directory itself.
    """
    for sub in pth.iterdir():
        if sub.is_dir():
            delete_folder(sub)
        else:
            sub.unlink()
    pth.rmdir()


def to_pickle(ob, fn):
    """
    Pickle object ``ob`` to the file ``fn``, creating parent directories.

    ``fn`` is a ``pathlib.Path``; any missing parent directories are created.
    """
    fn.parent.mkdir(parents=True, exist_ok=True)
    with open(fn, "wb") as outfile:
        pickle.dump(ob, outfile)


def read_pickle(fn):
    """
    Unpickle and return the object stored in the file ``fn``.
    """
    with open(fn, "rb") as infile:
        return pickle.load(infile)


def dsum(*dicts):
    """
    sum up numerical values in multiple dictionaries
    """
    ret = defaultdict(int)
    for d in dicts:
        for k, v in d.items():
            ret[k] += v
    return dict(ret)
