import concurrent.futures
import logging
import math
import os
import pickle
from collections import defaultdict

import numpy as np
from gensim.utils import flatten
from scipy.sparse import csr_matrix, dok_matrix
from tqdm import tqdm

num_cpu = os.cpu_count()

logger = logging.getLogger(__name__)


def save_arrays(f, a1, a2):
    if type(f) != str:
        f = str(f)
    np.savez_compressed(f, a1=a1, a2=a2)


def load_arrays(f):
    if type(f) != str:
        f = str(f)
    if not f.endswith(".npz"):
        f += ".npz"
    loader = np.load(f)
    return loader["a1"], loader["a2"]


def save_matrix(f, m):
    if type(f) != str:
        f = str(f)
    np.savez_compressed(
        f, data=m.data, indices=m.indices, indptr=m.indptr, shape=m.shape
    )


def load_matrix(f):
    if type(f) != str:
        f = str(f)
    if not f.endswith(".npz"):
        f += ".npz"
    loader = np.load(f)
    return csr_matrix(
        (loader["data"], loader["indices"], loader["indptr"]), shape=loader["shape"]
    )


def chunks(l, n):
    """
    Yield successive n-sized chunks from l.
    """
    for i in range(0, len(l), n):
        yield l[i : i + n]


# TODO: more perfz
def combine_chunks(chunks):
    for c in chunks:
        for x in c:
            yield x


def map_pool_chunks(
    array, fun, num_chunks=100, chunk_size=None, combine=True, **kwargs
):
    if chunk_size is None:
        chunk_size = math.ceil(len(array) / num_chunks)
    results = map_pool(chunks(array, chunk_size), fun, total=len(array), **kwargs)
    if combine:
        results = list(combine_chunks(results))
    return results


def map_pool(array, fun, total=None, desc=None, process_chunksize=100):
    with concurrent.futures.ProcessPoolExecutor(num_cpu) as executor:
        results = list(
            tqdm(
                executor.map(fun, array, chunksize=process_chunksize),
                total=len(array) if total is None else total,
                desc=desc,
            )
        )
    return results


def delete_folder(pth):
    for sub in pth.iterdir():
        if sub.is_dir():
            delete_folder(sub)
        else:
            sub.unlink()
    pth.rmdir()

def to_pickle(ob, fn):
    fn.parent.mkdir(parents=True, exist_ok=True)
    with open(fn, "wb") as outfile:
        pickle.dump(ob, outfile)

def read_pickle(fn):
    with open(fn, "rb") as infile:
        return pickle.load(infile)


def dsum(*dicts):
    ret = defaultdict(int)
    for d in dicts:
        for k, v in d.items():
            ret[k] += v
    return dict(ret)
