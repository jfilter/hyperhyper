import os

import numpy as np
from gensim.utils import flatten
from joblib import Parallel, delayed
from scipy.sparse import csr_matrix
from tqdm import tqdm

num_cpu = os.cpu_count()


def save_matrix(f, m):
    if type(f) != str:
        f = str(f)
    np.savez_compressed(
        f, data=m.data, indices=m.indices, indptr=m.indptr, shape=m.shape
    )


def load_matrix(f):
    if type(f) != str:
        f = str(f)
    # if not f.endswith(".npz"):
    #     f += ".npz"
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


def combine_chunks(chunks):
    for c in chunks:
        for x in c:
            yield x


def map_chunks(array, fun, chunk_size=100000, **kwargs):
    results = Parallel(n_jobs=num_cpu)(
        delayed(fun)(x)
        for x in tqdm(
            chunks(array, chunk_size), total=len(array) / chunk_size, **kwargs
        )
    )
    results = list(combine_chunks(results))
    return results
