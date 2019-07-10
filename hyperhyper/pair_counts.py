import random
from collections import Counter, defaultdict
from math import fabs, sqrt, ceil
import logging

from array import array

import math
import numpy as np
from gensim.utils import SaveLoad
from joblib import Parallel, delayed
from scipy.sparse import csr_matrix, dok_matrix, coo_matrix
from tqdm import tqdm
import concurrent.futures


from .utils import chunks

logger = logging.getLogger(__name__)


class PairCounts(SaveLoad):
    counter = defaultdict(int)

    def __getitem__(self, key):
        return self.counter[key]

    def __setitem__(self, key, value):
        self.counter[key] = value

    def items(self):
        return self.counter.items()

    def __str__(self):
        return str(self.counter)


def to_count_matrix(pair_counts, vocab_size):
    """
    Reads the counts into a sparse matrix (CSR) from the count-word-context textual format.
    """
    cols = []
    rows = []
    data = []
    for k, v in tqdm(pair_counts.items(), desc="transform counts to matrix"):
        rows.append(k[0])
        cols.append(k[1])
        data.append(v)
    # why is float32 so important?
    count_matrix = coo_matrix(
        (data, (rows, cols)), shape=(vocab_size + 1, vocab_size + 1), dtype=np.float32
    )
    logger.info(f"num non-zeros: {count_matrix.count_nonzero()}")
    return count_matrix


def count_pars_map(array, fun, num_chunks=100, chunk_size=None, desc=None):
    if chunk_size is None:
        # default to 100 chunks
        chunk_size = math.ceil(len(array) / num_chunks)

    total = math.ceil(len(array) / chunk_size)
    array = chunks(array, chunk_size)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        res = defaultdict(int)
        for x in tqdm(executor.map(fun, array, chunksize=1), desc=desc, total=total):
            for key, value in x.items():
                res[key] += value
    return res


def iterate_tokens(
    tokens,
    window,
    dynamic_window,
    weighted_window,
    delete_oov,
    subsampler_prob,
    unkown_id,
):
    if delete_oov:
        tokens = [t for t in tokens if t != unkown_id]

    if not subsampler_prob is None:
        tokens = [
            t
            if t not in subsampler_prob or random.random() <= subsampler_prob[t]
            else None
            for t in tokens
        ]

    len_tokens = len(tokens)
    res = []
    for i, tok in enumerate(tokens):
        if tok is not None:
            if dynamic_window:
                offset = random.randint(1, window)
            else:
                offset = window
            start = i - offset
            if start < 0:
                start = 0
            end = i + offset + 1
            if end > len_tokens:
                end = len_tokens
            for j in range(start, end):
                if j != i and tokens[j] is not None:
                    if weighted_window:
                        distance = fabs(i - j)
                        count = (window + 1 - distance) / window
                    else:
                        count = 1
                    res.append((tok, tokens[j], count))
    return res


class CountPairsClosure(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __call__(self, texts):

        counter = defaultdict(int)
        bla = []
        for t in texts:
            bla.append(
                list(
                    iterate_tokens(
                        t,
                        self.window,
                        self.dynamic_window,
                        self.weighted_window,
                        self.delete_oov,
                        self.subsampler_prob,
                        self.vocab_size,  # <UKN> id
                    )
                )
            )

        for b in bla:
            for pair in b:
                counter[pair[0], pair[1]] += pair[2]
        return counter


def count_pairs(
    corpus,
    window=2,
    dynamic_window=False,
    weighted_window=False,
    delete_oov=False,
    subsample_prob=False,
    subsample_deter=False,
    subsample_factor=None,  # defaults later on
    seed=1312,
    num_chunks=1000,
):
    if dynamic_window and weighted_window:
        raise ValueError("Dynamic and weighted window options are exclusive!")
    if subsample_prob and subsample_deter:
        raise ValueError("Subsampling options are exclusive!")

    random.seed(seed)

    subsampler_prob = None
    if subsample_prob:
        # original implementation

        if subsample_prob:
            if subsample_factor is None:
                subsample_factor = 1e-5
            subsampler_prob = subsample_factor * corpus.size
            subsampler_prob = {
                word: 1 - sqrt(subsampler_prob / count)
                for word, count in corpus.counts.items()
                if count > subsampler_prob
            }

    fun = CountPairsClosure(
        window=window,
        dynamic_window=dynamic_window,
        weighted_window=weighted_window,
        delete_oov=delete_oov,
        subsampler_prob=subsampler_prob,
        vocab_size=corpus.vocab.size,
    )

    count_dic = count_pars_map(
        corpus.texts, fun, desc="generating pairs", num_chunks=num_chunks
    )
    count_matrix = to_count_matrix(count_dic, corpus.vocab.size)

    # down sample in a deterministic way
    if subsample_deter:
        if subsample_factor is None:
            subsample_factor = 1e-4
        subsampler = subsample_factor * corpus.size
        subsampler = defaultdict(
            lambda: 1,
            {
                word: sqrt(subsampler / count)
                for word, count in corpus.counts.items()
                if count > subsampler
            },
        )
        # coo for interation but access with csr
        cx = count_matrix
        count_matrix = count_matrix.tocsr()
        for i, j, value in zip(cx.row, cx.col, cx.data):
            count_matrix[(i, j)] = subsampler[i] * subsampler[j] * value

    return count_matrix.tocsr()