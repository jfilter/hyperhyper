import os
import random
from collections import Counter, defaultdict
from math import fabs, sqrt

import numpy as np
from gensim.utils import SaveLoad
from joblib import Parallel, delayed
from scipy.sparse import csr_matrix, dok_matrix
from tqdm import tqdm

from .utils import map_chunks

num_cpu = os.cpu_count()


def to_count_matrix(pair_counts, vocab_size):
    """
    Reads the counts into a sparse matrix (CSR) from the count-word-context textual format.
    """
    num_words = vocab_size + 1  # for UKN

    counts = csr_matrix((num_words, num_words), dtype=np.float32)
    tmp_counts = dok_matrix((num_words, num_words), dtype=np.float32)
    update_threshold = 100000
    i = 0
    for k, v in tqdm(pair_counts.items(), desc="transform counts to matrix"):
        # increase by 1 because the token for unknown words is -1
        tmp_counts[k[0] + 1, k[1] + 1] = v
        i += 1
        if i == update_threshold:
            counts = counts + tmp_counts.tocsr()
            tmp_counts = dok_matrix((num_words, num_words), dtype=np.float32)
            i = 0

    counts = counts + tmp_counts.tocsr()
    return counts


class PairCounts(SaveLoad):
    counter = defaultdict(int)

    def __getitem__(self, key):
        return self.counter[key]

    def __setitem__(self, key, value):
        self.counter[key] = value

    def items(self):
        return self.counter.items()


def iterate_tokens(
    tokens,
    window,
    dynamic_window,
    weighted_window,
    delete_oov,
    subsample_prob,
    subsample_deter,
    subsampler,
):
    if delete_oov:
        tokens = [t for t in tokens if t != -1]  # -1 is the OOV token id

    if subsample_prob:
        tokens = [
            t if t not in subsampler or random.random() <= subsampler[t] else None
            for t in tokens
        ]

    len_tokens = len(tokens)
    pairs = []
    for i, tok in enumerate(tokens):
        if tok is not None:
            # This is described differently in the paper. However, on scale the outcome should be the same. Instead of weight
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
                    if subsample_deter:
                        count = subsampler[tok] * subsampler[tokens[j]] * count
                    pairs.append((tok, tokens[j], count))
    return pairs


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
    n_jobs=1,
):
    if dynamic_window and weighted_window:
        raise ValueError("Dynamic and weighted window options are exclusive!")
    if subsample_prob and subsample_deter:
        raise ValueError("Subsampling options are exclusive!")

    random.seed(seed)

    subsampler = None
    if subsample_prob or subsample_deter:
        # original implementation

        if subsample_prob:
            if subsample_factor is None:
                subsampler = 1e-5
            subsampler *= corpus.size
            subsampler = {
                word: 1 - sqrt(subsampler / count)
                for word, count in corpus.vocab.items()
                if count > subsampler
            }
        else:
            if subsample_factor is None:
                subsampler = 1e-4
            subsampler *= corpus.size
            subsampler = defaultdict(
                lambda: 1,
                {
                    word: sqrt(subsampler / count)
                    for word, count in corpus.vocab.items()
                    if count > subsampler
                },
            )

    # map
    def fun(texts):
        return [
            iterate_tokens(
                t,
                window,
                dynamic_window,
                weighted_window,
                delete_oov,
                subsample_prob,
                subsample_deter,
                subsampler,
            )
            for t in texts
        ]

    results = map_chunks(corpus.texts, fun, desc="generating pairs")

    # reduce
    counter = PairCounts()
    for r in tqdm(results, desc="counting pairs"):
        for p in r:
            counter[(p[0], p[1])] += p[2]

    return to_count_matrix(counter, corpus.vocab.size)
